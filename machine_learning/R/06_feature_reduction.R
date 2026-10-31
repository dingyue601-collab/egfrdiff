source("shared/R/config.R")
options(stringsAsFactors = FALSE)
suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(jsonlite))
setDTthreads(1L)
set.seed(123)
root <- getwd()
processed <- file.path(root, "results")
source <- file.path(processed, "models")
data_source <- file.path(processed, "ml_imputation")
out <- file.path(processed, "reduction")
args <- commandArgs(TRUE)
mode <- if (length(args)) args[1L] else "prepare"
write_json_file <- function(x, f) write_json(x, file.path(out, f), pretty = TRUE, auto_unbox = TRUE, null = "null", 
    digits = 16)
read_json_file <- function(f) read_json(file.path(out, f), simplifyVector = TRUE)
note <- function(...) cat(format(Sys.time(), "%H:%M:%S"), ..., "\n")
scores <- function(y, p, calibration = TRUE) {
    stopifnot(length(y) == length(p), all(y %in% c(0, 1)), all(is.finite(p)), all(p >= 0 & p <= 1))
    n1 <- sum(y == 1)
    n0 <- sum(y == 0)
    auc <- if (n1 * n0) 
        (sum(rank(p)[y == 1]) - n1 * (n1 + 1)/2)/(n1 * n0)
    else NA_real_
    z <- data.table(y = y, p = p)[, .(Positive = sum(y), N = .N), by = p][order(-p)]
    ap <- if (n1) 
        sum(z$Positive/n1 * cumsum(z$Positive)/cumsum(z$N))
    else NA_real_
    tp <- as.double(sum(y == 1 & p >= 0.5))
    fp <- as.double(sum(y == 0 & p >= 0.5))
    fn <- as.double(sum(y == 1 & p < 0.5))
    tn <- as.double(sum(y == 0 & p < 0.5))
    safe <- function(a, b) if (b > 0) 
        a/b
    else NA_real_
    result <- c(AUC = auc, AP = ap, Brier = mean((y - p)^2), Accuracy = (tp + tn)/length(y), Sensitivity = safe(tp, 
        tp + fn), Specificity = safe(tn, tn + fp), PPV = safe(tp, tp + fp), NPV = safe(tn, tn + fn), F1 = safe(2 * 
        tp, 2 * tp + fp + fn), MCC = safe(tp * tn - fp * fn, sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))), 
        TP = tp, FP = fp, FN = fn, TN = tn)
    if (calibration) {
        lp <- qlogis(pmin(pmax(p, 1e-07), 1 - 1e-07))
        intercept <- suppressWarnings(glm(y ~ 1 + offset(lp), family = binomial()))
        slope <- suppressWarnings(glm(y ~ lp, family = binomial()))
        result <- c(result, CalibrationIntercept = unname(coef(intercept)[1]), CalibrationSlope = unname(coef(slope)[2]))
    }
    result
}
if (mode == "prepare") {
    dir.create(out, recursive = TRUE, showWarnings = FALSE)
    p <- readRDS(paste0(source, "_MODEL_PREPARATION.rds"))
    manifest <- read_json(paste0(source, "_TAB_MANIFEST.json"), simplifyVector = FALSE)
    model_choice <- readRDS(file.path(processed, "selection/Training_model_selection.rds"))
    stopifnot(model_choice$winner == "TabM")
    selected <- read_json(paste0(source, "_TabM_TUNING.json"), simplifyVector = TRUE)
    a <- readRDS(paste0(data_source, "_inputs.rds"))
    stopifnot(length(p$features) > 0L, length(p$train_y) == 6852L, nrow(p$evaluation_keys) ==
        3038L, uniqueN(p$train_keys$PatientID) == 2056L, uniqueN(p$evaluation_keys$PatientID) == 881L, !length(intersect(p$train_keys$PatientID, 
        p$evaluation_keys$PatientID)), identical(as.character(unlist(manifest$blocks$Full$features)), p$features))
    fold <- integer(6852L)
    for (k in 1:5) {
        block <- p$blocks[[paste0("Fold", k)]]
        stopifnot(!length(intersect(p$train_keys$PatientID[block$train_rows], p$train_keys$PatientID[block$query_rows])))
        fold[block$query_rows] <- k
        permutations <- as.data.table(replicate(3L, sample.int(length(block$query_rows)) - 1L))
        setnames(permutations, paste0("Repeat", 1:3))
        fwrite(permutations, file.path(out, sprintf("Training_PFI_Fold%d_permutations.csv", k)))
    }
    stopifnot(all(fold %in% 1:5), all(vapply(split(fold, p$train_keys$PatientID), uniqueN, 0L) == 1L))
    train <- data.table(Row = seq_along(p$train_y), Fold = fold, Label = p$train_y, Cluster = match(p$train_keys$PatientID, 
        unique(p$train_keys$PatientID)))
    stopifnot(sum(train$Label) == 1514L)
    fwrite(train, file.path(out, "Training_reference_v1.csv"))
    saveRDS(list(train = train), file.path(out, "Statistical_reference_v1.rds"))
    job <- list(seed = 123, source_manifest = paste0(source, "_TAB_MANIFEST.json"), source_prefix = source, output_dir = out, 
        features = p$features, cv_imputations = 1:5, cv_folds = 1:5, final_imputations = 30L, permutation_repeats = 3L, 
        TabM_parameters = selected$parameters, TabM_epochs = selected$epochs, loss_tolerances = c(0.01, 0.02), 
        ranking = "Training-fold permutation mean AUC decrease; pooled five MI before scoring", deletion = "One fixed ranking path per model,21 to1; preserve original input order in every subset", 
        selection = "Smallest subset with cumulative mean-fold AUC loss <= tolerance versus same-procedure Full21", 
        imputation_boundary = "Reduced-input analyses retain the original imputation auxiliaries",
        evaluation_status = "Patient-separated internal validation")
    write_json_file(job, "Reduction_job_v1.json")
    note("PREPARE_PASS TRAIN6852 EVALUATION3038 FIVE_PATIENT_FOLDS FIVE_MI_CV THIRTY_MI_FINAL")
    quit(save = "no")
}
job <- read_json_file("Reduction_job_v1.json")
reference <- readRDS(file.path(out, "Statistical_reference_v1.rds"))
score_set <- function(family, tag) {
    d <- fread(file.path(out, paste0(family, "_", tag, "_OOF.csv")))
    stopifnot(identical(d$Row, reference$train$Row), identical(d$Fold, reference$train$Fold), ncol(d) == 7L)
    probability <- rowMeans(as.matrix(d[, paste0("m", 1:5), with = FALSE]))
    folds <- rbindlist(lapply(1:5, function(k) {
        ix <- d$Fold == k
        as.data.table(c(list(Fold = k), as.list(scores(reference$train$Label[ix], probability[ix]))))
    }))
    spec <- read_json_file(paste0(family, "_", tag, "_spec.json"))
    result <- list(Model = family, Tag = tag, NFeatures = length(spec$features), Features = spec$features, MeanFoldAUC = mean(folds$AUC), 
        PooledOOF = as.list(scores(reference$train$Label, probability)), FoldScores = folds)
    write_json_file(result, paste0(family, "_", tag, "_CV_score.json"))
    note(family, tag, "N", length(spec$features), "MEAN_FOLD_AUC", round(result$MeanFoldAUC, 6))
    invisible(result)
}
if (mode == "score") {
    stopifnot(length(args) == 3L)
    score_set(args[2], args[3])
    quit(save = "no")
}
if (mode == "rank") {
    family <- args[2]
    d <- fread(file.path(out, paste0(family, "_Training_PFI_predictions_v1.csv")))
    stopifnot(identical(d$Row, reference$train$Row), identical(d$Fold, reference$train$Fold), ncol(d) == 66L)
    rows <- list()
    for (v in job$features) for (k in 1:5) for (r in 1:3) {
        ix <- d$Fold == k
        y <- reference$train$Label[ix]
        base <- scores(y, d$Baseline[ix], FALSE)
        perturbed <- scores(y, d[[paste0(v, "__r", r)]][ix], FALSE)
        rows[[length(rows) + 1L]] <- data.table(Feature = v, Fold = k, Repeat = r, DeltaAUC = base["AUC"] - perturbed["AUC"], 
            DeltaAP = base["AP"] - perturbed["AP"], DeltaBrier = perturbed["Brier"] - base["Brier"])
    }
    details <- rbindlist(rows)
    ranking <- details[, .(MeanDeltaAUC = mean(DeltaAUC), MeanDeltaAP = mean(DeltaAP), MeanDeltaBrier = mean(DeltaBrier)), 
        by = Feature]
    ranking[, `:=`(OriginalOrder, match(Feature, job$features))]
    setorder(ranking, -MeanDeltaAUC, OriginalOrder)
    ranking[, `:=`(Rank, .I)]
    fwrite(details, file.path(out, paste0(family, "_Training_PFI_repeats_v1.csv")))
    fwrite(ranking, file.path(out, paste0(family, "_Training_PFI_rank_v1.csv")))
    print(ranking)
    quit(save = "no")
}
if (mode == "select") {
    family <- args[2]
    tags <- c("Full21", sprintf("Path_k%02d", 20:1))
    z <- lapply(tags, function(tag) read_json_file(paste0(family, "_", tag, "_CV_score.json")))
    path <- rbindlist(lapply(z, function(x) data.table(Model = family, Tag = x$Tag, NFeatures = x$NFeatures, MeanFoldAUC = x$MeanFoldAUC, 
        Features = paste(x$Features, collapse = ";"))))
    path[, `:=`(CumulativeAUCLoss, path$MeanFoldAUC[path$Tag == "Full21"] - MeanFoldAUC)]
    selected <- lapply(job$loss_tolerances, function(tolerance) {
        row <- path[CumulativeAUCLoss <= tolerance][order(NFeatures)][1L]
        z <- z[[match(row$Tag, tags)]]
        list(Tolerance = tolerance, Tag = z$Tag, NFeatures = z$NFeatures, Features = z$Features, MeanFoldAUC = z$MeanFoldAUC, 
            CumulativeAUCLoss = row$CumulativeAUCLoss)
    })
    write_json_file(list(Model = family, Selected = selected), paste0(family, "_Selection_v1.json"))
    fwrite(path, file.path(out, paste0(family, "_Recursive_path_v1.csv")))
    print(rbindlist(lapply(selected, function(z) data.table(Tolerance = z$Tolerance, Tag = z$Tag, NFeatures = z$NFeatures, 
        MeanFoldAUC = z$MeanFoldAUC, CumulativeAUCLoss = z$CumulativeAUCLoss))))
    quit(save = "no")
}
