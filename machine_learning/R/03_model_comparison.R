source("shared/R/config.R")
options(encoding = "UTF-8", stringsAsFactors = FALSE, width = 150, warn = 1)
Sys.setenv(OMP_NUM_THREADS = "2", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1")
suppressPackageStartupMessages(library(data.table))
setDTthreads(1L)
root <- getwd()
out <- file.path(root, "results")
data_prefix <- file.path(out, "ml_imputation")
prefix <- file.path(out, "models")
python <- Sys.getenv("EGFR_PYTHON", "python")
seed <- 123L
M <- 30L
tune_m <- 1L
confirm_m <- seq_len(5L)
set.seed(seed)
mode <- if (length(commandArgs(TRUE))) commandArgs(TRUE)[1L] else "prepare"
note <- function(...) {
    cat(format(Sys.time(), "%F %T"), ..., "\n")
    flush.console()
}
source("shared/R/estimators.R")
legacy_fit_one <- fit_one
prep_path <- paste0(prefix, "_MODEL_PREPARATION.rds")
if (!file.exists(prep_path)) {
    a <- readRDS(paste0(data_prefix, "_inputs.rds"))
    sel <- readRDS(paste0(data_prefix, "_selection.rds"))
    tr <- readRDS(paste0(data_prefix, "_Train_completed.rds"))
    qu <- readRDS(paste0(data_prefix, "_Evaluation_completed.rds"))
    features <- sel$selected18
    stopifnot(length(features) > 0L, all(features %in% a$features), isTRUE(sel$no_forced_predictors))
    subset_frame <- function(x) {
        z <- x[, features, drop = FALSE]
        z$proteinuria <- if ("proteinuria" %in% features) 
            factor(z$proteinuria, levels = c("A1", "A2", "A3"), ordered = FALSE)
        else NULL
        z
    }
    blocks <- list(Full = list(train = lapply(tr$completed, subset_frame), query = lapply(qu$completed, subset_frame), 
        ytrain = a$y, train_rows = seq_len(nrow(a$train)), query_rows = seq_len(nrow(a$evaluation_raw)), features = features))
    for (k in seq_len(5L)) {
        x <- readRDS(paste0(data_prefix, "_Inner", k, "_Train.rds"))
        q <- readRDS(paste0(data_prefix, "_Inner", k, "_Query.rds"))
        stopifnot(!length(intersect(a$train_keys$PatientID[x$rows], a$train_keys$PatientID[q$rows])))
        blocks[[paste0("Fold", k)]] <- list(train = lapply(x$completed, subset_frame), query = lapply(q$completed, 
            subset_frame), ytrain = a$y[x$rows], yquery = a$y[q$rows], train_rows = x$rows, query_rows = q$rows, 
            features = features)
    }
    p <- list(blocks = blocks, features = features, train_y = a$y, train_keys = a$train_keys, evaluation_keys = a$evaluation_keys, 
        settings = list(seed = seed, label = "Measured relative eGFRdiff < -30%", selection = "LASSO lambda.1se >=18/30; NO forced additions", 
            conditional_cv = "Full-training LASSO list is fixed for tuning; CV scores are development scores, not an independent validation estimate", 
            fold_mi = "Separately refitted within each five-fold train/query split; query labels never used by MICE", 
            primary_metric = "Mean patient-grouped five-fold AUC", minimum_meaningful_improvement = 0.002, tune_imputation = 1L, 
            confirmation_imputations = 1:5, final_imputations = 30L, classic_max_trials = 40L, classic_min_trials = 20L, 
            classic_block = 10L, TabM_max_trials = 32L, TabM_min_trials = 16L, foundation_search = "None: fixed official default scientific configuration", 
            probabilities = "Average predictions; do not pick the best imputation", heldout_status = "Patient-separated internal validation; labels excluded from model selection"),
        packages = sapply(c("mice", "glmnet", "ranger", "xgboost", "lightgbm", "e1071", "reticulate"), function(x) as.character(packageVersion(x))), 
        sources = c(basename(paste0(data_prefix, "_inputs.rds")), basename(paste0(data_prefix, "_selection.rds"))), 
        created = Sys.time())
    saveRDS(p, prep_path)
    stopifnot(!"yquery" %in% names(p$blocks$Full))
    note("MODEL_INPUTS_READY", length(features), "FEATURES; NO EVALUATION LABELS")
}
p <- readRDS(prep_path)
stopifnot(p$settings$seed == 123L)
if (mode == "prepare") quit(save = "no")
data_blocks <- lapply(p$blocks, function(a) lapply(seq_along(a$train), function(m) make_design(a$train[[m]], a$query[[m]])))
compute_path <- paste0(prefix, "_COMPUTE.rds")
if (mode == "benchmark") {
    z <- benchmark()
    saveRDS(z, compute_path)
    print(z)
    quit(save = "no")
}
compute <- if (file.exists(compute_path)) readRDS(compute_path) else {
    z <- benchmark()
    saveRDS(z, compute_path)
    z
}
fit_one <- function(family, pa, bn, m, keep = FALSE) {
    if (family != "Logistic") 
        return(legacy_fit_one(family, pa, bn, m, keep))
    d <- data_blocks[[bn]][[m]]
    y <- p$blocks[[bn]]$ytrain
    f <- glmnet::glmnet(d$train, y, family = "binomial", alpha = pa$alpha, lambda = pa$lambda, standardize = TRUE, 
        thresh = 1e-08, maxit = 100000L)
    z <- as.numeric(predict(f, d$query, s = pa$lambda, type = "response"))
    list(prediction = z, training = if (keep) as.numeric(predict(f, d$train, s = pa$lambda, type = "response")) else NULL, 
        record = if (keep) list(model = f, parameters = pa, columns = colnames(d$train)) else NULL, warnings = character())
}
suggest <- function(family, t) {
    if (family == "Logistic") 
        return(list(alpha = t$suggest_float("alpha", 0, 1), lambda = t$suggest_float("lambda", 1e-05, 1, log = TRUE)))
    if (family == "XGBoost") 
        return(list(depth = t$suggest_int("depth", 1L, 6L), eta = t$suggest_float("eta", 0.01, 0.15, log = TRUE), 
            min_child = t$suggest_float("min_child", 3, 60), subsample = t$suggest_float("subsample", 0.6, 1), 
            colsample = t$suggest_float("colsample", 0.55, 1), l2 = t$suggest_float("l2", 0.1, 100, log = TRUE), 
            l1 = t$suggest_float("l1", 1e-05, 10, log = TRUE), gamma = t$suggest_float("gamma", 0, 4), rounds = t$suggest_int("rounds", 
                150L, 800L, step = 50L), weight = t$suggest_float("weight", 1, 3.5, log = TRUE)))
    if (family == "RF") 
        return(list(trees = t$suggest_int("trees", 400L, 1000L, step = 200L), mtry_fraction = t$suggest_float("mtry_fraction", 
            0.2, 1), node = t$suggest_int("node", 4L, 70L), fraction = t$suggest_float("fraction", 0.5, 1), weight = t$suggest_float("weight", 
            1, 3.5, log = TRUE)))
    if (family == "LightGBM") 
        return(list(eta = t$suggest_float("eta", 0.01, 0.15, log = TRUE), leaves = t$suggest_int("leaves", 4L, 
            47L), depth = t$suggest_int("depth", 2L, 7L), node = t$suggest_int("node", 15L, 120L), colsample = t$suggest_float("colsample", 
            0.55, 1), subsample = t$suggest_float("subsample", 0.6, 1), l1 = t$suggest_float("l1", 1e-05, 10, log = TRUE), 
            l2 = t$suggest_float("l2", 0.1, 100, log = TRUE), rounds = t$suggest_int("rounds", 150L, 800L, step = 50L), 
            weight = t$suggest_float("weight", 1, 3.5, log = TRUE)))
    if (family == "SVM") 
        return(list(kernel = t$suggest_categorical("kernel", list("radial", "linear")), cost = t$suggest_float("cost", 
            0.001, 100, log = TRUE), gamma = t$suggest_float("gamma", 1e-05, 0.5, log = TRUE), weight = t$suggest_float("weight", 
            1, 3.5, log = TRUE)))
    stop("Unsupported family")
}
warm <- list(Logistic = list(alpha = 0, lambda = 0.001), XGBoost = list(depth = 2L, eta = 0.04, min_child = 15, 
    subsample = 0.8, colsample = 0.85, l2 = 10, l1 = 0.1, gamma = 0.1, rounds = 300L, weight = 1), RF = list(trees = 600L, 
    mtry_fraction = 0.5, node = 15L, fraction = 0.8, weight = 1), LightGBM = list(eta = 0.04, leaves = 7L, depth = 3L, 
    node = 40L, colsample = 0.85, subsample = 0.8, l1 = 0.1, l2 = 10, rounds = 300L, weight = 1), SVM = list(kernel = "radial", 
    cost = 1, gamma = 0.03, weight = 1))
score_config <- function(family, pa, mm = 1L, trial = NULL) {
    scores <- briers <- numeric()
    oof <- rep(NA_real_, length(p$train_y))
    records <- list()
    for (k in seq_len(5L)) {
        bn <- paste0("Fold", k)
        aa <- lapply(mm, function(m) fit_one(family, pa, bn, m))
        pp <- rowMeans(do.call(cbind, lapply(aa, `[[`, "prediction")))
        yy <- p$blocks[[bn]]$yquery
        scores[k] <- auc(yy, pp)
        briers[k] <- mean((yy - pp)^2)
        oof[p$blocks[[bn]]$query_rows] <- pp
        records[[k]] <- list(fold = k, prediction = pp, warnings = unique(unlist(lapply(aa, `[[`, "warnings"))))
        if (!is.null(trial)) {
            trial$report(mean(scores), step = as.integer(k))
            if (k >= 2L && isTRUE(trial$should_prune())) 
                return(list(pruned = TRUE, score = mean(scores), folds = k))
        }
    }
    stopifnot(!anyNA(oof))
    list(score = mean(scores), brier = mean(briers), fold_scores = scores, oof = oof, records = records, pruned = FALSE)
}
if (mode == "classic") {
    reticulate::use_python(python, required = TRUE)
    op <- reticulate::import("optuna")
    op$logging$set_verbosity(op$logging$WARNING)
    op_raw <- reticulate::import("optuna", convert = FALSE)
    for (family in names(warm)) {
        dest <- paste0(prefix, "_", family, "_CLASSIC.rds")
        if (file.exists(dest)) {
            note("COMPLETED_MODEL_REUSED", family)
            next
        }
        db <- gsub("\\\\", "/", paste0(prefix, "_OPTUNA_", family, ".sqlite3"))
        study <- op$create_study(study_name = family, direction = "maximize", storage = paste0("sqlite:///", db), 
            load_if_exists = TRUE, sampler = op$samplers$TPESampler(seed = as.integer(seed), n_startup_trials = 8L), 
            pruner = op$pruners$MedianPruner(n_startup_trials = 8L, n_warmup_steps = 2L))
        if (length(study$trials) == 0L) 
            study$enqueue_trial(warm[[family]])
        for (oldtrial in study$trials) {
            if (isTRUE(oldtrial$state == 0L)) {
                ck <- paste0(prefix, sprintf("_%s_TRIAL%03d.rds", family, as.integer(oldtrial$number)))
                if (file.exists(ck)) {
                  rr <- readRDS(ck)
                  if (isTRUE(rr$result$pruned)) 
                    study$tell(as.integer(oldtrial$number), state = op_raw$trial$TrialState$PRUNED)
                  else if (!is.null(rr$error)) 
                    study$tell(as.integer(oldtrial$number), state = op_raw$trial$TrialState$FAIL)
                  else if (!is.null(rr$result$score)) 
                    study$tell(as.integer(oldtrial$number), rr$result$score)
                }
            }
        }
        best <- tryCatch(as.numeric(study$best_value), error = function(e) -Inf)
        stall <- 0L
        blockbest <- best
        for (j in seq_len(max(0L, 40L - length(study$trials)))) {
            t <- study$ask()
            pa <- suggest(family, t)
            t0 <- proc.time()[3L]
            err <- NULL
            z <- tryCatch(score_config(family, pa, 1L, t), error = function(e) {
                err <<- conditionMessage(e)
                NULL
            })
            rec <- list(trial = as.integer(t$number), family = family, parameters = pa, result = z, error = err, 
                seconds = unname(proc.time()[3L] - t0))
            saveRDS(rec, paste0(prefix, sprintf("_%s_TRIAL%03d.rds", family, as.integer(t$number))))
            if (!is.null(err)) {
                study$tell(t, state = op_raw$trial$TrialState$FAIL)
                note("TRIAL_FAILED", family, t$number, err)
                next
            }
            if (isTRUE(z$pruned)) 
                study$tell(t, state = op_raw$trial$TrialState$PRUNED)
            else {
                study$tell(t, z$score)
                best <- max(best, z$score)
            }
            n <- length(study$trials)
            note("TRIAL_DONE", family, t$number, "AUC", round(z$score, 4), "PRUNED", z$pruned, "SECONDS", round(rec$seconds))
            if (n >= 20L && n%%10L == 0L) {
                stall <- if (best - blockbest < 0.002) 
                  stall + 1L
                else 0L
                blockbest <- best
                if (stall >= 2L) 
                  break
            }
        }
        paths <- list.files(out, pattern = paste0("^", basename(prefix), "_", family, "_TRIAL[0-9]+[.]rds$"), full.names = TRUE)
        records <- lapply(paths, readRDS)
        complete <- Filter(function(x) is.null(x$error) && !is.null(x$result) && !isTRUE(x$result$pruned), records)
        stopifnot(length(complete) > 0L)
        ord <- order(vapply(complete, function(x) x$result$score, 0), decreasing = TRUE)
        contenders <- complete[head(ord, 3L)]
        confirms <- lapply(contenders, function(a) {
            f <- paste0(prefix, sprintf("_%s_CONFIRM%03d.rds", family, a$trial))
            if (file.exists(f)) 
                return(readRDS(f))
            z <- score_config(family, a$parameters, confirm_m)
            rr <- list(trial = a$trial, parameters = a$parameters, result = z)
            saveRDS(rr, f)
            rr
        })
        ii <- order(vapply(confirms, function(x) x$result$score, 0), -vapply(confirms, function(x) x$result$brier, 
            0), decreasing = TRUE)[1L]
        selected <- confirms[[ii]]
        note("PARAMETERS_FROZEN", family, "CV_AUC", round(selected$result$score, 4))
        final <- vector("list", M)
        for (m in seq_len(M)) {
            f <- paste0(prefix, sprintf("_%s_FINAL_m%02d.rds", family, m))
            final[[m]] <- if (file.exists(f)) 
                readRDS(f)
            else {
                z <- fit_one(family, selected$parameters, "Full", m, keep = TRUE)
                saveRDS(z, f)
                z
            }
        }
        prediction <- rowMeans(do.call(cbind, lapply(final, `[[`, "prediction")))
        training_prediction <- rowMeans(do.call(cbind, lapply(final, `[[`, "training")))
        state <- list(family = family, selected = selected, cv_auc = selected$result$score, features = p$features, 
            probability = prediction, training_probability = training_prediction, records = records, confirmations = confirms, 
            parameters = selected$parameters, models = lapply(final, `[[`, "record"), m = M, compute = compute, 
            settings = p$settings, created = Sys.time())
        saveRDS(state, dest)
        note("FINAL30_COMPLETE", family, "EVALUATION_NOT_YET_SCORED")
    }
    quit(save = "no")
}
if (mode == "foundation_inputs") {
    catmap <- list(sex = c(0, 1), proteinuria = c(1, 2, 3), reported_weight_loss = c(0, 1), history_hypertension = c(0, 
        1), history_coronary_disease = c(0, 1))
    numframe <- function(x) {
        z <- as.data.frame(x)
        for (v in names(z)) if (is.factor(z[[v]])) 
            z[[v]] <- if (v == "proteinuria") 
                as.integer(z[[v]])
            else as.integer(as.character(z[[v]]))
        z
    }
    jobs <- list()
    for (bn in names(p$blocks)) {
        a <- p$blocks[[bn]]
        dd <- file.path(out, paste0(basename(prefix), "_tab_inputs"), bn)
        dir.create(dd, recursive = TRUE, showWarnings = FALSE)
        mm <- if (bn == "Full") 
            seq_len(M)
        else seq_len(5L)
        inputs <- lapply(mm, function(m) {
            tf <- file.path(dd, sprintf("train_m%02d.csv", m))
            qf <- file.path(dd, sprintf("query_m%02d.csv", m))
            aa <- numframe(a$train[[m]])
            bb <- numframe(a$query[[m]])
            stopifnot(all(is.finite(as.matrix(aa))), all(is.finite(as.matrix(bb))))
            fwrite(data.table(label = a$ytrain, aa), tf)
            fwrite(bb, qf)
            stopifnot(!"label" %in% names(fread(qf)))
            list(train = tf, query = qf)
        })
        lf <- NULL
        if (bn != "Full") {
            lf <- file.path(dd, "development_labels.csv")
            fwrite(data.table(label = a$yquery), lf)
        }
        jobs[[bn]] <- list(block = bn, features = p$features, categorical = catmap[names(catmap) %in% p$features], 
            inputs = inputs, query_labels = lf, query_rows = a$query_rows)
    }
    manifest <- list(prefix = prefix, output_dir = out, seed = seed, blocks = jobs, settings = p$settings, python = python, 
        precision = "float32", TabM_members = 32L, foundation_min_members = 16L, no_weight_finetuning = TRUE)
    jsonlite::write_json(manifest, paste0(prefix, "_TAB_MANIFEST.json"), auto_unbox = TRUE, pretty = TRUE, null = "null")
    note("FOUNDATION_MANIFEST_READY; NO FINAL EVALUATION LABELS EXPORTED")
    quit(save = "no")
}
