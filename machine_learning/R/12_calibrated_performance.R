source("shared/R/config.R")
options(encoding = "UTF-8", stringsAsFactors = FALSE, width = 170)
suppressPackageStartupMessages({
    library(data.table)
    library(pROC)
    library(jsonlite)
})
setDTthreads(1L)
set.seed(123)
root <- Sys.getenv("EGFR_PROJECT_ROOT", normalizePath(getwd(), winslash = "/"))
base <- file.path(root, "results")
out <- file.path(base, "metrics")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
s <- readRDS(file.path(base, "models_PERFORMANCE_v2.rds"))
e <- readRDS(file.path(base, "model_diagnostics/TabM9_evaluation_results.rds"))
z <- readRDS(file.path(base, "calibration/TabM9_OOF_calibration_results.rds"))
p <- readRDS(file.path(base, "ml_imputation_inputs.rds"))
cohort <- readRDS(file.path(base, "patient_split.rds"))
ref <- e$reference
stopifnot(nrow(ref$TrainingOOF) == 6852L, nrow(ref$Validation) == 3038L, sum(ref$Validation$Label) == 658L, length(intersect(p$train_keys$PatientID, 
    p$evaluation_keys$PatientID)) == 0L, all(data.table(id = p$train_keys$PatientID, f = p$foldid)[, uniqueN(f), 
    by = id]$V1 == 1L), all(data.table(id = p$train_keys$PatientID, f = p$original_foldid)[, uniqueN(f), by = id]$V1 == 
    1L))
ad <- as.data.table(cohort$admissions)[match(p$evaluation_keys$EncounterID, EncounterID)]
ad[, `:=`(Index, seq_len(.N))]
stopifnot(identical(as.character(ad$EncounterID), as.character(p$evaluation_keys$EncounterID)), all(ad$negative30 == 
    ref$Validation$Label), all(cohort$admissions$negative30 == as.integer(100 * (cohort$admissions$egfrcys - cohort$admissions$egfrcr)/cohort$admissions$egfrcr < 
    -30)), all(abs(as.numeric(cohort$admissions$CysDate - cohort$admissions$ScrDate)) <= 1))
clamp <- function(x) pmin(pmax(x, 1e-08), 1 - 1e-08)
calibrate <- function(x, a, b) plogis(a + b * qlogis(clamp(x)))
metrics <- function(y, p, t, w = rep(1, length(y)), calibration = FALSE) {
    stopifnot(length(y) == length(p), length(w) == length(y), all(is.finite(p)), all(p >= 0 & p <= 1), all(w > 
        0))
    d <- data.table(p = p, y = y, w = w)[, .(W1 = sum(w * y), W0 = sum(w * (1 - y))), by = p][order(p)]
    n1 <- sum(d$W1)
    n0 <- sum(d$W0)
    nn <- sum(w)
    auc <- if (n1 > 0 && n0 > 0) 
        sum(d$W1 * (cumsum(d$W0) - d$W0/2))/(n1 * n0)
    else NA_real_
    d <- d[order(-p)]
    ap <- if (n1 > 0) 
        sum(d$W1/n1 * cumsum(d$W1)/cumsum(d$W1 + d$W0))
    else NA_real_
    flag <- p >= t
    tp <- sum(w * flag * y)
    fp <- sum(w * flag * (1 - y))
    fn <- sum(w * (!flag) * y)
    tn <- sum(w * (!flag) * (1 - y))
    den <- sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    q <- clamp(p)
    ans <- c(AUC = auc, AP = ap, Brier = sum(w * (p - y)^2)/nn, LogLoss = -sum(w * (y * log(q) + (1 - y) * log1p(-q)))/nn, 
        Sensitivity = tp/n1, Specificity = tn/n0, PPV = if (tp + fp > 0) tp/(tp + fp) else NA_real_, NPV = if (tn + 
            fn > 0) tn/(tn + fn) else NA_real_, Accuracy = (tp + tn)/nn, F1 = 2 * tp/(2 * tp + fp + fn), MCC = if (den > 
            0) (tp * tn - fp * fn)/den else NA_real_, TestingFraction = (tp + fp)/nn, MeanPrediction = sum(w * 
            p)/nn, Observed = n1/nn, TP = tp, FP = fp, FN = fn, TN = tn)
    if (calibration) {
        lp <- qlogis(q)
        f1 <- suppressWarnings(glm(y ~ 1 + offset(lp), family = binomial(), weights = w))
        f2 <- suppressWarnings(glm(y ~ lp, family = binomial(), weights = w))
        ans <- c(ans, CalibrationIntercept = unname(coef(f1)[1]), CalibrationSlope = unname(coef(f2)[2]))
    }
    ans
}
curve <- function(y, p) {
    d <- data.table(p = p, y = y)[, .(TP = sum(y), FP = sum(1 - y)), by = p][order(-p)]
    d[, `:=`(TP = cumsum(TP), FP = cumsum(FP))]
    d[, `:=`(Recall = TP/sum(y), FPR = FP/sum(1 - y), Precision = TP/(TP + FP))]
    rbind(data.table(p = Inf, TP = 0, FP = 0, Recall = 0, FPR = 0, Precision = 1), d)
}
raw <- lapply(s$probabilities, function(a) list(TrainingOOF = a$Training70_conditional_OOF, Validation = a$Evaluation30))
raw$TabM9 <- e$predictions
parameters <- z$all_model_parameters[match(names(raw), Algorithm)]
stopifnot(identical(parameters$Algorithm, names(raw)), identical(z$reference$Label, ref$TrainingOOF$Label))
for (m in names(raw)) stopifnot(max(abs(raw[[m]]$TrainingOOF - z$raw_training[[m]])) < 1e-12)
prob <- setNames(lapply(names(raw), function(m) {
    pa <- parameters[Algorithm == m]
    lapply(raw[[m]], calibrate, a = pa$Intercept, b = pa$Slope)
}), names(raw))
performance <- rbindlist(lapply(names(prob), function(m) rbindlist(lapply(names(ref), function(part) {
    pa <- parameters[Algorithm == m]
    v <- metrics(ref[[part]]$Label, prob[[m]][[part]], pa$Threshold80, calibration = TRUE)
    stopifnot(abs(v["AUC"] - as.numeric(pROC::auc(ref[[part]]$Label, raw[[m]][[part]], direction = "<", quiet = TRUE))) < 
        1e-12)
    as.data.table(c(list(Algorithm = m, Partition = part, Inputs = pa$Inputs, Admissions = nrow(ref[[part]]), Patients = uniqueN(ref[[part]]$Cluster), 
        Threshold = pa$Threshold80, TargetRecall = 0.8), as.list(v)))
}))))
fixed05 <- rbindlist(lapply(names(prob), function(m) {
    v <- metrics(ref$Validation$Label, prob[[m]]$Validation, 0.5, calibration = TRUE)
    as.data.table(c(list(Algorithm = m, Threshold = 0.5), as.list(v)))
}))
table2 <- copy(performance[Partition == "Validation"])
table2 <- table2[match(names(prob), Algorithm)]
stopifnot(all(table2$TP + table2$FN == 658), all(table2$FP + table2$TN == 2380))
fwrite(table2, file.path(out, "Table2_Platt80_nine_models.csv"))
fwrite(performance, file.path(out, "Model_performance_trainingOOF_and_evaluation.csv"))
fwrite(fixed05, file.path(out, "TableS_fixed_probability_0p5.csv"))
fwrite(parameters, file.path(out, "Platt_parameters_all_nine_models.csv"))
curves <- rbindlist(lapply(names(prob), function(m) {
    x <- curve(ref$Validation$Label, prob[[m]]$Validation)
    x[, `:=`(Algorithm, m)]
    x
}))
calcurves <- rbindlist(lapply(names(prob), function(m) {
    pp <- prob[[m]]$Validation
    lp <- qlogis(clamp(pp))
    y <- ref$Validation$Label
    fit <- glm(y ~ splines::ns(lp, df = 3), family = binomial())
    grid <- seq(quantile(pp, 0.025), quantile(pp, 0.975), length.out = 80L)
    data.table(Algorithm = m, Predicted = grid, Observed = as.numeric(predict(fit, newdata = data.frame(lp = qlogis(grid)), 
        type = "response")))
}))
nb <- function(y, p, ts) vapply(ts, function(t) mean((p >= t) * y) - mean((p >= t) * (1 - y)) * t/(1 - t), 1)
ts <- seq(0.01, 0.8, 0.01)
dca <- rbindlist(lapply(names(prob), function(m) data.table(Algorithm = m, Threshold = ts, NetBenefit = nb(ref$Validation$Label, 
    prob[[m]]$Validation, ts))))
dca <- rbind(dca, data.table(Algorithm = "Test all", Threshold = ts, NetBenefit = mean(ref$Validation$Label) - 
    (1 - mean(ref$Validation$Label)) * ts/(1 - ts)), data.table(Algorithm = "Test none", Threshold = ts, NetBenefit = 0))
fwrite(curves, file.path(out, "Figure3_ROC_PR_source.csv"))
fwrite(calcurves, file.path(out, "Figure3_calibration_source.csv"))
fwrite(dca, file.path(out, "Figure3_DCA_source.csv"))
features <- c("age", "sex", "scr", "hba1c", "hemoglobin", "uric_acid", "albumin", "alp", "ada")
threshold <- parameters[Algorithm == "TabM9", Threshold80]
q <- prob$TabM9$Validation
y <- ref$Validation$Label
sets <- list(All_admissions = seq_len(nrow(ad)),
    Complete_nine_inputs = which(complete.cases(p$evaluation_raw[, features, drop = FALSE])),
    Excluding_boundary_plusminus5 = which(abs(ad$eGFRdiff + 30) > 5))
robust <- rbindlist(lapply(names(sets), function(nm) {
    ix <- sets[[nm]]
    as.data.table(c(list(Analysis = nm, Admissions = length(ix), Patients = uniqueN(ref$Validation$Cluster[ix]), 
        Positive = sum(y[ix])), as.list(metrics(y[ix], q[ix], threshold))))
}))
w <- 1/ave(rep(1, length(q)), ref$Validation$Cluster, FUN = length)
robust <- rbind(robust, as.data.table(c(list(Analysis = "Patient_equal_weights", Admissions = length(q), Patients = 881L, 
    Positive = sum(y)), as.list(metrics(y, q, threshold, w)))), fill = TRUE)
fwrite(robust, file.path(out, "TableS_TabM9_Platt80_sensitivity.csv"))
saveRDS(list(performance = performance, table2 = table2, fixed05 = fixed05, parameters = parameters, probabilities = prob, 
    curves = curves, calcurves = calcurves, dca = dca, robustness = robust,
    reference = ref, seed = 123L, target_recall = 0.8), file.path(out, "Release_metrics.rds"))
write_json(list(seed = 123, primary_recall = 0.8, primary_threshold = threshold, probability = "Training conditional-OOF Platt; applied after member/MI averaging", 
    class1 = "Measured relative eGFRdiff strictly below -30%", no_classifier_refit = TRUE, assessment = "Post-selection internal evaluation"),
    file.path(out, "Research_model_use.json"), auto_unbox = TRUE, pretty = TRUE, digits = 16)
cat("PLATT80_METRICS_COMPLETE\n")
print(table2[, .(Algorithm, AUC, Brier, Sensitivity, Specificity, PPV, TestingFraction)])
print(robust[, .(Analysis, Admissions, AUC, Brier, Sensitivity)])
