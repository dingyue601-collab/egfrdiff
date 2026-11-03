source("shared/R/config.R")
options(encoding = "UTF-8", stringsAsFactors = FALSE, width = 160)
invisible(Sys.setlocale("LC_CTYPE", "English_United States.utf8"))
suppressPackageStartupMessages({
    library(data.table)
    library(matrixStats)
    library(jsonlite)
})
setDTthreads(1L)
set.seed(123)
root <- Sys.getenv("EGFR_PROJECT_ROOT", normalizePath(getwd(), winslash = "/"))
base <- file.path(root, "results")
old <- file.path(base, "metrics")
out <- file.path(base, "bootstrap")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
B <- 5000L
status <- function(stage, detail = "") {
    write_json(list(stage = stage, detail = detail, bootstrap = B, seed = 123L, updated = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z")), 
        file.path(out, "BOOTSTRAP_RUN_STATUS.json"), auto_unbox = TRUE, pretty = TRUE)
    cat(format(Sys.time(), "%H:%M:%S"), stage, detail, "\n")
    flush.console()
}
status("STARTED")
a <- readRDS(file.path(old, "Release_metrics.rds"))
y <- a$reference$Validation$Label
groups <- split(seq_along(y), a$reference$Validation$Cluster)
id <- match(as.character(a$reference$Validation$Cluster), names(groups))
stopifnot(length(y) == 3038L, sum(y) == 658L, length(groups) == 881L, length(a$probabilities) == 9L, all(!is.na(id)))
set.seed(123)
counts <- vapply(seq_len(B), function(b) tabulate(sample.int(length(groups), length(groups), replace = TRUE), nbins = length(groups)), 
    integer(length(groups)))
saveRDS(list(multiplicity = counts, bootstrap = B, seed = 123L, method = "Resample 881 patients with replacement; retain all their eligible admissions"), 
    file.path(out, "Bootstrap_patient_multiplicities.rds"))
clamp <- function(x) pmin(pmax(x, 1e-08), 1 - 1e-08)
safe <- function(x, d) ifelse(d > 0, x/d, NA_real_)
metric_names <- c("AUC", "AP", "Brier", "LogLoss", "Sensitivity", "Specificity", "PPV", "NPV", "Accuracy", "F1", 
    "MCC", "TestingFraction", "MeanPrediction", "Observed", "TP", "FP", "FN", "TN")
evaluate_weights <- function(y, p, w, thresholds) {
    ord <- order(p)
    ties <- cumsum(c(TRUE, diff(p[ord]) != 0))
    pw <- rowsum(w[ord, , drop = FALSE] * y[ord], ties, reorder = FALSE)
    nw <- rowsum(w[ord, , drop = FALSE] * (1 - y[ord]), ties, reorder = FALSE)
    n1 <- colSums(pw)
    n0 <- colSums(nw)
    nn <- n1 + n0
    auc <- colSums(pw * (colCumsums(nw) - nw/2))/(n1 * n0)
    revix <- rev(seq_len(nrow(pw)))
    cp <- colCumsums(pw[revix, , drop = FALSE])
    cn <- colCumsums(nw[revix, , drop = FALSE])
    precision <- cp/(cp + cn)
    precision[!is.finite(precision)] <- 0
    ap <- colSums(pw[revix, , drop = FALSE] * precision)/n1
    q <- clamp(p)
    core <- rbind(AUC = auc, AP = ap, Brier = colSums(w * (p - y)^2)/nn, LogLoss = -colSums(w * (y * log(q) + (1 - 
        y) * log1p(-q)))/nn)
    ans <- lapply(thresholds, function(t) {
        tp <- colSums(w * ((p >= t) * y))
        fp <- colSums(w * ((p >= t) * (1 - y)))
        fn <- n1 - tp
        tn <- n0 - fp
        den <- sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        rbind(core, Sensitivity = tp/n1, Specificity = tn/n0, PPV = safe(tp, tp + fp), NPV = safe(tn, tn + fn), 
            Accuracy = (tp + tn)/nn, F1 = safe(2 * tp, 2 * tp + fp + fn), MCC = safe(tp * tn - fp * fn, den), TestingFraction = (tp + 
                fp)/nn, MeanPrediction = colSums(w * p)/nn, Observed = n1/nn, TP = tp, FP = fp, FN = fn, TN = tn)
    })
    list(metrics = ans, cp = cp, cn = cn, precision = precision, n1 = n1, n0 = n0, nn = nn)
}
ci <- function(x) apply(x, 1L, quantile, probs = c(0.025, 0.975), na.rm = TRUE, names = FALSE)
roc_grid <- seq(0, 1, 0.01)
pr_grid <- seq(0.01, 0.99, 0.01)
recall_source <- readRDS(file.path(base, "calibration/TabM9_calibrated_recall60_90_v1.rds"))
recall_thresholds <- as.data.table(recall_source$thresholds)[Method == "Platt"]
models <- names(a$probabilities)
all_results <- setNames(vector("list", length(models)), models)
for (m in models) {
    checkpoint <- file.path(out, paste0("Bootstrap_", m, "_B5000.rds"))
    if (file.exists(checkpoint)) {
        all_results[[m]] <- readRDS(checkpoint)
        stopifnot(all_results[[m]]$bootstrap == B)
        status("MODEL_REUSED", m)
        next
    }
    status("MODEL_BOOTSTRAP", m)
    p <- a$probabilities[[m]]$Validation
    ts <- c(Fixed0p5 = 0.5, Recall80 = a$parameters[Algorithm == m, Threshold80])
    if (m == "TabM9") {
        extra <- recall_thresholds[TargetRecall != 0.8]
        ts <- c(ts, setNames(extra$Threshold, paste0("Recall", round(100 * extra$TargetRecall))))
    }
    met <- setNames(lapply(ts, function(t) matrix(NA_real_, 18L, B, dimnames = list(metric_names, NULL))), names(ts))
    rb <- matrix(NA_real_, length(roc_grid), B)
    pb <- matrix(NA_real_, length(pr_grid), B)
    for (start in seq.int(1L, B, 250L)) {
        bb <- start:min(B, start + 249L)
        v <- evaluate_weights(y, p, counts[id, bb, drop = FALSE], ts)
        for (nm in names(ts)) met[[nm]][, bb] <- v$metrics[[nm]]
        for (j in seq_along(bb)) {
            fpr <- c(0, v$cn[, j]/v$n0[j], 1)
            tpr <- c(0, v$cp[, j]/v$n1[j], 1)
            keep <- !duplicated(fpr, fromLast = TRUE)
            rb[, bb[j]] <- approx(fpr[keep], tpr[keep], xout = roc_grid, ties = "ordered", rule = 2)$y
            k <- pmin(nrow(v$cp), findInterval(pr_grid, v$cp[, j]/v$n1[j], left.open = TRUE) + 1L)
            pb[, bb[j]] <- v$precision[k, j]
        }
    }
    point <- evaluate_weights(y, p, matrix(1, length(y), 1), ts)$metrics
    original <- a$table2[Algorithm == m]
    stopifnot(max(abs(as.numeric(point$Recall80[, 1]) - as.numeric(original[, ..metric_names]))) < 1e-10)
    result <- list(Algorithm = m, bootstrap = B, thresholds = ts, point = point, metrics = met, roc_grid = roc_grid, 
        roc = rb, pr_grid = pr_grid, pr = pb)
    saveRDS(result, checkpoint)
    all_results[[m]] <- result
    status("MODEL_COMPLETE", m)
}
intervals <- rbindlist(lapply(models, function(m) rbindlist(lapply(names(all_results[[m]]$metrics), function(nm) {
    x <- all_results[[m]]
    z <- ci(x$metrics[[nm]])
    data.table(Algorithm = m, Rule = nm, Threshold = x$thresholds[nm], Metric = metric_names, Estimate = as.numeric(x$point[[nm]][, 
        1]), Lower95 = z[1, ], Upper95 = z[2, ], Bootstrap = B)
}))))
roc_bands <- rbindlist(lapply(models, function(m) {
    x <- all_results[[m]]
    z <- ci(x$roc)
    data.table(Algorithm = m, FPR = roc_grid, Lower95 = z[1, ], Upper95 = z[2, ])
}))
pr_bands <- rbindlist(lapply(models, function(m) {
    x <- all_results[[m]]
    z <- ci(x$pr)
    data.table(Algorithm = m, Recall = pr_grid, Lower95 = z[1, ], Upper95 = z[2, ])
}))
status("CALIBRATION_DIAGNOSTIC_BOOTSTRAP", "TabM9; fixed Platt probabilities")
cal_file <- file.path(out, "Calibration_TabM9_B5000.rds")
if (file.exists(cal_file)) {
    cal_result <- readRDS(cal_file)
} else {
    p <- a$probabilities$TabM9$Validation
    lp <- qlogis(clamp(p))
    basis <- splines::ns(lp, df = 3)
    X <- cbind(1, basis)
    gx <- seq(quantile(p, 0.025), quantile(p, 0.975), length.out = 80L)
    GX <- cbind(1, predict(basis, newx = qlogis(gx)))
    Xlinear <- cbind(1, lp)
    calc_cal <- function(bb) {
        values <- matrix(NA_real_, length(gx) + 2L, length(bb))
        converged <- matrix(FALSE, 3L, length(bb))
        for (j in seq_along(bb)) {
            w <- counts[id, bb[j]]
            f <- suppressWarnings(glm.fit(X, y, weights = w, family = binomial()))
            f0 <- suppressWarnings(glm.fit(matrix(1, length(y), 1), y, weights = w, offset = lp, family = binomial()))
            f1 <- suppressWarnings(glm.fit(Xlinear, y, weights = w, family = binomial()))
            converged[, j] <- c(f$converged, f0$converged, f1$converged)
            if (all(converged[, j])) 
                values[, j] <- c(plogis(GX %*% f$coefficients), f0$coefficients[1], f1$coefficients[2])
        }
        list(index = bb, values = values, converged = converged)
    }
    cl <- parallel::makeCluster(4L)
    parallel::clusterExport(cl, c("counts", "id", "y", "lp", "X", "GX", "Xlinear", "gx", "calc_cal"), envir = environment())
    jobs <- parallel::parLapply(cl, split(seq_len(B), rep(1:4, length.out = B)), calc_cal)
    parallel::stopCluster(cl)
    cb <- matrix(NA_real_, length(gx) + 2L, B)
    conv <- matrix(FALSE, 3L, B)
    for (z in jobs) {
        cb[, z$index] <- z$values
        conv[, z$index] <- z$converged
    }
    stopifnot(all(conv), all(is.finite(cb)))
    z <- ci(cb)
    fitted <- glm.fit(X, y, family = binomial())
    cc <- data.table(Predicted = gx, Observed = as.numeric(plogis(GX %*% fitted$coefficients)), Lower95 = z[1, 
        seq_along(gx)], Upper95 = z[2, seq_along(gx)])
    refcal <- a$calcurves[Algorithm == "TabM9"]
    stopifnot(max(abs(cc$Observed - refcal$Observed)) < 1e-10)
    bin <- as.integer(cut(p, breaks = unique(quantile(p, seq(0, 1, 0.1))), include.lowest = TRUE))
    bin_data <- rbindlist(lapply(sort(unique(bin)), function(k) {
        ix <- which(bin == k)
        w <- counts[id[ix], , drop = FALSE]
        obs <- colSums(w * y[ix])/colSums(w)
        data.table(Bin = k, N = length(ix), MeanPredicted = mean(p[ix]), Observed = mean(y[ix]), Lower95 = quantile(obs, 
            0.025), Upper95 = quantile(obs, 0.975))
    }))
    cal_result <- list(curve = cc, bins = bin_data, bootstrap = B, converged = colSums(conv) == 3L, parameters = data.table(Metric = c("CalibrationIntercept", 
        "CalibrationSlope"), Estimate = c(a$table2[Algorithm == "TabM9", CalibrationIntercept], a$table2[Algorithm == 
        "TabM9", CalibrationSlope]), Lower95 = z[1, length(gx) + (1:2)], Upper95 = z[2, length(gx) + (1:2)]), replicates = cb)
    saveRDS(cal_result, cal_file)
}
status("DCA_AND_PAIRED_COMPARISONS")
p <- a$probabilities$TabM9$Validation
threshold_grid <- seq(0.01, 0.8, 0.01)
contribution <- vapply(threshold_grid, function(t) (p >= t) * (y - (1 - y) * t/(1 - t)), numeric(length(y)))
patient_contribution <- rowsum(contribution, id, reorder = TRUE)
nb <- crossprod(counts, patient_contribution)/as.numeric(crossprod(counts, lengths(groups)))
qnb <- apply(nb, 2, quantile, probs = c(0.025, 0.975))
dca_bands <- data.table(Threshold = threshold_grid, NetBenefit = colMeans(contribution), Lower95 = qnb[1, ], Upper95 = qnb[2, 
    ])
full_models <- setdiff(models, "TabM9")
pairs <- combn(full_models, 2L, simplify = FALSE)
paired <- rbindlist(lapply(pairs, function(pair) {
    aa <- all_results[[pair[1]]]$metrics$Recall80["AUC", ]
    bb <- all_results[[pair[2]]]$metrics$Recall80["AUC", ]
    delta <- all_results[[pair[1]]]$point$Recall80["AUC", 1] - all_results[[pair[2]]]$point$Recall80["AUC", 1]
    d <- aa - bb
    q <- quantile(d, c(0.025, 0.975))
    data.table(ModelA = pair[1], ModelB = pair[2], DeltaAUC = delta, Lower95 = q[1], Upper95 = q[2], P = (1 + sum(abs(d - 
        delta) >= abs(delta)))/(B + 1L), Bootstrap = B)
}))
paired[, `:=`(P_Holm, p.adjust(P, "holm"))]
status("ROBUSTNESS_BOOTSTRAP")
inputs <- readRDS(file.path(base, "ml_imputation_inputs.rds"))
cohort <- readRDS(file.path(base, "patient_split.rds"))
ad <- as.data.table(cohort$admissions)[match(inputs$evaluation_keys$EncounterID, EncounterID)]
stopifnot(all(ad$negative30 == y))
features <- c("age", "sex", "scr", "hba1c", "hemoglobin", "uric_acid", "albumin", "alp", "ada")
ad[, `:=`(Index, seq_len(.N))]
sets <- list(All_admissions = seq_along(y),
    Complete_nine_inputs = which(complete.cases(inputs$evaluation_raw[, features, drop = FALSE])),
    Excluding_boundary_plusminus5 = which(abs(ad$eGFRdiff + 30) > 5), Patient_equal_weights = seq_along(y))
robust_intervals <- rbindlist(lapply(names(sets), function(nm) {
    ix <- sets[[nm]]
    patient <- a$reference$Validation$Cluster[ix]
    gr <- split(seq_along(ix), patient)
    ids <- match(as.character(patient), names(gr))
    set.seed(123)
    mult <- vapply(seq_len(B), function(b) tabulate(sample.int(length(gr), length(gr), replace = TRUE), nbins = length(gr)), 
        integer(length(gr)))
    w0 <- if (nm == "Patient_equal_weights") 
        1/ave(rep(1, length(ix)), patient, FUN = length)
    else rep(1, length(ix))
    mat <- matrix(NA_real_, 18L, B, dimnames = list(metric_names, NULL))
    for (start in seq.int(1L, B, 250L)) {
        bb <- start:min(B, start + 249L)
        mat[, bb] <- evaluate_weights(y[ix], p[ix], mult[ids, bb, drop = FALSE] * w0, c(Recall80 = a$parameters[Algorithm == 
            "TabM9", Threshold80]))$metrics$Recall80
    }
    z <- ci(mat)
    original <- a$robustness[Analysis == nm]
    point <- evaluate_weights(y[ix], p[ix], matrix(w0, ncol = 1), c(Recall80 = a$parameters[Algorithm == "TabM9", 
        Threshold80]))$metrics$Recall80[, 1]
    stopifnot(max(abs(point - as.numeric(original[, ..metric_names]))) < 1e-10)
    data.table(Analysis = nm, Admissions = length(ix), Patients = length(gr), Positive = sum(y[ix]), Metric = metric_names, 
        Estimate = point, Lower95 = z[1, ], Upper95 = z[2, ], Bootstrap = B)
}))
saveRDS(list(intervals = intervals, roc_bands = roc_bands, pr_bands = pr_bands, calibration = cal_result, dca_bands = dca_bands, 
    paired_full21 = paired, robustness = robust_intervals, bootstrap = B, seed = 123L, patients = 881L, admissions = 3038L, 
    positives = 658L, methods = "Fixed-prediction patient-cluster percentile intervals; not simultaneous bands, pipeline optimism correction or threshold-selection uncertainty"), 
    file.path(out, "Publication_B5000_results.rds"))
fwrite(intervals, file.path(out, "Performance_intervals_B5000.csv"))
fwrite(paired, file.path(out, "Full21_pairwise_AUC_B5000.csv"))
fwrite(robust_intervals, file.path(out, "TabM9_robustness_B5000.csv"))
status("COMPLETE", "Nine ROC/PR bands, fixed-rule metrics, TabM9 calibration/DCA, pairwise AUC and robustness intervals")
print(intervals[Algorithm == "TabM9" & Rule == "Recall80" & Metric %in% c("AUC", "AP", "Brier", "Sensitivity", 
    "Specificity")])
