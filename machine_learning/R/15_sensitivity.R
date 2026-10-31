source("shared/R/config.R")
options(encoding = "UTF-8", stringsAsFactors = FALSE)
suppressPackageStartupMessages({
    library(data.table)
    library(matrixStats)
    library(pROC)
})
setDTthreads(1L)
set.seed(123)
root <- Sys.getenv("EGFR_PROJECT_ROOT", normalizePath(getwd(), winslash = "/"))
base <- file.path(root, "results")
out <- file.path(base, "publication")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
source(file.path(root, "shared/R/prediction_metrics.R"))
a <- readRDS(file.path(base, "metrics/Release_metrics.rds"))
u <- readRDS(file.path(base, "bootstrap/Publication_B5000_results.rds"))
inputs <- readRDS(file.path(base, "ml_imputation_inputs.rds"))
cohort <- readRDS(file.path(base, "patient_split.rds"))
ad <- as.data.table(cohort$admissions)[match(inputs$evaluation_keys$EncounterID, EncounterID)]
y <- a$reference$Validation$Label
p <- a$probabilities$TabM9$Validation
patient <- a$reference$Validation$Cluster
threshold <- a$parameters[Algorithm == "TabM9", Threshold80]
stopifnot(nrow(ad) == 3038L, all(ad$negative30 == y), sum(y) == 658L, uniqueN(patient) == 881L, abs(threshold - 
    0.163686262288466) < 1e-14, !anyNA(ad$eGFRdiff))
point_check <- prediction_metrics(y, p, matrix(1, nrow(ad), 1), c(Recall80 = threshold))$Recall80
saved <- a$table2[Algorithm == "TabM9"]
stopifnot(max(abs(point_check[, 1] - as.numeric(saved[, colnames(t(point_check)), with = FALSE]))) < 1e-10, abs(point_check["AUC", 
    1] - as.numeric(pROC::auc(y, p, levels = c(0, 1), direction = "<", quiet = TRUE))) < 1e-12)
result <- list()
for (gap in 5L) {
    ix <- which(abs(ad$eGFRdiff + 30) > gap)
    g <- patient[ix]
    groups <- split(seq_along(ix), g)
    id <- match(as.character(g), names(groups))
    weights <- 1/ave(rep(1, length(ix)), g, FUN = length)
    stopifnot(max(abs(rowsum(weights, g) - 1)) < 1e-12)
    set.seed(123)
    mult <- vapply(seq_len(5000L), function(b) tabulate(sample.int(length(groups), length(groups), TRUE), nbins = length(groups)), 
        integer(length(groups)))
    point <- prediction_metrics(y[ix], p[ix], matrix(weights, ncol = 1), c(Recall80 = threshold))$Recall80[, 1]
    draws <- matrix(NA_real_, length(point), 5000L, dimnames = list(names(point), NULL))
    for (start in seq.int(1L, 5000L, 250L)) {
        bb <- start:min(5000L, start + 249L)
        draws[, bb] <- prediction_metrics(y[ix], p[ix], mult[id, bb, drop = FALSE] * weights, c(Recall80 = threshold))$Recall80
    }
    stopifnot(all(is.finite(draws)))
    ci <- apply(draws, 1, quantile, c(0.025, 0.975), names = FALSE)
    rank <- data.table(p = p[ix], pos = weights * y[ix], neg = weights * (1 - y[ix]))[, .(pos = sum(pos), neg = sum(neg)), 
        by = p][order(p)]
    independent_auc <- rank[, sum(pos * (cumsum(neg) - neg/2))/(sum(pos) * sum(neg))]
    stopifnot(abs(independent_auc - point["AUC"]) < 1e-12)
    name <- paste0("Boundary", gap, "_patient_equal")
    result[[name]] <- data.table(Analysis = name, Admissions = length(ix), Patients = length(groups), Positive = sum(y[ix]), 
        Metric = names(point), Estimate = point, Lower95 = ci[1, ], Upper95 = ci[2, ], Bootstrap = 5000L)
    saveRDS(list(gap = gap, point = point, bootstrap = draws, seed = 123L, threshold = threshold, method = "Exclude boundary admissions; recompute1/n retained admissions for each patient; fixed predictions; patient-cluster bootstrap"), 
        file.path(out, paste0(name, "_source.rds")))
    cat(name, "N", length(ix), "patients", length(groups), "AUC", sprintf("%.6f", point["AUC"]), "\n")
    flush.console()
}
all <- rbindlist(result)
saveRDS(list(new_results = all, original_robustness = u$robustness[!grepl("10", Analysis)], threshold = threshold, 
    bootstrap = 5000L, seed = 123L), file.path(out, "Boundary_equal_weight_sensitivity.rds"))
fwrite(all, file.path(out, "Boundary_equal_weight_sensitivity.csv"))
cat("BOUNDARY_PATIENT_EQUAL_EVALUATION_COMPLETE\n")
