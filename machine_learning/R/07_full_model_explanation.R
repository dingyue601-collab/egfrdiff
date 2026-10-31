source("shared/R/config.R")
options(encoding = "UTF-8", stringsAsFactors = FALSE, width = 150)
suppressPackageStartupMessages(library(data.table))
setDTthreads(1L)
root <- getwd()
processed <- file.path(root, "results")
source_prefix <- file.path(processed, "models")
out <- file.path(processed, "shap_full")
mode <- if (length(commandArgs(TRUE))) commandArgs(TRUE)[1L] else "prepare"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
if (mode == "prepare") {
    set.seed(123)
    reference <- readRDS(paste0(source_prefix, "_PERFORMANCE_v2.rds"))
    n_query <- 3038L
    n_train <- 6852L
    n_explain <- 128L
    n_background <- 8L
    donors <- sample.int(n_train, n_background)
    rows <- sort(sample.int(n_query, n_explain))
    features <- reference$tables$Selected_features$Variable
    orders <- lapply(seq_len(n_background), function(k) sample.int(length(features)) - 1L)
    plan <- list(seed = 123L, enabled_models = "TabM", source_manifest = paste0(source_prefix, "_TAB_MANIFEST.json"), 
        evaluation_rows_zero_based = rows - 1L, background_rows_zero_based = donors - 1L, permutation_orders_zero_based = orders, 
        features = features, sampling = "Simple random validation admissions, no label-based selection; common sample for both models", 
        background = "Eight simple-random training admissions; common donor identities across MI and algorithms", 
        rationale = "Bounded model-agnostic explanation cost, not a change to the full3038-admission validation analysis", 
        interpretation = "Predictive associations on this random subsample; no causal or mechanistic claim")
    path <- file.path(out, "SHAP_plan_v1.json")
    if (file.exists(path)) {
        previous <- read_json(path, simplifyVector = TRUE)
        stopifnot(identical(as.integer(previous$evaluation_rows_zero_based), rows - 1L), identical(as.integer(previous$background_rows_zero_based), 
            donors - 1L))
    }
    else jsonlite::write_json(plan, path, auto_unbox = TRUE, pretty = TRUE)
    saveRDS(plan, file.path(out, "SHAP_plan_v1.rds"))
    cat("FIXED_SAMPLE_READY", n_explain, "VALIDATION_ADMISSIONS; BACKGROUND", n_background, "TRAINING_ADMISSIONS\n")
    quit(save = "no")
}
