# Same-admission validation against a later measured reference pair.
options(encoding = "UTF-8", stringsAsFactors = FALSE)
source("shared/R/config.R")
source("shared/R/prediction_metrics.R")
suppressPackageStartupMessages({library(data.table); library(mice); library(jsonlite); library(pROC)})
setDTthreads(1L)
mode <- if (length(commandArgs(TRUE))) commandArgs(TRUE)[1L] else "impute"
out <- "results/supplementary_validation"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
features <- c("age", "sex", "scr", "hba1c", "hemoglobin", "uric_acid", "albumin", "alp", "ada")
d <- fread(file.path(prepared_dir, "supplementary_validation.csv"))
raw <- as.data.frame(lapply(d[, ..features], as.numeric))
stopifnot(!anyDuplicated(d$Record), all(rowSums(!is.na(raw)) >= 7L),
  !anyNA(raw[, c("age", "scr", "ada")]), all(d$Label %in% c(0L, 1L)),
  all(d$PriorGroup %in% c("Original_validation_patient", "New_patient")))
if (mode == "impute") {
  RNGkind("L'Ecuyer-CMRG"); set.seed(123L)
  state <- .Random.seed; complete <- vector("list", 30L); events <- vector("list", 6L)
  method <- setNames(ifelse(colSums(is.na(raw)) > 0, "rf", ""), features)
  pred <- matrix(1L, length(features), length(features), dimnames = list(features, features)); diag(pred) <- 0L
  init <- raw
  for (v in features) init[[v]][is.na(init[[v]])] <- median(raw[[v]], na.rm = TRUE)
  for (j in seq_len(6L)) {
    assign(".Random.seed", state, envir = .GlobalEnv)
    next_state <- parallel::nextRNGStream(state)
    imp <- mice::mice(raw, m = 5L, maxit = 20L, method = method, predictorMatrix = pred,
      visitSequence = names(method)[method != ""], data.init = init, seed = NA, printFlag = FALSE,
      rfPackage = "ranger", ntree = 10L, num.threads = 1L, remove.constant = FALSE, remove.collinear = FALSE)
    events[[j]] <- imp$loggedEvents
    for (k in seq_len(5L)) complete[[(j - 1L)*5L + k]] <- mice::complete(imp, k)
    state <- next_state
  }
  measured <- !is.na(as.matrix(raw))
  stopifnot(all(vapply(complete, function(x) all(is.finite(as.matrix(x))) &&
    identical(as.matrix(x)[measured], as.matrix(raw)[measured]), TRUE)))
  saveRDS(list(completed = complete, loggedEvents = events), file.path(out, "Completed_imputations.rds"))
  inputs <- rbindlist(lapply(seq_len(30L), function(m) {
    x <- as.data.table(complete[[m]]); x[, `:=`(Record = d$Record, Imputation = m)]; x
  }))
  setcolorder(inputs, c("Record", "Imputation", features))
  fwrite(inputs, file.path(out, "Inference_inputs.csv"))
}
if (mode == "evaluate") {
  p <- fread(file.path(out, "Frozen_predictions.csv")); p <- p[match(d$Record, Record)]
  stopifnot(identical(p$Record, d$Record), all(is.finite(p$Probability)))
  config <- jsonlite::fromJSON("calculator/model_config.json")
  threshold <- config$threshold80
  groups <- list(Overall = rep(TRUE, nrow(d)), Original_validation = d$PriorGroup == "Original_validation_patient",
    Outside_original = d$PriorGroup == "New_patient")
  keep <- c("AUC", "AP", "Brier", "Accuracy", "PPV", "Sensitivity", "Specificity", "NPV", "F1", "TestingFraction", "TP", "FP", "FN", "TN")
  results <- lapply(names(groups), function(name) {
    ix <- which(groups[[name]]); z <- d[ix]; prob <- p$Probability[ix]
    people <- unique(z$Patient); cluster <- match(z$Patient, people)
    metric <- function(w) prediction_metrics(z$Label, prob, w, threshold)[[1L]][keep, , drop = FALSE]
    point <- metric(matrix(1, nrow(z), 1L))[, 1L]
    check <- as.numeric(pROC::auc(pROC::roc(z$Label, prob, levels = c(0L, 1L), direction = "<", quiet = TRUE)))
    stopifnot(abs(check - point["AUC"]) < 1e-10)
    set.seed(123L); boot <- matrix(NA_real_, length(keep), 5000L, dimnames = list(keep, NULL))
    for (start in seq(1L, 5000L, 250L)) {
      draw <- rmultinom(250L, length(people), rep(1/length(people), length(people)))
      boot[, start:(start + 249L)] <- metric(draw[cluster, , drop = FALSE])
    }
    interval <- t(apply(boot, 1L, quantile, probs = c(.025, .975), na.rm = TRUE))
    row <- data.table(Group = name, Records = nrow(z), Patients = length(people), Positive = sum(z$Label), Threshold = threshold)
    for (v in keep) {
      label <- if (v == "AP") "AUPRC" else v
      row[, (label) := point[[v]]]
      row[, (paste0(label, "_lower")) := interval[v, 1L]]
      row[, (paste0(label, "_upper")) := interval[v, 2L]]
    }
    row
  })
  fwrite(rbindlist(results), file.path(out, "Performance.csv"))
}
