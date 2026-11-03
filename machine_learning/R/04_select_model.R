# Training cross-validation model selection
source("shared/R/config.R")
library(data.table)
library(jsonlite)
set.seed(123)
prefix <- Sys.getenv("EGFR_MODEL_PREFIX", "results/models")
prepared <- readRDS(paste0(prefix, "_MODEL_PREPARATION.rds"))
fold <- integer(length(prepared$train_y))
for (k in seq_len(5L)) fold[prepared$blocks[[paste0("Fold", k)]]$query_rows] <- k
stopifnot(all(fold %in% seq_len(5L)),
          all(data.table(Patient = prepared$train_keys$PatientID, Fold = fold)[, uniqueN(Fold), by = Patient]$V1 == 1L))
auc <- function(y, p) {
  n1 <- sum(y == 1L); n0 <- sum(y == 0L)
  stopifnot(n1 > 0L, n0 > 0L)
  (sum(rank(p)[y == 1L]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
families <- c("Logistic", "XGBoost", "RF", "LightGBM", "SVM", "TabPFN", "TabICL", "TabM")
scores <- rbindlist(lapply(families, function(family) {
  if (family %in% c("Logistic", "XGBoost", "RF", "LightGBM", "SVM")) {
    fitted <- readRDS(paste0(prefix, "_", family, "_CLASSIC.rds"))
    prediction <- fitted$selected$result$oof
    recorded <- fitted$cv_auc
  } else {
    fitted <- read_json(paste0(prefix, "_", family, "_TUNING.json"), simplifyVector = TRUE)
    prediction <- fitted$oof
    recorded <- fitted$cv_auc
  }
  stopifnot(length(prediction) == length(fold), all(is.finite(prediction)), all(prediction >= 0 & prediction <= 1))
  by_fold <- vapply(seq_len(5L), function(k) auc(prepared$train_y[fold == k], prediction[fold == k]), 1)
  stopifnot(abs(mean(by_fold) - recorded) < 1e-10)
  data.table(Algorithm = family, CV_MeanAUC = mean(by_fold), TrainingOOF_AUC = auc(prepared$train_y, prediction),
             TrainingOOF_Brier = mean((prediction - prepared$train_y)^2))
}))
setorder(scores, -CV_MeanAUC, TrainingOOF_Brier)
dir.create("results/selection", recursive = TRUE, showWarnings = FALSE)
fwrite(scores, "results/selection/Training_model_comparison.csv")
saveRDS(list(winner = scores$Algorithm[1], comparison = scores, seed = 123L,
             selection = "Highest mean patient-grouped five-fold training AUC; Brier breaks exact ties",
             score_type = "Development CV conditional on the training-selected LASSO list"),
        "results/selection/Training_model_selection.rds")
write_json(list(winner = scores$Algorithm[1], seed = 123L, validation_labels_used = FALSE),
           "results/selection/Training_model_selection.json", auto_unbox = TRUE, pretty = TRUE)
print(scores)
