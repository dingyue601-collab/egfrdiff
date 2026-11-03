# Nine-predictor probabilities
source("shared/R/config.R")
library(data.table)
out <- "results/model_diagnostics"
dir.create(out,showWarnings=FALSE)
calibration <- readRDS("results/calibration/TabM9_OOF_calibration_results.rds")
thresholds <- readRDS("results/calibration/TabM9_calibrated_recall60_90_v1.rds")$thresholds
stopifnot(nrow(thresholds[Method == "Platt" & TargetRecall == .8]) == 1L)
reference <- list(train = calibration$reference)
prepared <- readRDS("results/ml_imputation_inputs.rds")
labels <- readRDS("results/ml_imputation_evaluation_labels.rds")
validation <- data.table(Row=seq_len(nrow(prepared$evaluation_keys)),Label=labels$label,
                         Cluster=match(prepared$evaluation_keys$PatientID,unique(prepared$evaluation_keys$PatientID)))
reference <- list(TrainingOOF=reference$train,Validation=validation)
predictions <- list()
for (partition in names(reference)) {
  file <- if(partition=="Validation")"TabM_Path_k09_Evaluation.csv" else "TabM_Path_k09_OOF.csv"
  data <- fread(file.path("results/reduction",file))
  stopifnot(identical(data$Row,reference[[partition]]$Row))
  columns <- paste0("m",seq_len(if(partition=="Validation")30 else 5))
  predictions[[partition]] <- rowMeans(as.matrix(data[,columns,with=FALSE]))
}
stopifnot(sum(reference$Validation$Label)==658, nrow(reference$Validation)==3038)
saveRDS(list(reference=reference,predictions=predictions),file.path(out,"TabM9_evaluation_results.rds"))
