# Training-derived operating thresholds
source("shared/R/config.R")
library(data.table)
calibration <- readRDS("results/calibration/TabM9_OOF_calibration_results.rds")
y <- calibration$reference$Label
thresholds <- rbindlist(lapply(names(calibration$probabilities$TrainingOOF),function(method){
  probability <- calibration$probabilities$TrainingOOF[[method]]
  target <- c(.60,.70,.80,.90)
  positive <- sort(probability[y==1],decreasing=TRUE)
  data.table(Method=method,TargetRecall=target,Threshold=positive[ceiling(target*length(positive))])
}))
saveRDS(list(thresholds=thresholds),"results/calibration/TabM9_calibrated_recall60_90_v1.rds")
