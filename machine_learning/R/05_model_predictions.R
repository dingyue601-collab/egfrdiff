# Mean model probabilities
source("shared/R/config.R")
library(data.table)
prefix <- "results/models"
prepared <- readRDS(paste0(prefix,"_MODEL_PREPARATION.rds"))
probabilities <- list()
for (name in c("Logistic","XGBoost","RF","LightGBM","SVM","TabPFN","TabICL","TabM")) {
  if (name %in% c("Logistic","XGBoost","RF","LightGBM","SVM")) {
    fitted <- readRDS(paste0(prefix,"_",name,"_CLASSIC.rds"))
    stopifnot(fitted$m==30L, length(fitted$models)==30L)
    p <- list(Evaluation30=fitted$probability, Training70_conditional_OOF=fitted$selected$result$oof,
              Training70_apparent=fitted$training_probability)
  } else {
    fitted <- read_json(paste0(prefix,"_",name,"_TUNING.json"),simplifyVector=TRUE)
    validation <- fread(paste0(prefix,"_",name,"_Full.csv"))
    stopifnot(identical(names(validation),paste0("m",1:30)))
    p <- list(Evaluation30=rowMeans(as.matrix(validation)), Training70_conditional_OOF=as.numeric(fitted$oof))
    if (name=="TabM") p$Training70_apparent <- rowMeans(as.matrix(fread(paste0(prefix,"_TabM_Train.csv"))))
  }
  stopifnot(length(p$Evaluation30)==3038, length(p$Training70_conditional_OOF)==6852,
            all(is.finite(p$Evaluation30)), all(is.finite(p$Training70_conditional_OOF)))
  probabilities[[name]] <- p
}
saveRDS(list(probabilities=probabilities,
  tables=list(Selected_features=data.table(Variable=prepared$features))),paste0(prefix,"_PERFORMANCE_v2.rds"))
