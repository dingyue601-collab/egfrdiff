# Training-OOF calibration
source("shared/R/config.R")
library(data.table)
library(jsonlite)
prefix <- Sys.getenv("EGFR_MODEL_PREFIX", "results/models")
prepared <- readRDS(paste0(prefix, "_MODEL_PREPARATION.rds"))
reference <- readRDS("results/reduction/Statistical_reference_v1.rds")$train
stopifnot(identical(reference$Label, prepared$train_y))
families <- c("Logistic", "XGBoost", "RF", "LightGBM", "SVM", "TabPFN", "TabICL", "TabM")
raw <- setNames(lapply(families, function(family) {
  if (family %in% c("Logistic", "XGBoost", "RF", "LightGBM", "SVM")) {
    readRDS(paste0(prefix, "_", family, "_CLASSIC.rds"))$selected$result$oof
  } else read_json(paste0(prefix, "_", family, "_TUNING.json"), simplifyVector = TRUE)$oof
}), families)
reduced <- fread("results/reduction/TabM_Path_k09_OOF.csv")
stopifnot(identical(reduced$Row, reference$Row), identical(reduced$Fold, reference$Fold))
raw$TabM9 <- rowMeans(as.matrix(reduced[, paste0("m", 1:5), with = FALSE]))
clamp <- function(p) pmax(1e-8, pmin(1-1e-8, p))
y <- reference$Label
all_parameters <- rbindlist(lapply(names(raw), function(family) {
  lp <- qlogis(clamp(raw[[family]]))
  fit <- glm(y ~ lp, family = binomial())
  a <- unname(coef(fit)[1]); b <- unname(coef(fit)[2])
  stopifnot(length(lp) == length(y), fit$converged, is.finite(a), is.finite(b), b > 0)
  cutoff <- sort(raw[[family]][y == 1], decreasing = TRUE)[ceiling(.8 * sum(y))]
  data.table(Algorithm = family, Inputs = if (family == "TabM9") 9L else 21L,
             Intercept = a, Slope = b, Threshold80 = plogis(a + b * qlogis(clamp(cutoff))))
}))
lp <- qlogis(clamp(raw$TabM9))
intercept <- glm(y ~ 1 + offset(lp), family = binomial())
platt <- all_parameters[Algorithm == "TabM9"]
parameters <- data.table(Method = c("Raw", "Intercept-only", "Platt"),
  Intercept = c(0, unname(coef(intercept)[1]), platt$Intercept), Slope = c(1, 1, platt$Slope))
probabilities <- list(TrainingOOF = setNames(lapply(parameters$Method, function(method) {
  pa <- parameters[Method == method]
  if (method == "Raw") raw$TabM9 else plogis(pa$Intercept + pa$Slope * lp)
}), parameters$Method))
dir.create("results/calibration", showWarnings = FALSE)
saveRDS(list(parameters = parameters, probabilities = probabilities, all_model_parameters = all_parameters,
             reference = reference, raw_training = raw),
        "results/calibration/TabM9_OOF_calibration_results.rds")
