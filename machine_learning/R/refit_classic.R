# Refit published model specifications
source("shared/R/config.R")
library(data.table)
source("shared/R/estimators.R")
legacy_fit_one <- fit_one
functions <- parse("machine_learning/R/03_model_comparison.R")
for (expr in functions) {
  if (is.call(expr) && identical(expr[[1]], as.name("<-")) && is.symbol(expr[[2]]) &&
      as.character(expr[[2]]) %in% c("fit_one", "score_config")) eval(expr)
}
p <- readRDS("results/models_MODEL_PREPARATION.rds")
spec <- read_json("docs/published_parameters.json", simplifyVector=FALSE)
seed <- 123L
compute <- spec$compute
data_blocks <- lapply(p$blocks, function(block) lapply(seq_along(block$train), function(m) {
  make_design(block$train[[m]], block$query[[m]])
}))
prefix <- file.path(getwd(), "results/models")
for (family in names(spec$classic)) {
  path <- paste0(prefix, "_", family, "_CLASSIC.rds")
  if (file.exists(path)) next
  parameters <- spec$classic[[family]]$parameters
  selected <- list(parameters=parameters, result=score_config(family, parameters, 1:5))
  final <- lapply(1:30, function(m) {
    path <- paste0(prefix, sprintf("_%s_FINAL_m%02d.rds", family, m))
    if (file.exists(path)) return(readRDS(path))
    fitted <- fit_one(family, parameters, "Full", m, keep=TRUE)
    saveRDS(fitted, path)
    fitted
  })
  records <- lapply(spec$classic[[family]]$trials, function(x) {
    list(trial=x$trial, parameters=x$parameters, error=x$error,
         result=list(score=x$score, pruned=x$pruned))
  })
  state <- list(family=family, selected=selected, cv_auc=selected$result$score, features=p$features,
    probability=rowMeans(do.call(cbind, lapply(final, `[[`, "prediction"))),
    training_probability=rowMeans(do.call(cbind, lapply(final, `[[`, "training"))),
    records=records, parameters=parameters, models=lapply(final, `[[`, "record"), m=30L,
    compute=compute, settings=p$settings)
  saveRDS(state,path)
}
