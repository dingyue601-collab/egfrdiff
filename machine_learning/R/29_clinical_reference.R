source("shared/R/config.R")
options(encoding = "UTF-8", stringsAsFactors = FALSE)
suppressPackageStartupMessages({library(data.table); library(pROC); library(jsonlite)})
setDTthreads(1L)
set.seed(123)
base <- "results"
out <- file.path(base, "clinical_reference")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
prep <- readRDS(file.path(base, "ml_imputation_inputs.rds"))
release <- readRDS(file.path(base, "metrics/Release_metrics.rds"))
reference <- list(train = release$reference$TrainingOOF, evaluation = release$reference$Validation)
features <- c("age", "sex", "scr")
nine <- c(features, "hba1c", "hemoglobin", "uric_acid", "albumin", "alp", "ada")
train <- as.data.frame(prep$train[, features])
validation <- as.data.frame(prep$evaluation_raw[, features])
train$sex <- factor(train$sex, levels = c("0", "1"))
validation$sex <- factor(validation$sex, levels = c("0", "1"))
y <- reference$train$Label
yv <- reference$evaluation$Label
fold <- reference$train$Fold
stopifnot(nrow(train) == 6852L, nrow(validation) == 3038L, sum(yv) == 658L,
          identical(reference$train$Row, seq_len(nrow(train))),
          identical(reference$evaluation$Row, seq_len(nrow(validation))),
          identical(y, prep$y), identical(fold, prep$foldid),
          identical(reference$train$Cluster, match(prep$train_keys$PatientID, unique(prep$train_keys$PatientID))),
          identical(reference$evaluation$Cluster, match(prep$evaluation_keys$PatientID, unique(prep$evaluation_keys$PatientID))),
          !anyNA(train), !anyNA(validation), all(train$scr > 0), all(validation$scr > 0),
          !length(intersect(prep$train_keys$PatientID, prep$evaluation_keys$PatientID)),
          all(vapply(split(fold, reference$train$Cluster), function(z) length(unique(z)), 1L) == 1L))
train$Label <- y
forms <- list(Logistic3 = Label ~ age + sex + scr)

oof <- list(TabM9 = release$probabilities$TabM9$TrainingOOF)
pv <- list(TabM9 = release$probabilities$TabM9$Validation)
if (is.null(oof$TabM9) || is.null(pv$TabM9)) {
  stop("Inspect the saved probability names before analysis")
}
models <- list()
calibrators <- list()
thresholds <- c(TabM9 = release$parameters[Algorithm == "TabM9", Threshold80])
clip <- function(p) pmin(pmax(p, 1e-8), 1 - 1e-8)
threshold <- function(y, p) sort(p[y == 1L], decreasing = TRUE)[ceiling(.8 * sum(y))]
for (name in names(forms)) {
  raw <- rep(NA_real_, length(y))
  for (k in 1:5) {
    f <- glm(forms[[name]], data = train[fold != k, ], family = binomial())
    raw[fold == k] <- predict(f, train[fold == k, ], type = "response")
  }
  stopifnot(all(is.finite(raw)))
  calibrators[[name]] <- glm(y ~ qlogis(clip(raw)), family = binomial())
  ab <- coef(calibrators[[name]])
  stopifnot(ab[2] > 0)
  map <- function(p) plogis(ab[1] + ab[2] * qlogis(clip(p)))
  models[[name]] <- glm(forms[[name]], data = train, family = binomial())
  oof[[name]] <- map(raw)
  pv[[name]] <- map(predict(models[[name]], validation, type = "response"))
  thresholds[name] <- threshold(y, oof[[name]])
}
stopifnot(length(thresholds) == 2L, all(vapply(pv, length, 1L) == 3038L),
          abs(thresholds["TabM9"] - .16368626228846581) < 1e-12)

scores <- function(y, p, selected) {
  tp <- sum(selected & y == 1L); fp <- sum(selected & y == 0L)
  fn <- sum(!selected & y == 1L); tn <- sum(!selected & y == 0L)
  d <- data.table(p, y)[, .(n1 = sum(y), n0 = sum(1L - y)), by = p][order(p)]
  auc <- sum(d$n1 * (cumsum(d$n0) - d$n0 / 2)) / (sum(y) * sum(1L - y))
  d <- d[order(-p)]
  ap <- sum(d$n1 / sum(y) * cumsum(d$n1) / cumsum(d$n1 + d$n0))
  c(AUC = auc, AP = ap, Brier = mean((p - y)^2), Sensitivity = tp / (tp + fn),
    Specificity = tn / (tn + fp), PPV = tp / (tp + fp), NPV = tn / (tn + fn),
    Accuracy = (tp + tn) / length(y), F1 = 2 * tp / (2 * tp + fp + fn),
    TestingFraction = mean(selected), TP = tp, FP = fp, FN = fn, TN = tn)
}
capacity <- function(p, selected) {
  z <- rep(FALSE, length(p))
  n <- sum(selected)
  if (n) z[order(-p, seq_along(p))[seq_len(n)]] <- TRUE
  z
}
selections <- function(ix, name, rule) {
  if (rule == "Training80") pv[[name]][ix] >= thresholds[name]
  else if (name == "TabM9") pv[[name]][ix] >= thresholds[name]
  else capacity(pv[[name]][ix], pv$TabM9[ix] >= thresholds["TabM9"])
}
complete <- complete.cases(prep$evaluation_raw[, nine, drop = FALSE])
subsets <- list(All_validation = rep(TRUE, length(yv)), Complete_nine_inputs = complete)
rows <- list()
for (population in names(subsets)) for (rule in c("Training80", "Matched_testing")) for (name in names(pv)) {
  ix <- which(subsets[[population]])
  s <- scores(yv[ix], pv[[name]][ix], selections(ix, name, rule))
  rows[[length(rows) + 1L]] <- as.data.table(c(list(Population = population, Rule = rule, Model = name,
    N = length(ix), Patients = uniqueN(reference$evaluation$Cluster[ix]), Positives = sum(yv[ix]),
    Threshold = if (rule == "Training80" || name == "TabM9") thresholds[name] else NA_real_), as.list(s)))
}
performance <- rbindlist(rows)
for (name in names(pv)) stopifnot(abs(as.numeric(pROC::auc(yv, pv[[name]], quiet = TRUE, direction = "<")) -
  performance[Population == "All_validation" & Rule == "Training80" & Model == name, AUC]) < 1e-10)
ci_metrics <- c("AUC", "Brier", "Sensitivity", "PPV", "TestingFraction")
group <- split(seq_along(yv), reference$evaluation$Cluster)
B <- 5000L
boots <- array(NA_real_, dim = c(B, nrow(performance), length(ci_metrics)),
               dimnames = list(NULL, NULL, ci_metrics))
for (b in seq_len(B)) {
  ix0 <- unlist(group[sample.int(length(group), length(group), replace = TRUE)], use.names = FALSE)
  for (i in seq_len(nrow(performance))) {
    rec <- performance[i]
    ix <- ix0[subsets[[rec$Population]][ix0]]
    boots[b, i, ] <- scores(yv[ix], pv[[rec$Model]][ix], selections(ix, rec$Model, rec$Rule))[ci_metrics]
  }
  if (b %% 500L == 0L) {cat("BOOTSTRAP", b, "OF", B, "\n"); flush.console()}
}
intervals <- rbindlist(lapply(seq_len(nrow(performance)), function(i) rbindlist(lapply(ci_metrics, function(v) {
  q <- quantile(boots[, i, v], c(.025, .975), na.rm = TRUE)
  data.table(Population = performance$Population[i], Rule = performance$Rule[i], Model = performance$Model[i],
    Metric = v, Estimate = performance[[v]][i], Lower95 = q[1], Upper95 = q[2])
}))))
contrasts <- list()
for (population in names(subsets)) for (rule in c("Training80", "Matched_testing")) for (name in names(forms)) {
  a <- which(performance$Population == population & performance$Rule == rule & performance$Model == "TabM9")
  c <- which(performance$Population == population & performance$Rule == rule & performance$Model == name)
  for (v in ci_metrics) {
    q <- quantile(boots[, a, v] - boots[, c, v], c(.025, .975), na.rm = TRUE)
    contrasts[[length(contrasts) + 1L]] <- data.table(Population = population, Rule = rule, Comparator = name,
      Metric = v, Difference = performance[[v]][a] - performance[[v]][c], Lower95 = q[1], Upper95 = q[2])
  }
}
contrasts <- rbindlist(contrasts)
training <- rbindlist(lapply(names(oof), function(name) as.data.table(c(list(Model = name, N = length(y),
  Threshold80 = thresholds[name]), as.list(scores(y, oof[[name]], oof[[name]] >= thresholds[name]))))))
fwrite(performance, file.path(out, "Clinical_reference_performance.csv"))
fwrite(intervals, file.path(out, "Clinical_reference_intervals.csv"))
fwrite(contrasts, file.path(out, "Clinical_reference_paired_differences.csv"))
fwrite(training, file.path(out, "Training_OOF_reference.csv"))
saveRDS(list(performance = performance, intervals = intervals, contrasts = contrasts, training = training,
  probabilities = pv, oof = oof, models = models, calibrators = calibrators, thresholds = thresholds,
  formulas = lapply(forms, deparse), seed = 123L, bootstrap = B,
  population = "Existing patient 70/30 split; no new renal or random-admission subgroup",
  benchmark = "Training80 thresholds are training-derived; matched capacity uses validation scores, not labels",
  reference_missingness = "Age, sex and creatinine complete; no reference imputation or repeated identical MI fits",
  matched_ties = "Deterministic row order; exact same number selected as frozen TabM9 within each population",
  uncertainty = "Paired patient-cluster bootstrap of frozen probabilities; no refitted optimism correction"),
  file.path(out, "Clinical_reference_results.rds"))
write_json(list(performance = performance, intervals = intervals, contrasts = contrasts, training = training,
  bootstrap = B, seed = 123L), file.path(out, "Clinical_reference_results.json"), pretty = TRUE, auto_unbox = TRUE, na = "null", digits = 16)
print(performance[Population == "All_validation", .(Rule, Model, AUC, Brier, Sensitivity, PPV, TestingFraction, TP, FP)])
cat("CLINICAL_REFERENCE_COMPLETE\n")
