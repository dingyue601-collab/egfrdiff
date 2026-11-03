# Prognostic imputation and regression
source("shared/R/config.R")
suppressPackageStartupMessages({library(data.table); library(mice); library(cmprsk); library(survival)})
setDTthreads(1L)
RNGkind("L'Ecuyer-CMRG")
set.seed(123L)
mode <- if (length(commandArgs(TRUE))) commandArgs(TRUE)[1L] else "all"
stopifnot(mode %in% c("prepare", "impute", "fit", "all"))
prefix <- "results/prognosis"
p <- readRDS(paste0(prefix, "_inputs.rds"))
mi_file <- paste0(prefix, "_MICE20.rds")
M <- 20L
source("shared/R/rubin_pool.R")

check_observed <- function(x) {
  stopifnot(identical(names(x), names(p$input)), nrow(x) == nrow(p$input),
            !anyNA(x[, p$imputed_variables, drop = FALSE]))
  for (v in names(x)) {
    i <- !is.na(p$input[[v]])
    stopifnot(identical(x[[v]][i], p$input[[v]][i]))
  }
}

mi_worker <- function(j) {
  path <- paste0(prefix, sprintf("_MIchunk%02d.rds", j))
  if (file.exists(path)) return(path)
  assign(".Random.seed", p$streams[[j]], envir = .GlobalEnv)
  a <- mice::mice(p$input, m = 5L, maxit = 20L, seed = NA, method = p$method,
    predictorMatrix = p$predictor, visitSequence = p$visit, printFlag = FALSE,
    rfPackage = "ranger", ntree = 10L, num.threads = 1L, polr.to.loggedEvents = TRUE)
  saveRDS(a, path)
  path
}

chain_summary <- function(mi) {
  rbindlist(lapply(c("chainMean", "chainVar"), function(stat) {
    z <- mi[[stat]]
    rbindlist(lapply(dimnames(z)[[1L]], function(v) {
      x <- z[v, , ]
      if (!any(is.finite(x))) return(NULL)
      x <- tail(x, max(2L, mi$iteration %/% 2L))
      n <- nrow(x); w <- mean(apply(x, 2L, var)); b <- n * var(colMeans(x))
      rhat <- if (w > 0) sqrt(((n - 1) * w / n + b / n) / w) else if (b == 0) 1 else Inf
      data.table(Variable = v, Statistic = stat, Rhat = rhat)
    }))
  }))
}

if (mode == "prepare") quit(save = "no")
if (mode %in% c("impute", "all") && !file.exists(mi_file)) {
  cl <- parallel::makePSOCKcluster(4L)
  parallel::clusterEvalQ(cl, {
    Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1")
    suppressPackageStartupMessages(library(mice))
    NULL
  })
  parallel::clusterExport(cl, c("p", "prefix", "mi_worker"), envir = .GlobalEnv)
  paths <- tryCatch(parallel::parLapply(cl, 1:4, mi_worker), finally = parallel::stopCluster(cl))
  source("shared/R/mice_bind.R")
  mi <- Reduce(safe_ibind, lapply(paths, readRDS))
  for (k in seq_len(M)) check_observed(mice::complete(mi, k))
  saveRDS(list(mids = mi, imputation_input = p$input, source_data = p$source_data,
    imputed_variables = p$imputed_variables, diagnostics = list(convergence = chain_summary(mi),
      loggedEvents = mi$loggedEvents, method = mi$method, predictor = mi$predictorMatrix),
    settings = p$settings, sources = p$sources), mi_file)
}
if (mode == "impute") quit(save = "no")

# Death is a competing event for nonfatal outcomes, not for all-cause mortality.
master <- readRDS(mi_file)
stopifnot(master$mids$m == M, identical(master$imputation_input, p$input),
  all(master$mids$method[c("hba1c", "sbp", "ldl_c")] == "rf"))
clinical <- c("age", "sex", "egfrcr", "proteinuria", "hba1c", "sbp", "cci",
              "ldl_c", "hypertension_rechecked", "smoking_rechecked")
adjustments <- c("Unadjusted", "Demographic adjusted", "Fully adjusted")
terms <- c("egfrdiff_groupNegative", "egfrdiff_groupPositive")
checkpoint <- paste0(prefix, "_FineGray_checkpoint.rds")
raw <- fits <- list()
if (file.exists(checkpoint)) {
  z <- base::readRDS(checkpoint)
  raw <- z$raw; fits <- z$fits
}
for (k in seq_len(M)) {
  complete <- mice::complete(master$mids, k)
  check_observed(complete)
  d <- copy(p$source_data)
  for (v in p$imputed_variables) d[, (v) := complete[[v]][match(.row_id, complete$.row_id)]]
  d[, proteinuria := factor(as.character(proteinuria), levels = c("A1", "A2", "A3"))]
  for (oc in c("KF", "CCE", "Mortality")) for (model in adjustments) {
    key <- paste(oc, model, k, sep = "|")
    if (!is.null(raw[[key]])) next
    rr <- p$risks[[oc]]
    dat <- copy(d[match(rr$row_index, row_index)])
    vars <- c("egfrdiff_group", switch(model, "Unadjusted" = character(),
      "Demographic adjusted" = c("age", "sex"),
      "Fully adjusted" = c(clinical, if (oc == "CCE") "PriorCCE")))
    x <- model.matrix(reformulate(vars), dat)[, -1L, drop = FALSE]
    stopifnot(identical(as.double(dat$row_index), as.double(rr$row_index)),
      !anyDuplicated(rr$row_index), all(rr$Event + rr$CompetingDeath <= 1L),
      all(rr$FollowupYears > 0), all(rr$EndDate <= as.Date("2025-06-30")),
      nrow(x) == nrow(rr), all(is.finite(x)), qr(x)$rank == ncol(x),
      identical(levels(dat$egfrdiff_group), c("Reference", "Negative", "Positive")))
    if (oc == "Mortality") {
      dat[, `:=`(time = rr$FollowupYears * 12, event = rr$Event)]
      fit <- survival::coxph(reformulate(vars, "survival::Surv(time, event)"), dat, ties = "efron")
      b <- coef(fit); v <- vcov(fit); method <- "Cox HR"
    } else {
      status <- ifelse(rr$Event == 1L, 1L, ifelse(rr$CompetingDeath == 1L, 2L, 0L))
      fit <- cmprsk::crr(rr$FollowupYears * 12, status, cov1 = x,
                         failcode = 1L, cencode = 0L, maxiter = 100L)
      stopifnot(fit$converged)
      b <- fit$coef; v <- fit$var; method <- "Fine-Gray sHR"
    }
    i <- match(terms, names(b))
    stopifnot(!anyNA(i), all(is.finite(b)), all(is.finite(v)), all(diag(v) > 0))
    raw[[key]] <- data.table(Imputation = k, Outcome = oc, Model = model, Method = method,
      Contrast = c("Negative", "Positive"), Beta = as.numeric(b[i]), Variance = as.numeric(diag(v)[i]),
      N = nrow(rr), Events = sum(rr$Event), CompetingDeaths = sum(rr$CompetingDeath))
    fits[[key]] <- list(coefficients = b, variance = v, covariates = vars)
    saveRDS(list(raw = raw, fits = fits), checkpoint)
    cat(sprintf("Imputation %02d/20 | %s | %s\n", k, oc, model)); flush.console()
  }
}
per_mi <- rbindlist(raw)
pooled <- per_mi[, {
  z <- as.data.table(manual_rubin_pool(Beta, Variance))
  z[, `:=`(N = unique(N), Events = unique(Events), CompetingDeaths = unique(CompetingDeaths))]
  z
}, by = .(Outcome, Model, Method, Contrast)]
pooled[, `:=`(Estimate = exp(estimate_pool_scale), Lower95 = exp(lower_pool_scale), Upper95 = exp(upper_pool_scale))]
stopifnot(nrow(per_mi) == 360L, nrow(pooled) == 18L, all(pooled$m_success == M))
saveRDS(list(pooled = pooled, per_imputation = per_mi, fits = fits, counts = p$counts,
  covariates = clinical, seed = 123L, m = M), paste0(prefix, "_results.rds"))
fwrite(pooled, paste0(prefix, "_estimates.csv"))
print(pooled[Model == "Fully adjusted", .(Outcome, Contrast, N, Events, Estimate, Lower95, Upper95, p_value)])
