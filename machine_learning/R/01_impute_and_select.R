source("shared/R/config.R")
options(encoding = "UTF-8", stringsAsFactors = FALSE, width = 150, warn = 1)
Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1")
suppressPackageStartupMessages({
    library(data.table)
    library(mice)
    library(glmnet)
})
setDTthreads(1L)
root <- getwd()
out <- file.path(root, "results")
prefix <- file.path(out, "ml_imputation")
seed <- 123L
M <- 30L
iterations <- 20L
workers <- 6L
RNGkind("L'Ecuyer-CMRG")
set.seed(seed)
mode <- if (length(commandArgs(TRUE))) commandArgs(TRUE)[1L] else "prepare"
note <- function(...) {
    cat(format(Sys.time(), "%F %T"), ..., "\n")
    flush.console()
}
prep_path <- paste0(prefix, "_inputs.rds")
p <- readRDS(prep_path)
stopifnot(p$settings$seed == 123L, length(p$rng_streams) == 23L)
mi_spec <- function(x) {
    meth <- setNames(ifelse(colSums(is.na(x)) > 0, "rf", ""), names(x))
    for (v in names(x)[vapply(x, is.factor, TRUE)]) if (anyNA(x[[v]])) 
        meth[v] <- if (is.ordered(x[[v]])) 
            "polr"
        else "logreg"
    pred <- matrix(1L, ncol(x), ncol(x), dimnames = list(names(x), names(x)))
    diag(pred) <- 0L
    white <- intersect(c("wbc", "neutrophils_abs", "lymphocytes_abs", "monocytes_abs", "nlr"), names(x))
    pred[white, white] <- 0L
    pred["neutrophils_abs", "wbc"] <- 1L
    pred["lymphocytes_abs", c("wbc", "neutrophils_abs")] <- 1L
    pred["monocytes_abs", c("wbc", "neutrophils_abs", "lymphocytes_abs")] <- 1L
    pred[, "nlr"] <- 0L
    if (anyNA(x$nlr)) 
        meth["nlr"] <- "~I(neutrophils_abs / lymphocytes_abs)"
    visit <- c(setdiff(names(meth)[meth != ""], "nlr"), if (meth["nlr"] != "") "nlr")
    list(method = meth, predictor = pred, visit = visit)
}
fit_mi <- function(x, m, s, maxit = iterations) {
    assign(".Random.seed", s, envir = .GlobalEnv)
    init <- x
    for (v in names(x)) {
        zz <- x[[v]][!is.na(x[[v]])]
        stopifnot(length(zz) > 0)
        fill <- if (is.factor(zz)) 
            names(which.max(table(zz)))
        else median(zz)
        init[[v]][is.na(init[[v]])] <- fill
    }
    a <- mi_spec(x)
    mice::mice(x, m = m, maxit = maxit, method = a$method, predictorMatrix = a$predictor, visitSequence = a$visit, 
        data.init = init, seed = NA, printFlag = FALSE, rfPackage = "ranger", ntree = 10L, num.threads = 1L, polr.to.loggedEvents = TRUE, 
        remove.constant = FALSE, remove.collinear = FALSE)
}
check_complete <- function(x, raw) {
    stopifnot(identical(names(x), names(raw)), nrow(x) == nrow(raw), !anyNA(x))
    for (v in names(x)) {
        ii <- !is.na(raw[[v]])
        stopifnot(identical(x[[v]][ii], raw[[v]][ii]))
        if (is.factor(x[[v]])) 
            stopifnot(all(as.character(x[[v]]) %in% levels(raw[[v]])))
        else stopifnot(all(is.finite(x[[v]])))
    }
    stopifnot(all(x$neutrophils_abs > 0), all(x$lymphocytes_abs > 0), all(x$nlr > 0))
    TRUE
}
mi_job <- function(job) {
    dest <- paste0(prefix, "_", job$partition, sprintf("_chunk%02d.rds", job$chunk))
    if (file.exists(dest)) {
        note("MICE_RESUME", job$partition, job$chunk)
        return(dest)
    }
    raw <- if (job$partition == "Train") 
        p$train
    else p$evaluation_raw
    note("MICE_RF_START", job$partition, job$chunk, "ROWS", nrow(raw))
    t0 <- proc.time()[3L]
    imp <- fit_mi(raw, 5L, p$rng_streams[[paste0(job$partition, job$chunk)]])
    for (k in seq_len(imp$m)) check_complete(mice::complete(imp, k), raw)
    saveRDS(list(partition = job$partition, chunk = job$chunk, imputation = imp, seconds = unname(proc.time()[3L] - 
        t0)), dest)
    note("MICE_RF_DONE", job$partition, job$chunk, "SECONDS", round(proc.time()[3L] - t0))
    dest
}
start_cluster <- function(exports) {
    cl <- parallel::makePSOCKcluster(workers, outfile = paste0(prefix, "_workers.log"))
    parallel::clusterEvalQ(cl, {
        options(encoding = "UTF-8", stringsAsFactors = FALSE, warn = 1)
        Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1")
        suppressPackageStartupMessages({
            library(data.table)
            library(mice)
            library(glmnet)
        })
        setDTthreads(1L)
        NULL
    })
    parallel::clusterExport(cl, exports, envir = .GlobalEnv)
    cl
}
chain_summary <- function(imp) {
    rbindlist(lapply(c("chainMean", "chainVar"), function(stat) {
        z <- imp[[stat]]
        rbindlist(lapply(dimnames(z)[[1L]], function(v) {
            mat <- z[v, , ]
            if (!any(is.finite(mat))) 
                return(NULL)
            mat <- tail(mat, max(2L, imp$iteration%/%2L))
            n <- nrow(mat)
            w <- mean(apply(mat, 2L, var))
            b <- n * var(colMeans(mat))
            rh <- if (w > 0) 
                sqrt(((n - 1) * w/n + b/n)/w)
            else if (b == 0) 
                1
            else Inf
            data.table(Variable = v, Statistic = stat, Rhat = rh, IterationsUsed = n, Chains = ncol(mat))
        }))
    }))
}
combine_partition <- function(partition) {
    src <- lapply(seq_len(6L), function(j) readRDS(paste0(prefix, "_", partition, sprintf("_chunk%02d.rds", j))))
    source("shared/R/mice_bind.R")
    imp <- Reduce(safe_ibind, lapply(src, `[[`, "imputation"))
    raw <- if (partition == "Train") 
        p$train
    else p$evaluation_raw
    stopifnot(imp$m == 30L, identical(imp$data, raw))
    completed <- lapply(seq_len(M), function(m) {
        a <- mice::complete(imp, m)
        check_complete(a, raw)
        ref <- mice::complete(src[[(m - 1L)%/%5L + 1L]]$imputation, (m - 1L)%%5L + 1L)
        stopifnot(identical(a, ref))
        a
    })
    dist <- rbindlist(lapply(names(raw), function(v) {
        a <- raw[[v]]
        ii <- is.na(a)
        if (is.factor(a)) 
            return(data.table(Variable = v, Type = "categorical", Level = levels(a), ObservedPct = 100 * as.numeric(prop.table(table(a))), 
                ImputedPct = if (any(ii)) 100 * as.numeric(prop.table(table(factor(unlist(lapply(completed, function(d) as.character(d[[v]][ii]))), 
                  levels = levels(a))))) else NA_real_))
        vv <- if (any(ii)) 
            unlist(lapply(completed, function(d) d[[v]][ii]))
        else numeric()
        data.table(Variable = v, Type = "continuous", ObservedMean = mean(a, na.rm = TRUE), ObservedSD = sd(a, 
            na.rm = TRUE), ImputedMean = if (length(vv)) 
            mean(vv)
        else NA_real_, ImputedSD = if (length(vv)) 
            sd(vv)
        else NA_real_, ObservedMin = min(a, na.rm = TRUE), ObservedMax = max(a, na.rm = TRUE), ImputedMin = if (length(vv)) 
            min(vv)
        else NA_real_, ImputedMax = if (length(vv)) 
            max(vv)
        else NA_real_)
    }), fill = TRUE)
    diag <- list(convergence = chain_summary(imp), distribution = dist, loggedEvents = imp$loggedEvents, methods = imp$method, 
        predictors = imp$predictorMatrix, partition = partition)
    saveRDS(imp, paste0(prefix, "_", partition, "_mids.rds"))
    saveRDS(list(completed = completed, diagnostics = diag, m = M, settings = p$settings, source = basename(prep_path)), 
        paste0(prefix, "_", partition, "_completed.rds"))
    note("MICE_RF_COMBINED", partition, "MAX_CHAIN_RHAT", round(max(diag$convergence$Rhat, na.rm = TRUE), 3), "LOGGED_EVENTS", 
        if (is.null(imp$loggedEvents)) 
            0L
        else nrow(imp$loggedEvents))
    invisible(diag)
}
if (mode == "prepare") quit(save = "no")
if (mode == "combine_train") {
    combine_partition("Train")
    quit(save = "no")
}
if (mode %in% c("benchmark", "all")) {
    dest <- paste0(prefix, "_benchmark.rds")
    if (!file.exists(dest)) {
        t0 <- proc.time()[3L]
        a <- fit_mi(p$train, 1L, p$rng_streams$benchmark, maxit = 2L)
        check_complete(mice::complete(a, 1L), p$train)
        saveRDS(list(seconds = unname(proc.time()[3L] - t0), rows = nrow(p$train), m = 1L, maxit = 2L, methods = a$method, 
            loggedEvents = a$loggedEvents, purpose = "Full-size feasibility test, NEVER used as a formal imputation dataset"), 
            dest)
        note("FULL_SIZE_RF_BENCHMARK_SECONDS", round(proc.time()[3L] - t0))
    }
    if (mode == "benchmark") 
        quit(save = "no")
}
if (mode %in% c("impute", "all")) {
    jobs <- unlist(lapply(c("Train", "Evaluation"), function(a) lapply(seq_len(6L), function(j) list(partition = a, 
        chunk = j))), recursive = FALSE)
    cl <- start_cluster(c("p", "prefix", "seed", "iterations", "mi_spec", "fit_mi", "check_complete", "mi_job", 
        "note"))
    tryCatch(parallel::parLapplyLB(cl, jobs, mi_job), finally = parallel::stopCluster(cl))
    combine_partition("Train")
    combine_partition("Evaluation")
}
inner_job <- function(job) {
    dest <- paste0(prefix, "_Inner", job$fold, "_", job$partition, ".rds")
    if (file.exists(dest)) 
        return(dest)
    ix <- if (job$partition == "Train") 
        which(p$foldid != job$fold)
    else which(p$foldid == job$fold)
    raw <- p$train[ix, , drop = FALSE]
    t0 <- proc.time()[3L]
    imp <- fit_mi(raw, 5L, p$rng_streams[[paste0("Inner", job$fold, job$partition)]])
    dd <- lapply(seq_len(5L), function(k) {
        x <- mice::complete(imp, k)
        check_complete(x, raw)
        x
    })
    saveRDS(list(completed = dd, rows = ix, imputation = imp, partition = job$partition, fold = job$fold, query_label_used = FALSE, 
        seconds = unname(proc.time()[3L] - t0)), dest)
    note("INNER_RF_DONE", job$fold, job$partition, "SECONDS", round(proc.time()[3L] - t0))
    dest
}
if (mode %in% c("impute_inner", "all")) {
    jobs <- unlist(lapply(seq_len(5L), function(k) lapply(c("Train", "Query"), function(a) list(fold = k, partition = a))), 
        recursive = FALSE)
    cl <- start_cluster(c("p", "prefix", "seed", "iterations", "mi_spec", "fit_mi", "check_complete", "inner_job", 
        "note"))
    tryCatch(parallel::parLapplyLB(cl, jobs, inner_job), finally = parallel::stopCluster(cl))
}
lasso_job <- function(k) {
    dest <- paste0(prefix, sprintf("_LASSO_%02d.rds", k))
    if (file.exists(dest)) 
        return(dest)
    x <- training$completed[[k]][, p$features, drop = FALSE]
    x$proteinuria <- factor(x$proteinuria, levels = c("A1", "A2", "A3"), ordered = FALSE)
    set.seed(seed)
    t0 <- proc.time()[3L]
    mm <- model.matrix(reformulate(p$features), x)
    map <- p$features[attr(mm, "assign")[-1L]]
    xx <- mm[, -1L, drop = FALSE]
    b <- glmnet::cv.glmnet(xx, p$y, family = "binomial", alpha = 1, foldid = p$original_foldid, lambda = exp(seq(log(100), 
        log(1e-05), length.out = 180L)), type.measure = "deviance", standardize = TRUE, maxit = 100000L, parallel = FALSE)
    selected <- function(s) {
        bb <- as.matrix(coef(b, s = s))[-1L, 1L]
        unique(map[bb != 0])
    }
    saveRDS(list(imputation = k, fit = b, lambda1se = b$lambda.1se, lambdamin = b$lambda.min, selected1se = selected("lambda.1se"), 
        selectedmin = selected("lambda.min"), columns = setNames(map, colnames(xx)), seconds = unname(proc.time()[3L] - 
            t0), note = "Ten fixed patient-grouped folds select lambda; MI precedes this selector CV; not an unbiased performance estimate"), 
        dest)
    note("LASSO_RF_DONE", k, "N1SE", length(selected("lambda.1se")))
    dest
}
if (mode %in% c("lasso", "all")) {
    training <- readRDS(paste0(prefix, "_Train_completed.rds"))
    cl <- start_cluster(c("p", "training", "prefix", "seed", "lasso_job", "note"))
    tryCatch(parallel::parLapplyLB(cl, seq_len(M), lasso_job), finally = parallel::stopCluster(cl))
}
if (mode %in% c("export", "all")) {
    files <- paste0(prefix, sprintf("_LASSO_%02d.rds", seq_len(M)))
    done <- which(file.exists(files))
    bt <- if (length(done)) 
        rbindlist(lapply(files[done], function(f) {
            a <- readRDS(f)
            data.table(Variable = p$features, Imputation = a$imputation, LASSO1SE = p$features %in% a$selected1se, 
                LASSOMIN = p$features %in% a$selectedmin, Lambda1SE = a$lambda1se, LambdaMIN = a$lambdamin)
        }))
    else data.table()
    freq <- data.table(Variable = p$features)
    if (nrow(bt)) {
        ff <- bt[, .(Completed = .N, LASSO1SE_n = sum(LASSO1SE), LASSOMIN_n = sum(LASSOMIN)), by = Variable]
        freq <- merge(freq, ff, by = "Variable", all.x = TRUE, sort = FALSE)
        freq[, `:=`(Selected18 = if (length(done) == M) 
            LASSO1SE_n >= 18L
        else NA, Stable30 = if (length(done) == M) 
            LASSO1SE_n == 30L
        else NA)]
    }
    settings <- data.table(Item = c("Status", "Source", "Candidate count", "Prior retention", "Imputation", "Auxiliary only", 
        "Development and validation"), Detail = c(if (length(done) == M) "LASSO_COMPLETE" else sprintf("LASSO_%d_OF_30_PENDING", 
        length(done)), paste(p$source, collapse = "; "), "34; RBC/neutrophil count not independently eligible", 
        p$settings$prior_rule, "Train and evaluation separately m=30; continuous RF; no outcomes/labels in either imputer", 
        "Neutrophils only support passive missing NLR; not eligible as a model input", "Patient-level split; five development folds for tuning and model comparison; reduction within training folds"))
    tabs <- list(Methods = settings, Candidates_missingness = p$missingness, LASSO_frequency = freq)
    for (a in c("Train", "Evaluation")) {
        f <- paste0(prefix, "_", a, "_completed.rds")
        if (file.exists(f)) {
            d <- readRDS(f)$diagnostics
            tabs[[paste0(a, "_MI_methods")]] <- data.table(Variable = names(d$methods), Method = unname(d$methods))
            tabs[[paste0(a, "_MI_chains")]] <- d$convergence
            tabs[[paste0(a, "_MI_distribution")]] <- d$distribution
            if (!is.null(d$loggedEvents)) 
                tabs[[paste0(a, "_MI_events")]] <- as.data.table(d$loggedEvents)
        }
    }
    if (nrow(bt)) 
        tabs$LASSO_per_imputation <- bt
    if (length(done) == M) {
        selected <- freq[Selected18 == TRUE, Variable]
        stopifnot(length(selected) > 0)
        saveRDS(list(selected18 = selected, stable30 = freq[Stable30 == TRUE, Variable], frequency = freq, no_forced_predictors = TRUE, 
            source = basename(prep_path)), paste0(prefix, "_selection.rds"))
    }
    openxlsx::write.xlsx(tabs, paste0(prefix, "_results.xlsx"), overwrite = TRUE)
    rb <- readxl::read_excel(paste0(prefix, "_results.xlsx"), sheet = "Candidates_missingness")
    stopifnot(nrow(rb) == 36L, sum(rb$Candidate) == 34L)
    saveRDS(list(tables = tabs, lasso_completed = length(done), settings = p$settings, created = Sys.time()), paste0(prefix, 
        "_results.rds"))
    note("EXPORTED", length(done), "LASSO FITS; PRIOR RESULTS AND PROGNOSTIC MI UNCHANGED")
}
