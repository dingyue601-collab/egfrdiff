source("shared/R/config.R")
auc <- function(y, z) {
    n1 <- sum(y == 1L)
    n0 <- sum(y == 0L)
    stopifnot(n1 > 0, n0 > 0, all(is.finite(z)))
    (sum(rank(z)[y == 1L]) - n1 * (n1 + 1)/2)/(n1 * n0)
}
ap <- function(y, z) {
    a <- data.table(y = y, p = z)[, .(Events = sum(y), N = .N), by = p][order(-p)]
    sum(a$Events/sum(y) * cumsum(a$Events)/cumsum(a$N))
}
make_design <- function(tr, qu) {
    a <- model.matrix(~., tr)[, -1L, drop = FALSE]
    b <- model.matrix(~., qu)[, -1L, drop = FALSE]
    stopifnot(identical(colnames(a), colnames(b)), all(is.finite(a)), all(is.finite(b)))
    mu <- colMeans(a)
    ss <- apply(a, 2L, sd)
    ss[!is.finite(ss) | ss <= 0] <- 1
    list(train = a, query = b, center = mu, scale = ss, scaled_train = sweep(sweep(a, 2L, mu, "-"), 2L, ss, "/"), 
        scaled_query = sweep(sweep(b, 2L, mu, "-"), 2L, ss, "/"))
}
fit_one <- function(family, pa, bn, m, keep = FALSE) {
    d <- data_blocks[[bn]][[m]]
    y <- p$blocks[[bn]]$ytrain
    set.seed(seed + m)
    ww <- character()
    tp <- record <- NULL
    z <- withCallingHandlers({
        if (family == "Logistic") {
            aa <- cbind(Intercept = 1, d$scaled_train)
            bb <- cbind(Intercept = 1, d$scaled_query)
            f <- glm.fit(aa, y, family = binomial(), control = glm.control(maxit = 100L))
            stopifnot(f$converged, f$rank == ncol(aa), all(is.finite(f$coefficients)))
            z <- plogis(drop(bb %*% f$coefficients))
            if (keep) {
                tp <- f$fitted.values
                record <- list(coefficients = f$coefficients, center = d$center, scale = d$scale, columns = colnames(d$train))
            }
        }
        else if (family == "XGBoost") {
            f <- xgboost::xgb.train(list(objective = "binary:logistic", eval_metric = "auc", tree_method = "hist", 
                device = compute$device, nthread = 2L, max_depth = as.integer(pa$depth), eta = pa$eta, min_child_weight = pa$min_child, 
                subsample = pa$subsample, colsample_bytree = pa$colsample, lambda = pa$l2, alpha = pa$l1, gamma = pa$gamma, 
                scale_pos_weight = pa$weight, seed = seed + m), xgboost::xgb.DMatrix(d$train, label = y, nthread = 1L), 
                nrounds = as.integer(pa$rounds), verbose = 0L)
            z <- as.numeric(predict(f, xgboost::xgb.DMatrix(d$query, nthread = 1L)))
            if (keep) {
                tp <- as.numeric(predict(f, xgboost::xgb.DMatrix(d$train, nthread = 1L)))
                raw <- xgboost::xgb.save.raw(f, raw_format = "ubj")
                rr <- xgboost::xgb.load.raw(raw)
                stopifnot(max(abs(predict(rr, xgboost::xgb.DMatrix(d$query, nthread = 1L)) - z)) < 1e-07)
                record <- list(model_raw = raw, columns = colnames(d$train))
            }
        }
        else if (family == "RF") {
            x <- as.data.frame(d$train)
            q <- as.data.frame(d$query)
            x$.label <- factor(y, levels = 0:1)
            mt <- max(1L, min(ncol(d$train), round(pa$mtry_fraction * ncol(d$train))))
            f <- ranger::ranger(.label ~ ., x, probability = TRUE, num.trees = as.integer(pa$trees), mtry = mt, 
                min.node.size = as.integer(pa$node), sample.fraction = pa$fraction, replace = TRUE, class.weights = c(1, 
                  pa$weight), num.threads = 2L, seed = seed + m, write.forest = TRUE)
            z <- predict(f, q, num.threads = 2L)$predictions[, "1"]
            if (keep) {
                tp <- predict(f, x, num.threads = 2L)$predictions[, "1"]
                record <- list(model = f, columns = colnames(d$train))
            }
        }
        else if (family == "LightGBM") {
            f <- lightgbm::lgb.train(list(objective = "binary", metric = "auc", verbosity = -1L, num_threads = 2L, 
                device_type = compute$lightgbm_device, learning_rate = pa$eta, num_leaves = as.integer(pa$leaves), 
                max_depth = as.integer(pa$depth), min_data_in_leaf = as.integer(pa$node), feature_fraction = pa$colsample, 
                bagging_fraction = pa$subsample, bagging_freq = 1L, lambda_l1 = pa$l1, lambda_l2 = pa$l2, scale_pos_weight = pa$weight, 
                seed = seed + m, deterministic = TRUE, force_col_wise = TRUE), lightgbm::lgb.Dataset(d$train, label = y), 
                nrounds = as.integer(pa$rounds))
            z <- as.numeric(predict(f, d$query))
            if (keep) {
                tp <- as.numeric(predict(f, d$train))
                record <- list(model_string = f$save_model_to_string(), columns = colnames(d$train))
            }
        }
        else if (family == "SVM") {
            f <- e1071::svm(x = d$scaled_train, y = factor(y, levels = 0:1), type = "C-classification", kernel = pa$kernel, 
                cost = pa$cost, gamma = pa$gamma, scale = FALSE, probability = TRUE, cachesize = 200, class.weights = c(`0` = 1, 
                  `1` = pa$weight), tolerance = 0.001)
            z <- attr(predict(f, d$scaled_query, probability = TRUE), "probabilities")[, "1"]
            if (keep) {
                tp <- attr(predict(f, d$scaled_train, probability = TRUE), "probabilities")[, "1"]
                record <- list(model = f, center = d$center, scale = d$scale, columns = colnames(d$train))
            }
        }
        else stop("Unknown family")
        as.numeric(z)
    }, warning = function(w) {
        ww <<- c(ww, conditionMessage(w))
        invokeRestart("muffleWarning")
    })
    stopifnot(length(z) == nrow(d$query), all(is.finite(z)), all(z >= 0 & z <= 1))
    list(prediction = z, training = tp, record = record, warnings = unique(ww))
}
benchmark <- function() {
    d <- data_blocks$Full[[1L]]
    a <- list()
    for (dev in c("cpu", "cuda")) {
        ww <- character()
        err <- NULL
        actual <- FALSE
        sec <- system.time({
            f <- tryCatch(withCallingHandlers(xgboost::xgb.train(list(objective = "binary:logistic", tree_method = "hist", 
                device = dev, nthread = 2L, max_depth = 3L, eta = 0.05, seed = seed), xgboost::xgb.DMatrix(d$train, 
                label = p$blocks$Full$ytrain, nthread = 1L), nrounds = 100L, verbose = 0L), warning = function(w) {
                ww <<- c(ww, conditionMessage(w))
                invokeRestart("muffleWarning")
            }), error = function(e) {
                err <<- conditionMessage(e)
                NULL
            })
        })[["elapsed"]]
        if (!is.null(f)) {
            js <- jsonlite::toJSON(xgboost::xgb.config(f), auto_unbox = TRUE)
            actual <- grepl("cuda", js, fixed = TRUE) && !length(grep("not compiled|No visible GPU", ww))
        }
        a[[dev]] <- list(seconds = sec, actual_cuda = actual, warnings = unique(ww), error = err)
    }
    lb <- list()
    for (dev in c("cpu", "gpu")) {
        err <- NULL
        sec <- system.time({
            f <- tryCatch(lightgbm::lgb.train(list(objective = "binary", metric = "auc", verbosity = -1L, num_threads = 2L, 
                device_type = dev, num_leaves = 7L, max_depth = 3L, seed = seed), lightgbm::lgb.Dataset(d$train, 
                label = p$blocks$Full$ytrain), nrounds = 100L), error = function(e) {
                err <<- conditionMessage(e)
                NULL
            })
        })[["elapsed"]]
        lb[[dev]] <- list(seconds = sec, error = err, available = !is.null(f))
    }
    list(device = if (isTRUE(a$cuda$actual_cuda) && a$cuda$seconds < a$cpu$seconds) "cuda" else "cpu", lightgbm_device = if (isTRUE(lb$gpu$available) && 
        lb$gpu$seconds < lb$cpu$seconds) "gpu" else "cpu", benchmark = a, lightgbm_benchmark = lb, reason = "GPU requested and measured; installed R builds may be CPU-only. Foundation models use existing Python CUDA.")
}
