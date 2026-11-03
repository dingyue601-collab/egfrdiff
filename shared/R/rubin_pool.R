source("shared/R/config.R")
manual_rubin_pool <- function(q, u, conf_level = 0.95) {
    q <- as.numeric(q)
    u <- as.numeric(u)
    ok <- is.finite(q) & is.finite(u) & u >= 0
    q <- q[ok]
    u <- u[ok]
    m <- length(q)
    if (m == 0) {
        return(tibble::tibble(m_success = 0L, estimate_pool_scale = NA_real_, se_pool_scale = NA_real_, df = NA_real_, 
            lower_pool_scale = NA_real_, upper_pool_scale = NA_real_, p_value = NA_real_, within_var = NA_real_, 
            between_var = NA_real_, total_var = NA_real_, lambda = NA_real_))
    }
    qbar <- mean(q)
    ubar <- mean(u)
    between <- if (m > 1) 
        stats::var(q)
    else 0
    total <- ubar + (1 + 1/m) * between
    se <- sqrt(total)
    if (!is.finite(between) || between <= .Machine$double.eps || !is.finite(ubar) || ubar <= 0) {
        df <- Inf
    }
    else {
        r <- ((1 + 1/m) * between)/ubar
        df <- (m - 1) * (1 + 1/r)^2
    }
    alpha <- 1 - conf_level
    crit <- if (is.finite(df)) {
        stats::qt(1 - alpha/2, df = df)
    }
    else {
        stats::qnorm(1 - alpha/2)
    }
    stat <- qbar/se
    p <- if (is.finite(df)) {
        2 * stats::pt(abs(stat), df = df, lower.tail = FALSE)
    }
    else {
        2 * stats::pnorm(abs(stat), lower.tail = FALSE)
    }
    lambda <- if (total > 0) {
        ((1 + 1/m) * between)/total
    }
    else {
        0
    }
    tibble::tibble(m_success = as.integer(m), estimate_pool_scale = qbar, se_pool_scale = se, df = df, lower_pool_scale = qbar - 
        crit * se, upper_pool_scale = qbar + crit * se, p_value = p, within_var = ubar, between_var = between, 
        total_var = total, lambda = lambda)
}
