source("shared/R/config.R")
safe_ibind <- function(x, y) {
    normalize <- function(z) {
        for (v in names(z$data)) {
            if (is.null(z$imp[[v]])) {
                stopifnot(z$nmis[v] == 0L)
                z$imp[v] <- list(as.data.frame(matrix(numeric(0), nrow = 0L, ncol = z$m, dimnames = list(NULL, 
                  as.character(seq_len(z$m))))))
            }
        }
        stopifnot(identical(names(z$imp), names(z$data)), length(z$imp) == ncol(z$data))
        z
    }
    z <- mice::ibind(normalize(x), normalize(y))
    stopifnot(identical(names(z$imp), names(z$data)), length(z$imp) == ncol(z$data))
    z
}
