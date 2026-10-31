# Prognostic table
source("shared/R/config.R")
library(data.table)
a <- readRDS("results/prognosis_results.rds")
models <- copy(a$pooled)
adjustments <- c("Unadjusted", "Demographic adjusted", "Fully adjusted")
outcomes <- c(KF = "Kidney failure", CCE = "Composite cardiovascular events", Mortality = "All-cause mortality")
groups <- c(Negative = "≤−30%", Reference = ">−30% to <30%", Positive = "≥30%")
fmtp <- function(x) ifelse(x < .001, "<0.001", sprintf("%.3f", x))
body <- rbindlist(lapply(names(outcomes), function(oc) rbindlist(lapply(names(groups), function(g) {
  row <- data.table(Outcome = outcomes[[oc]], Category = groups[[g]])
  for (j in seq_along(adjustments)) {
    z <- models[Outcome == oc & Model == adjustments[j] & Contrast == g]
    if (g == "Reference") {
      est <- "Ref"; pv <- "—"
    } else {
      stopifnot(nrow(z) == 1L)
      est <- sprintf("%.2f (%.2f–%.2f)", z$Estimate, z$Lower95, z$Upper95)
      pv <- fmtp(z$p_value)
    }
    row[, (paste0("HR", j)) := est]
    row[, (paste0("P", j)) := pv]
  }
  row
}))))
saveRDS(list(body = body, models = models, counts = a$counts), "results/prognosis_models_table.rds")
fwrite(body, "results/prognosis_models_table.csv")
