source("shared/R/config.R")
options(encoding = "UTF-8", stringsAsFactors = FALSE)
suppressPackageStartupMessages({
    library(data.table)
    library(survey)
})
setDTthreads(1L)
set.seed(123)
base <- file.path(normalizePath(getwd(), winslash = "/"), "results")
out <- file.path(base, "metrics")
saved <- readRDS(file.path(base, "baseline_split.rds"))
cohort <- readRDS(file.path(base, "patient_split.rds"))
make <- function(tab, dd) {
    d <- as.data.table(dd)
    stat <- rbindlist(lapply(seq_len(nrow(tab$spec)), function(j) {
        sp <- tab$spec[j]
        v <- sp$Variable
        z <- data.frame(Cluster = as.character(d$PatientID), Partition = factor(d$DevelopmentPartition), Value = d[[v]])
        z <- z[!is.na(z$Value), , drop = FALSE]
        if (sp$Type %in% c("binary", "category")) 
            z$Value <- droplevels(factor(z$Value))
        method <- if (sp$Type %in% c("binary", "category")) 
            "Rao-Scott F (patient cluster)"
        else "Survey-adjusted Wilcoxon (patient cluster)"
        warning <- character()
        err <- ""
        pv <- tryCatch(withCallingHandlers({
            design <- svydesign(ids = ~Cluster, weights = ~1, data = z)
            if (length(unique(z$Value)) < 2L) 
                NA_real_
            else if (sp$Type %in% c("binary", "category")) 
                as.numeric(svychisq(~Value + Partition, design, statistic = "F")$p.value)
            else as.numeric(svyranktest(Value ~ Partition, design, test = "wilcoxon")$p.value)
        }, warning = function(w) {
            warning <<- c(warning, conditionMessage(w))
            invokeRestart("muffleWarning")
        }), error = function(e) {
            err <<- conditionMessage(e)
            NA_real_
        })
        data.table(Variable = v, P = pv, Method = method, ObservedN = nrow(z), Patients = uniqueN(z$Cluster), Warning = paste(unique(warning), 
            collapse = "; "), Error = err)
    }))
    stopifnot(all(stat$Error == ""), all(is.na(stat$P) | stat$P >= 0 & stat$P <= 1))
    tb <- copy(tab$data)
    tb[, `:=`(`P value`, rep("", .N))]
    for (v in stat$Variable) {
        rows <- which(tab$metadata$Variable == v & !tab$metadata$Section)
        if (length(rows)) {
            pv <- stat[Variable == v, P]
            tb[rows[1L], `:=`(`P value`, if (is.na(pv)) 
                "Not estimable"
            else if (pv < 0.001) 
                "<0.001"
            else sprintf("%.3f", pv))]
        }
    }
    stopifnot(identical(as.data.frame(tb[, names(tab$data), with = FALSE]), as.data.frame(tab$data)))
    list(data = tb, statistics = stat)
}
ad <- make(saved$admission, cohort$admissions)
pt <- make(saved$patient, cohort$patients)
stopifnot(nrow(ad$data) == 74L, ncol(ad$data) == 7L, nrow(pt$data) == 73L)
fwrite(ad$data, file.path(out, "Table1_training_evaluation_SMD_P.csv"), bom = TRUE)
fwrite(pt$data, file.path(out, "TableS_patient_split_SMD_P.csv"), bom = TRUE)
fwrite(ad$statistics, file.path(out, "Table1_cluster_tests_source.csv"))
saveRDS(list(admission = ad, patient = pt, seed = 123L, note = "Observed-data summaries and absolute SMDs unchanged. Continuous P: survey-adjusted Wilcoxon. Categorical P: overall Rao-Scott F. Unit weights with patient clusters; not survey population weighting. Exploratory unadjusted multiple balance checks, not eligibility/variable-selection criteria."), 
    file.path(out, "Table1_SMD_P.rds"))
cat("TABLE1_SMD_CLUSTER_P_COMPLETE; ORIGINAL_ALL_CELLS_PRESERVED\n")
print(ad$statistics[, .(Variable, P, Warning)])
