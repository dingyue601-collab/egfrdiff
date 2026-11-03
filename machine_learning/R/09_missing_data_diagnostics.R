source("shared/R/config.R")
options(encoding = "UTF-8", stringsAsFactors = FALSE)
suppressPackageStartupMessages({
    library(data.table)
    library(mice)
})
setDTthreads(1L)
set.seed(123)
base <- file.path(getwd(), "results")
out <- file.path(base, "model_diagnostics")
p <- readRDS(file.path(base, "ml_imputation_inputs.rds"))
b <- readRDS(file.path(base, "prognosis_inputs.rds"))
bm <- readRDS(file.path(base, "prognosis_MICE20.rds"))
tm <- readRDS(file.path(base, "ml_imputation_Train_mids.rds"))
em <- readRDS(file.path(base, "ml_imputation_Evaluation_mids.rds"))
stopifnot(nrow(b$source_data) == 2937L, tm$m == 30L, em$m == 30L, bm$mids$m == 20L)
selected <- c("age", "sex", "scr", "hba1c", "hemoglobin", "uric_acid", "albumin", "alp", "ada")
label <- c(age = "Age (years)", sex = "Sex", scr = "Serum creatinine (umol/L)", sbp = "Systolic BP (mmHg)", dbp = "Diastolic BP (mmHg)", 
    pulse = "Pulse (beats/min)", hba1c = "HbA1c (%)", hemoglobin = "Hemoglobin (g/L)", wbc = "White blood cells (10^9/L)", 
    platelets = "Platelets (10^9/L)", nlr = "Neutrophil-to-lymphocyte ratio", potassium = "Potassium (mmol/L)", 
    sodium = "Sodium (mmol/L)", bicarbonate = "Bicarbonate (mmol/L)", glucose = "Glucose (mmol/L)", uric_acid = "Uric acid (umol/L)", 
    calcium = "Calcium (mmol/L)", phosphate = "Phosphate (mmol/L)", alt = "ALT (U/L)", ast = "AST (U/L)", albumin = "Albumin (g/L)", 
    bilirubin = "Total bilirubin (umol/L)", alp = "Alkaline phosphatase (U/L)", ldh = "LDH (U/L)", triglycerides = "Triglycerides (mmol/L)", 
    total_cholesterol = "Total cholesterol (mmol/L)", hdl_c = "HDL cholesterol (mmol/L)", proteinuria = "Proteinuria category", 
    lymphocytes_abs = "Lymphocytes (10^9/L)", monocytes_abs = "Monocytes (10^9/L)", neutrophils_abs = "Neutrophils (auxiliary;10^9/L)", 
    reported_weight_loss = "Reported weight loss", history_hypertension = "Reported hypertension history", history_coronary_disease = "Reported coronary disease history", 
    ada = "Adenosine deaminase (U/L)", egfrcr = "eGFRcr (mL/min/1.73 m2)", cci = "Charlson comorbidity index", 
    ldl_c = "LDL cholesterol (mmol/L)", hypertension_rechecked = "Hypertension status", smoking_rechecked = "Smoking history", 
    PriorCCE = "Prior composite cardiovascular event")
variables <- unique(c(p$imputation_variables, "egfrcr", "cci", "ldl_c", "hypertension_rechecked", "smoking_rechecked", 
    "PriorCCE"))
friendly <- function(v) {
    z <- unname(label[v])
    z[is.na(z)] <- v[is.na(z)]
    z
}
method_name <- function(code, missing, used) {
    if (!used) 
        return("Not included in this imputation model")
    if (missing == 0) 
        return("No imputation required")
    switch(code, rf = "Random-forest MICE", logreg = "Binary logistic MICE", polr = "Ordinal logistic MICE", `~I(neutrophils_abs / lymphocytes_abs)` = "Passive component ratio", 
        paste0("Existing method: ", code))
}
missingness <- rbindlist(lapply(variables, function(v) {
    bv <- if (v %in% names(b$input)) 
        b$input[[v]]
    else b$source_data[[v]]
    ml <- v %in% names(p$train)
    tn <- if (ml) 
        sum(is.na(p$train[[v]]))
    else NA_integer_
    en <- if (ml) 
        sum(is.na(p$evaluation_raw[[v]]))
    else NA_integer_
    bn <- if (!is.null(bv)) 
        sum(is.na(bv))
    else NA_integer_
    data.table(Variable = v, Label = friendly(v), BaselineMissingN = bn, BaselineMissingPct = 100 * bn/2937, AdmissionMissingN = tn + 
        en, AdmissionMissingPct = 100 * (tn + en)/9890, TrainingMissingN = tn, TrainingMissingPct = 100 * tn/6852, 
        ValidationMissingN = en, ValidationMissingPct = 100 * en/3038, BaselineMethod = method_name(if (v %in% 
            names(b$method)) 
            b$method[v]
        else "", bn, v %in% names(b$method)), AdmissionMethod = method_name(if (ml) 
            tm$method[v]
        else "", tn + en, ml), FinalNine = v %in% selected, Role = if (v == "neutrophils_abs") 
            "Imputation auxiliary, not a candidate model input"
        else if (v %in% p$features) 
            "Candidate input"
        else "Prognostic adjustment only")
}))
stopifnot(all(missingness[Variable %in% selected, TrainingMissingPct] <= 30))
fwrite(missingness, file.path(out, "Missingness_and_methods.csv"))
partitions <- list(Baseline = list(raw = b$input, mi = bm$mids, variables = b$imputed_variables), Training = list(raw = p$train, 
    mi = tm, variables = names(tm$method)[tm$method != ""]), Validation = list(raw = p$evaluation_raw, mi = em, 
    variables = names(em$method)[em$method != ""]))
density_data <- list()
categorical <- list()
summaries <- list()
traces <- list()
for (part in names(partitions)) {
    z <- partitions[[part]]
    completed <- lapply(seq_len(z$mi$m), function(k) mice::complete(z$mi, k))
    for (v in z$variables) {
        raw <- z$raw[[v]]
        mask <- is.na(raw)
        if (!any(mask)) 
            next
        if (is.factor(raw)) {
            lev <- levels(raw)
            obs <- as.character(raw[!mask])
            imp <- unlist(lapply(completed, function(d) as.character(d[[v]][mask])))
            for (type in c("Observed", "Imputed")) {
                vv <- if (type == "Observed") 
                  obs
                else imp
                tab <- table(factor(vv, levels = lev))
                categorical[[length(categorical) + 1L]] <- data.table(Partition = part, Variable = v, Label = friendly(v), 
                  Kind = type, Level = lev, Proportion = as.numeric(tab)/sum(tab))
            }
        }
        else {
            obs <- as.numeric(raw[!mask])
            imp <- unlist(lapply(completed, function(d) as.numeric(d[[v]][mask])))
            stopifnot(all(is.finite(obs)), all(is.finite(imp)))
            lim <- range(c(quantile(obs, c(0.005, 0.995)), quantile(imp, c(0.005, 0.995))))
            if (diff(lim) == 0) 
                lim <- lim + c(-0.5, 0.5)
            for (type in c("Observed", "Imputed")) {
                vv <- if (type == "Observed") 
                  obs
                else imp
                dd <- density(vv, n = 256, from = lim[1], to = lim[2])
                density_data[[length(density_data) + 1L]] <- data.table(Partition = part, Variable = v, Label = friendly(v), 
                  Kind = type, Value = dd$x, Density = dd$y)
                summaries[[length(summaries) + 1L]] <- data.table(Partition = part, Variable = v, Kind = type, 
                  N = length(vv), Mean = mean(vv), SD = sd(vv), Min = min(vv), P005 = quantile(vv, 0.005), P25 = quantile(vv, 
                    0.25), Median = median(vv), P75 = quantile(vv, 0.75), P995 = quantile(vv, 0.995), Max = max(vv), 
                  OutsideDisplayN = sum(vv < lim[1] | vv > lim[2]), DisplayMin = lim[1], DisplayMax = lim[2])
            }
        }
    }
    for (stat in c("chainMean", "chainVar")) {
        a <- z$mi[[stat]]
        if (!"hba1c" %in% dimnames(a)[[1]]) 
            next
        x <- a["hba1c", , , drop = TRUE]
        for (k in seq_len(ncol(x))) traces[[length(traces) + 1L]] <- data.table(Partition = part, Statistic = stat, 
            Chain = k, Iteration = seq_len(nrow(x)), Value = x[, k])
    }
    rm(completed)
    invisible(gc())
}
dens <- rbindlist(density_data)
cats <- rbindlist(categorical)
summaries <- rbindlist(summaries)
traces <- rbindlist(traces)
fwrite(summaries, file.path(out, "Observed_imputed_distribution_summary.csv"))
fwrite(cats, file.path(out, "Observed_imputed_categorical_proportions.csv"))
mlresult <- readRDS(file.path(base, "ml_imputation_results.rds"))
chains <- rbindlist(list(as.data.table(bm$diagnostics$convergence)[, `:=`(Partition, "Baseline")], as.data.table(mlresult$tables$Train_MI_chains)[, 
    `:=`(Partition, "Training")], as.data.table(mlresult$tables$Evaluation_MI_chains)[, `:=`(Partition, "Validation")]), 
    fill = TRUE)
saveRDS(list(missingness = missingness, densities = dens, categorical = cats, summary = summaries, traces = traces, 
    chains = chains, labels = label, seed = 123, methods = list(baseline = b$settings, ML = p$settings)), file.path(out, 
    "Missingness_diagnostics.rds"))
writeLines(c("Supplementary methods: missing data", "Prognostic and admission-level recognition analyses used separate imputation procedures and did not share completed data sets.", 
    "For prognostic analyses,20 existing MICE data sets were used. Continuous HbA1c, systolic blood pressure and LDL cholesterol were imputed by random forests; proteinuria category by ordinal logistic regression; smoking by binary logistic regression. Exposure, outcome and time-related auxiliaries were retained only in the prognostic imputation model.", 
    "For recognition modelling, predictors with more than30% training missingness were ineligible. Training and validation data were separately imputed without the recognition label, cystatin C, eGFR-derived variables, identifiers or future outcomes. Each partition used30 MICE data sets,20 iterations and10 ranger trees per RF conditional model. Binary and ordered predictors used logistic and ordinal logistic models; missing NLR was passively derived from its components. Observed values were preserved.", 
    "Patient-grouped development folds used separate existing fold-specific imputations. Final model probabilities were averaged across30 completed data sets. The validation imputation is batch/transductive, not a training-fitted single-person imputer. The reduced nine-input model remains conditional on the original candidate-variable imputation system.", 
    "Diagnostics compared observed values with imputed-only values, separately by analysis and partition; imputed values were pooled across imputations for descriptive displays. Continuous density panels show the central99% support of each observed/imputed distribution; full ranges and tail counts are reported separately. Trace diagnostics do not prove a missing-at-random mechanism."), 
    file.path(out, "Supplementary_missing_data_methods.txt"), useBytes = TRUE)
cat("MISSINGNESS_AND_EXISTING_MI_DIAGNOSTICS_COMPLETE\n")
