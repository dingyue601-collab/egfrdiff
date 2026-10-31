source("shared/R/config.R")
source("shared/R/labels.R")
options(encoding = "UTF-8", stringsAsFactors = FALSE)
suppressPackageStartupMessages({
    library(data.table)
    library(jsonlite)
})
setDTthreads(1L)
set.seed(123)
root <- Sys.getenv("EGFR_PROJECT_ROOT", normalizePath(getwd(), winslash = "/"))
base <- file.path(root, "results")
out <- file.path(base, "publication")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
a <- readRDS(file.path(base, "metrics/Release_metrics.rds"))
u <- readRDS(file.path(base, "bootstrap/Publication_B5000_results.rds"))
extra <- readRDS(file.path(out, "Boundary_equal_weight_sensitivity.rds"))
w <- readRDS(file.path(base, "baseline_split.rds"))
cohort <- readRDS(file.path(base, "patient_split.rds"))
source_data <- readRDS(file.path(out, "Descriptive_rechecked_sources.rds"))
admission_ldl <- readRDS(file.path(out, "Admission_LDL_source.rds"))
cohort$admissions$ldl_c <- admission_ldl$LDL_C[match(cohort$admissions$EncounterID, admission_ldl$EncounterID)]
for (part in c("admissions", "patients")) {
    d <- as.data.table(copy(cohort[[part]]))
    d[, `:=`(history_hypertension, source_data$admission$Hypertension[match(EncounterID, source_data$admission$EncounterID)])]
    cohort[[part]] <- d
}
md <- readRDS(file.path(base, "model_diagnostics/Missingness_diagnostics.rds"))
regression <- as.data.table(readRDS(file.path(base, "prognosis_models_table.rds"))$models)
balance_tests <- readRDS(file.path(base, "metrics/Table1_SMD_P.rds"))
description <- readRDS(file.path(base, "descriptive.rds"))
stopifnot(u$bootstrap == 5000L, nrow(cohort$admissions) == 9890L, nrow(cohort$patients) == 2937L)
fmtp <- function(p) ifelse(p < 0.001, "<0.001", sprintf("%.3f", p))
ci <- function(e, l, h, digits = 3L, percent = FALSE) {
    k <- if (percent) 
        100
    else 1
    sprintf(paste0("%.", digits, "f (%.", digits, "f–%.", digits, "f)"), e * k, l * k, h * k)
}
metric <- function(m, k, rule = "Fixed0p5", percent = FALSE) {
    z <- u$intervals[Algorithm == m & Metric == k & Rule == rule]
    stopifnot(nrow(z) == 1L)
    ci(z$Estimate, z$Lower95, z$Upper95, if (percent) 
        1L
    else 3L, percent)
}
number <- function(x) format(x, big.mark = ",", trim = TRUE, scientific = FALSE)
variables<-c("age","sex","history_hypertension","history_coronary_disease","reported_weight_loss","PriorKRT","cci",
  "sbp","dbp","map","eGFRdiff","negative30","egfrcr","egfrcys","scr","cys","proteinuria","hba1c",
  "hemoglobin","rbc","wbc","platelets","neutrophils_abs","lymphocytes_abs","monocytes_abs","nlr",
  "potassium","sodium","bicarbonate","glucose","uric_acid","calcium","phosphate","albumin","alt","ast",
  "bilirubin","alp","ldh","ada","triglycerides","total_cholesterol","hdl_c","med_acei_arb_recorded",
  "med_sglt2i_recorded","med_insulin_recorded","med_metformin_recorded","med_statins_recorded","med_glp1ra_recorded")
precision<-c(cys=2L,total_cholesterol=2L,hdl_c=2L,triglycerides=2L,calcium=2L,phosphate=2L,rbc=2L,neutrophils_abs=2L,lymphocytes_abs=2L,monocytes_abs=2L,cci=0L)
baseline <- function(unit) {
    raw <- as.data.table(cohort[[if (unit == "admission") 
        "admissions"
    else "patients"]])
    state <- if (unit == "admission") 
        w$admission
    else description$supplementary
    tab <- copy(if (unit == "admission") 
        balance_tests$admission$data
    else state$data)
    meta <- as.data.table(copy(state$metadata))
    spec <- as.data.table(copy(w$admission$spec))
    if (unit == "patient") {
        lab <- which(meta$Section & tab$Characteristic == "Biochemical measurements")
        hba <- which(meta$Variable == "hba1c")
        ord <- c(seq_len(hba - 1L), lab, setdiff(seq.int(hba, nrow(tab)), lab))
        tab <- tab[ord]
        meta <- meta[ord]
        sex <- which(meta$Variable == "sex")
        stopifnot(length(sex) == 1L)
        rows <- rep(seq_len(nrow(tab)), ifelse(seq_len(nrow(tab)) == sex, 3L, 1L))
        tab <- tab[rows]
        meta <- meta[rows]
        pos <- which(meta$Variable == "sex")
        tab[pos, `:=`(Characteristic, c("Sex, No. (%)", "    Female", "    Male"))]
        for (j in 2:5) set(tab, pos, j, "")
        tab[pos[-1L], `:=`(`P value`, "")]
    }
    for (i in seq_len(nrow(tab))) {
        v <- meta$Variable[i]
        if (!meta$Section[i] && nzchar(v) && !startsWith(tab$Characteristic[i], "    ")) {
            lab <- spec[Variable == v, Label]
            if (length(lab) == 1L) 
                tab[i, `:=`(Characteristic, lab)]
        }
    }
    tab[meta$Variable == "history_hypertension", `:=`(Characteristic, "Hypertension, No. (%)")]
    tab[Characteristic == "Kidney function and hematology", `:=`(Characteristic, "Kidney function and proteinuria")]
    tab[Characteristic == "Biochemical measurements", `:=`(Characteristic, "Laboratory values")]
    tab[Characteristic %in% c("Recorded medication use", "Recorded medications, No. (%)"), `:=`(Characteristic, 
        "Medications, No. (%)")]
    plev <- which(meta$Variable == "proteinuria")[-1L]
    tab[plev, `:=`(Characteristic, paste0("    ", c("Grade 1: negative/trace or <0.3 g/L", "Grade 2: 1+ or 0.3–0.99 g/L", 
        "Grade 3: ≥2+ or ≥1.0 g/L")))]
    keep <- meta$Variable %in% if (unit == "patient") 
        setdiff(variables, "negative30")
    else variables
    sec <- which(meta$Section)
    for (j in seq_along(sec)) {
        end <- if (j < length(sec)) 
            sec[j + 1L] - 1L
        else nrow(meta)
        keep[sec[j]] <- any(keep[seq.int(sec[j], end)])
    }
    if (unit == "admission") 
        keep <- keep | tab$Characteristic == "Distinct patients, No."
    meta <- meta[keep]
    z <- copy(tab[keep])
    z[, `:=`(`Missing, No. (%)`, NULL)]
    setnames(z, sub("(n=", "(n = ", gsub(" admissions", "", names(z), fixed = TRUE), fixed = TRUE))
    stopifnot(nrow(z) == nrow(meta))
    sets <- if (unit == "admission") 
        list(seq_len(nrow(raw)), which(raw$DevelopmentPartition == "Train70"), which(raw$DevelopmentPartition == 
            "Evaluation30"))
    else list(seq_len(nrow(raw)), which(raw$eGFRdiff <= -30), which(raw$eGFRdiff > -30 & raw$eGFRdiff < 30), which(raw$eGFRdiff >= 
        30))
    for (i in seq_len(nrow(z))) {
        v <- meta$Variable[i]
        if (is.na(v) || !nzchar(v) || meta$Section[i]) 
            next
        type <- spec[Variable == v, Type]
        if (!length(type)) 
            next
        for (j in seq_along(sets)) {
            xx <- raw[[v]][sets[[j]]]
            xx <- xx[!is.na(xx)]
            if (!length(xx)) {
                set(z, i, j + 1L, "")
                next
            }
            label <- trimws(z$Characteristic[i])
            if (v == "sex" && label == "Sex, No. (%)") 
                next
            if (v == "proteinuria" && label == "Proteinuria, No. (%)") 
                next
            value <- if (type %in% c("mean", "median")) {
                d <- if (v %in% names(precision)) 
                  precision[[v]]
                else 1L
                vals <- if (type == "mean") 
                  c(mean(xx), sd(xx))
                else quantile(xx, c(0.5, 0.25, 0.75), names = FALSE, type = 2)
                form <- paste0("%.", d, "f")
                if (type == "mean") 
                  sprintf(paste0(form, " (", form, ")"), vals[1], vals[2])
                else sprintf(paste0(form, " (", form, "–", form, ")"), vals[1], vals[2], vals[3])
            }
            else {
                target <- if (v == "sex") 
                  if (label == "Female") 
                    0
                  else 1
                else if (v == "proteinuria") 
                  as.integer(sub("^Grade ([123]).*", "\\1", label))
                else 1
                n <- sum(as.character(xx) == as.character(target))
                paste0(number(n), sprintf(" (%.1f%%)", 100 * n/length(xx)))
            }
            set(z, i, j + 1L, value)
        }
    }
    z
}
table1 <- baseline("admission")
tableS1 <- baseline("patient")
# LDL-C was added to the descriptive table, not to model development.
ld <- as.data.table(cohort$admissions)
ld[, partition := factor(DevelopmentPartition, levels=c("Train70","Evaluation30"))]
ld_sets <- list(seq_len(nrow(ld)),which(ld$partition=="Train70"),which(ld$partition=="Evaluation30"))
ld_summary <- vapply(ld_sets,function(i) {
    q <- quantile(ld$ldl_c[i],c(.5,.25,.75),na.rm=TRUE,names=FALSE,type=2)
    sprintf("%.2f (%.2f–%.2f)",q[1],q[2],q[3])
}, "")
suppressPackageStartupMessages({library(survey);library(tableone)})
ld_design <- svydesign(ids=~PatientID,weights=~1,data=ld)
ld_p <- as.numeric(svyranktest(ldl_c~partition,ld_design,test="wilcoxon")$p.value)
ld_smd <- as.numeric(ExtractSmd(CreateTableOne(vars="ldl_c",strata="partition",data=as.data.frame(ld),test=FALSE,smd=TRUE))[1,1])
ld_row <- as.data.table(as.list(setNames(c("LDL-C, median (IQR), mmol/L",ld_summary,sprintf("%.3f",ld_smd),fmtp(ld_p)),names(table1))))
table1 <- rbindlist(list(table1,ld_row))
suppressPackageStartupMessages(library(survey))
htn <- cohort$admissions[, .(Cluster = PatientID, Partition = factor(DevelopmentPartition), Value = history_hypertension)]
design <- svydesign(ids = ~Cluster, weights = ~1, data = htn)
hp <- as.numeric(svychisq(~Value + Partition, design, statistic = "F")$p.value)
pr <- htn[, .(p = mean(Value, na.rm = TRUE)), by = Partition]$p
hsmd <- abs(diff(pr))/sqrt(mean(pr * (1 - pr)))
table1[Characteristic == "Hypertension, No. (%)", `:=`(SMD = sprintf("%.3f", hsmd), `P value` = fmtp(hp))]
b <- source_data$patient[match(cohort$patients$row_index, source_data$patient$row_index)]
grp <- factor(ifelse(b$eGFRdiff <= -30, "Negative", ifelse(b$eGFRdiff >= 30, "Positive", "Reference")), levels = c("Negative", 
    "Reference", "Positive"))
hp <- chisq.test(table(grp, b$hypertension_rechecked))$p.value
tableS1[Characteristic == "Hypertension, No. (%)", `:=`(`P value`, fmtp(hp))]
add_specs <- data.table(Variable = c("ldl_c", "smoking_rechecked", "PriorCCE"), Label = c("LDL-C, median (IQR), mmol/L", 
    "Smoking, No. (%)", "Prior cardiovascular events, No. (%)"), Type = c("median", "binary", "binary"))
addrows <- rbindlist(lapply(seq_len(nrow(add_specs)), function(i) {
    sp <- add_specs[i]
    x <- b[[sp$Variable]]
    sets <- c(list(seq_along(x)), lapply(levels(grp), function(g) which(grp == g)))
    value <- vapply(sets, function(ix) {
        z <- x[ix]
        z <- z[!is.na(z)]
        if (sp$Type == "median") {
            v <- quantile(z, c(0.5, 0.25, 0.75), names = FALSE, type = 2)
            sprintf("%.2f (%.2f–%.2f)", v[1], v[2], v[3])
        }
        else paste0(number(sum(z == 1)), sprintf(" (%.1f%%)", 100 * mean(z == 1)))
    }, "")
    pv <- if (sp$Type == "median") 
        kruskal.test(x ~ grp)$p.value
    else chisq.test(table(grp, x))$p.value
    as.data.table(as.list(setNames(c(sp$Label, value, fmtp(pv)), names(tableS1))))
}))
at <- which(tableS1$Characteristic == "Medications, No. (%)")[1L]
tableS1 <- rbindlist(list(tableS1[seq_len(at - 1L)], addrows, tableS1[seq.int(at, nrow(tableS1))]))
for (nm in c("table1", "tableS1")) {
    z <- get(nm)
    setnames(z, gsub("Evaluation", "Validation", names(z), fixed = TRUE))
    assign(nm, z)
}
group_baseline_rows <- function(tab) {
    sections <- list(
        "Demographic and clinical characteristics" = c("Age,", "Sex,", "Female", "Male", "Smoking,", "Hypertension,", "History of coronary", "Prior cardiovascular", "Reported weight", "Prior KRT", "Charlson"),
        "Clinical measurements" = c("SBP,", "DBP,", "MAP,"),
        "Kidney function and proteinuria" = c("eGFRdiff,", "Negative eGFRdiff", "eGFRcr,", "eGFRcys,", "Serum creatinine,", "Serum cystatin", "Proteinuria,", "Grade 1:", "Grade 2:", "Grade 3:", "Serum uric acid,"),
        "Glycemic measurements" = c("HbA1c,", "Glucose,"),
        "Hematology" = c("Hemoglobin,", "Red blood cell", "White blood cell", "Neutrophil count,", "Lymphocyte count,", "Monocytes,", "Platelet count,", "NLR,"),
        "Electrolytes and mineral metabolism" = c("Potassium,", "Sodium,", "Bicarbonate,", "Total calcium,", "Phosphate,"),
        "Liver function and serum enzymes" = c("Serum albumin,", "ALT,", "AST,", "Total bilirubin,", "ALP,", "LDH,", "ADA,"),
        "Lipid profile" = c("Triglycerides,", "Total cholesterol,", "HDL-C,", "LDL-C,"),
        "Medications, No. (%)" = character(),
        "    RAS inhibitors" = "ACEI/ARB",
        "    Glucose-lowering medications" = c("Metformin", "Insulin", "SGLT2 inhibitors", "GLP-1 receptor agonists"),
        "    Lipid-lowering medications" = "Statins")
    tab <- copy(tab)
    body <- tab[apply(as.matrix(tab[, -1, with = FALSE]), 1, function(x) any(!is.na(x) & x != ""))]
    labels <- trimws(body$Characteristic)
    rows <- list(body[startsWith(labels, "Distinct patients")])
    used <- startsWith(labels, "Distinct patients")
    for (heading in names(sections)) {
        row <- as.data.table(as.list(setNames(c(heading, rep("", ncol(tab) - 1L)), names(tab))))
        rows[[length(rows) + 1L]] <- row
        for (prefix in sections[[heading]]) {
            ix <- which(!used & startsWith(labels, prefix))
            if (length(ix)) rows[[length(rows) + 1L]] <- body[ix]
            used[ix] <- TRUE
        }
    }
    stopifnot(all(used))
    rbindlist(rows)
}
table1 <- group_baseline_rows(table1)
tableS1 <- group_baseline_rows(tableS1)
modelnames <- c(Logistic = "Logistic regression", XGBoost = "XGBoost", RF = "Random forest", LightGBM = "LightGBM", 
    SVM = "SVM", TabPFN = "TabPFN", TabICL = "TabICL", TabM = "TabM (21 variables)", TabM9 = "TabM (9 variables)")
order_models <- a$table2[order(-AUC), Algorithm]
table2 <- rbindlist(lapply(order_models, function(m) data.table(Model = unname(modelnames[m]), `AUC (95% CI)` = metric(m, 
    "AUC"), `AP (95% CI)` = metric(m, "AP"), `Brier score (95% CI)` = metric(m, "Brier"), `Accuracy, % (95% CI)` = metric(m, 
    "Accuracy", percent = TRUE), `Sensitivity, % (95% CI)` = metric(m, "Sensitivity", percent = TRUE), `Specificity, % (95% CI)` = metric(m, 
    "Specificity", percent = TRUE), `PPV, % (95% CI)` = metric(m, "PPV", percent = TRUE), `NPV, % (95% CI)` = metric(m, 
    "NPV", percent = TRUE), `F1 (95% CI)` = metric(m, "F1"))))
outcome_names <- c(KF = "Kidney failure", CCE = "Composite cardiovascular events", Mortality = "All-cause mortality")
groups <- c(Negative = "≤−30%", Reference = ">−30% to <30%", Positive = "≥30%")
adjustments <- c("Unadjusted", "Demographic adjusted", "Fully adjusted")
tableS2 <- rbindlist(lapply(names(outcome_names), function(oc) rbindlist(lapply(names(groups), function(g) {
    z <- data.table(Outcome = outcome_names[[oc]], eGFRdiff = groups[[g]])
    for (j in seq_along(adjustments)) {
        one <- regression[Outcome == oc & Model == adjustments[j] & Contrast == g]
        if (g == "Reference") {
            hr <- "Ref"
            p <- ""
        }
        else {
            stopifnot(nrow(one) == 1L)
            hr <- ci(one$Estimate, one$Lower95, one$Upper95, 2L)
            p <- fmtp(one$p_value)
        }
        z[, `:=`((paste0("HR", j)), hr)]
        z[, `:=`((paste0("P", j)), p)]
    }
    z
}))))
regression_headers <- list(c("Outcome", "eGFRdiff", "Unadjusted", "", "Demographic adjusted", "", "Fully adjusted", ""), 
    c("", "", "sHR / HR (95% CI)", "P value", "sHR / HR (95% CI)", "P value", "sHR / HR (95% CI)", "P value"))
missing <- copy(md$missingness[(is.finite(BaselineMissingPct) & BaselineMissingPct > 0) | (is.finite(AdmissionMissingPct) & 
    AdmissionMissingPct > 0)])
labels <- setNames(as.data.table(w$admission$spec)$Label, as.data.table(w$admission$spec)$Variable)
labels[c("ldl_c", "smoking_rechecked")] <- c("LDL-C, median (IQR), mmol/L", "Smoking, No. (%)")
label <- function(v, fallback) {
    x <- unname(labels[v])
    ifelse(is.na(x), fallback, x)
}
method <- function(x) {
    x <- gsub("Random-forest MICE", "RF", x, fixed = TRUE)
    x <- gsub("Passive component ratio", "Derived from component counts", x, fixed = TRUE)
    x <- gsub(" logistic MICE", " logistic regression", x, fixed = TRUE)
    ifelse(grepl("Not included|No imputation required", x), "—", x)
}
tableS3 <- missing[, .(Characteristic = label(Variable, Label), `Patient-level missing, %` = BaselineMissingPct, 
    PatientMethod = method(BaselineMethod), `Admission-level missing, %` = AdmissionMissingPct, AdmissionMethod = method(AdmissionMethod))]
tableS3[grepl("Hypertension", Characteristic, ignore.case = TRUE), `:=`(Characteristic, "Hypertension, No. (%)")]
tableS3[startsWith(Characteristic,"LDL-C,"), `:=`(`Admission-level missing, %`=100*mean(is.na(ld$ldl_c)),AdmissionMethod="—")]
# Descriptive red cell missingness
rbc_missing <- c(mean(is.na(cohort$patients$rbc)),mean(is.na(cohort$admissions$rbc)))*100
rbc_label <- as.data.table(w$admission$spec)[Variable=="rbc",Label]
rbc_row <- data.table(Characteristic=rbc_label,P=rbc_missing[1],PM="—",A=rbc_missing[2],AM="—")
setnames(rbc_row,names(tableS3)); tableS3 <- rbindlist(list(tableS3,rbc_row))
map_row <- data.table(Characteristic="MAP, mean (SD), mmHg",
    P=mean(is.na(cohort$patients$map))*100, PM="—",
    A=mean(is.na(cohort$admissions$map))*100, AM="—")
setnames(map_row,names(tableS3))
tableS3 <- rbindlist(list(tableS3[1:2],map_row,tableS3[-(1:2)]))
missing_headers <- list(c("Characteristic", "Patient-level missing, %", "Imputation method", "Admission-level missing, %", 
    "Imputation method"))
tableS4 <- rbindlist(lapply(combn(setdiff(order_models, "TabM9"), 2L, simplify = FALSE), function(pair) {
    z <- u$paired_full21[ModelA == pair[1] & ModelB == pair[2]]
    if (!nrow(z)) {
        z <- copy(u$paired_full21[ModelA == pair[2] & ModelB == pair[1]])
        z[, `:=`(DeltaAUC = -DeltaAUC, Lower95 = -Upper95, Upper95 = -Lower95)]
    }
    stopifnot(nrow(z) == 1L)
    data.table(Comparison = paste(unname(modelnames[pair]), collapse = " − "), `Difference in AUC (95% CI)` = ci(z$DeltaAUC, 
        z$Lower95, z$Upper95), `P value` = fmtp(z$P), `Holm-adjusted P value` = fmtp(z$P_Holm))
}))
fullshap <- fread(file.path(base, "shap_full/TabM_OFFICIAL_v2_importance.csv"))[order(-MeanAbsSHAP)]
shap_labels <- setNames(as.data.table(w$admission$spec)$Label, as.data.table(w$admission$spec)$Variable)
candidates <- readRDS(file.path(base, "ml_imputation_inputs.rds"))$features
selected <- readRDS(file.path(base, "ml_imputation_selection.rds"))$selected18
all_candidates <- c(fullshap$Feature, setdiff(candidates, fullshap$Feature))
tableS5 <- data.table(Characteristic = unname(shap_labels[all_candidates]), `Selected by LASSO` = ifelse(all_candidates %in% 
    selected, "Yes", "No"), `Mean absolute SHAP value` = ifelse(all_candidates %in% fullshap$Feature, sprintf("%.3f", 
    fullshap$MeanAbsSHAP[match(all_candidates, fullshap$Feature)]), "—"))
tableS5[grepl("Hypertension", Characteristic, ignore.case = TRUE), `:=`(Characteristic, "Hypertension, No. (%)")]
stopifnot(nrow(tableS5) == 34L, sum(tableS5[["Selected by LASSO"]] == "Yes") == 21L)
counts <- function(z) paste(vapply(c("TP", "FP", "FN", "TN"), function(k) number(z[Metric == k, Estimate]), ""), 
    collapse = " / ")
tableS6 <- rbindlist(lapply(c(60L, 70L, 80L, 90L), function(target) {
    rule <- paste0("Recall", target)
    z <- u$intervals[Algorithm == "TabM9" & Rule == rule]
    data.table(`Target sensitivity, %` = target, `Probability threshold` = unique(z$Threshold), `Accuracy, % (95% CI)` = metric("TabM9", 
        "Accuracy", rule, TRUE), `Sensitivity, % (95% CI)` = metric("TabM9", "Sensitivity", rule, TRUE), `Specificity, % (95% CI)` = metric("TabM9", 
        "Specificity", rule, TRUE), `PPV, % (95% CI)` = metric("TabM9", "PPV", rule, TRUE), `NPV, % (95% CI)` = metric("TabM9", 
        "NPV", rule, TRUE), `F1 (95% CI)` = metric("TabM9", "F1", rule), `Testing proportion, % (95% CI)` = metric("TabM9", "TestingFraction", rule, TRUE), `TP / FP / FN / TN` = counts(z))
}))
analyses <- c("Patient_equal_weights", "Excluding_boundary_plusminus5", "Boundary5_patient_equal", "Complete_nine_inputs")
analysis_names <- c("Equal patient weights", "Excluding -35% to -25%", "Excluding -35% to -25%; equal patient weights", 
    "Complete cases for nine inputs")
robust <- rbindlist(list(u$robustness, extra$new_results), use.names = TRUE, fill = TRUE)
tableS7 <- rbindlist(lapply(seq_along(analyses), function(i) {
    z <- robust[Analysis == analyses[i]]
    get <- function(k, percent = FALSE) {
        v <- z[Metric == k]
        stopifnot(nrow(v) == 1L)
        ci(v$Estimate, v$Lower95, v$Upper95, if (percent) 
            1L
        else 3L, percent)
    }
    data.table(Analysis = analysis_names[i], `n / patients` = paste(number(unique(z$Admissions)), number(unique(z$Patients)), 
        sep = " / "), `AUC (95% CI)` = get("AUC"), `AP (95% CI)` = get("AP"), `Brier score (95% CI)` = get("Brier"), 
        `Accuracy, % (95% CI)` = get("Accuracy", TRUE), `Sensitivity, % (95% CI)` = get("Sensitivity", TRUE), `Specificity, % (95% CI)` = get("Specificity", 
            TRUE), `PPV, % (95% CI)` = get("PPV", TRUE), `NPV, % (95% CI)` = get("NPV", TRUE), `F1 (95% CI)` = get("F1"), `Testing proportion, % (95% CI)` = get("TestingFraction", TRUE))
}))
for (nm in c("table1", "tableS1", "tableS3", "tableS5")) {
    z <- get(nm)
    z$Characteristic <- publication_labels(z$Characteristic)
    if (nm == "tableS3") for (column in c("PatientMethod", "AdmissionMethod")) z[[column]] <- gsub("RF", "Random forest", z[[column]], fixed = TRUE)
    assign(nm, z)
}
main <- list(Table1_Baseline = table1, Table2_Model_performance = table2)
supp <- list(S1_Patient_baseline = tableS1, S2_Prognostic_models = tableS2, S3_Missing_data = tableS3, S4_Pairwise_AUC = tableS4, 
    S5_Feature_contributions = tableS5, S6_Recall_targets = tableS6, S7_Sensitivity_analyses = tableS7)

saveRDS(list(main = main, supplementary = supp), file.path(out, "Numerical_summary_tables.rds"))
