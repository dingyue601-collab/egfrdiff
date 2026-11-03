publication_labels <- function(x) {
  x <- gsub(" (questionnaire)", "", x, fixed = TRUE)
  terms <- c(Creatinine = "Serum creatinine", Albumin = "Serum albumin", "Uric acid" = "Serum uric acid",
    "Adenosine deaminase" = "ADA", "Alkaline phosphatase" = "ALP", "Reported weight loss" = "Weight loss",
    "Coronary disease history" = "History of coronary disease", "Coronary disease" = "History of coronary disease",
    "Reported hypertension history" = "Hypertension", "Reported coronary disease history" = "History of coronary disease",
    Lymphocytes = "Lymphocyte count", Monocytes = "Monocyte count", Neutrophils = "Neutrophil count",
    "HDL cholesterol" = "HDL-C", Calcium = "Total calcium", "Proteinuria category" = "Proteinuria",
    "Systolic BP" = "SBP", "Diastolic BP" = "DBP", "White blood cells" = "White blood cell count", Platelets = "Platelet count")
  for (old in names(terms)) x <- sub(paste0("^", old, "(?= \\(|,|$)"), terms[[old]], x, perl = TRUE)
  x
}
