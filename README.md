# Negative eGFR discordance in T2DKD: analysis code and TabM calculator

This project examines associations between negative eGFR discordance and clinical outcomes in patients with type 2 diabetic kidney disease, and uses routine clinical measurements to identify admissions that may warrant additional cystatin C testing.

- Article: Negative eGFR discordance in type 2 diabetic kidney disease: prognostic associations and an interpretable model for targeted cystatin C testing
- Code repository: https://github.com/dingyue601-collab/egfrdiff
- English application: https://egfrdiffcaleng.streamlit.app/
- Chinese application: https://egfrdiffcalch.streamlit.app/

## Structure

```text
prognosis/R/              Prognostic analysis: imputation, regression and pooling
machine_learning/R/       Machine learning: imputation, selection, calibration and evaluation
machine_learning/python/  Tabular model training, feature reduction and calibrated SHAP
calculator/               English and Chinese Streamlit interfaces, inference and nine-variable models
shared/R/                 Data loading and shared statistical functions
data/reference/           Aggregate results used in the article
docs/                     Dependencies and selected model parameters
```

The public code does not include hospital database extraction, identity linkage or raw patient records.

## Environment

The analyses used R 4.5.2 and Python 3.12.14. R package versions, Python dependencies and model parameters are listed in `docs/R_packages.csv`, `docs/requirements.txt` and `docs/published_parameters.json`, respectively. Calculator dependencies are listed separately in `calculator/requirements.txt`. The Tab model analysis backends use CUDA and float32.

Analysis paths are relative to the repository root. `EGFR_PYTHON` specifies the Python interpreter for the model backends. `EGFR_INPUT_DIR` and `EGFR_REFERENCE_DIR` specify the prepared-data and aggregate-input directories, which default to `data/prepared/` and `data/reference/`.

## Core analysis code

### Prognostic analysis

`prognosis/R/02_prognosis.R` handles a separate set of 20 random-forest MICE imputations, Fine–Gray regression for kidney failure and composite cardiovascular events, Cox regression for all-cause mortality, and Rubin pooling. `04_prognosis_table.R` assembles the reported estimates.

### Machine learning

| Code | Purpose |
|---|---|
| `R/01_impute_and_select.R` | Separate training and validation imputation with 30 datasets each; ten-fold LASSO 1SE selection, retaining variables selected in ≥18/30 datasets |
| `R/03_model_comparison.R`, `R/refit_classic.R` and Python model backends | Eight-algorithm comparison, parameter settings and model fitting |
| `R/04_select_model.R` | Algorithm selection by mean patient-grouped training cross-validation AUC |
| `R/06_feature_reduction.R`, `python/feature_reduction.py` | Training-fold permutation importance and sequential feature reduction |
| `R/07_full_model_explanation.R`, `python/calibrated_shap.py` | Post-calibration SHAP for the full and reduced TabM models |
| Scripts beginning with `R/08` through `R/15` | Performance evaluation, missing-data diagnostics, probability calibration, recall thresholds, patient-cluster bootstrap and sensitivity analyses |
| `R/29_clinical_reference.R`, `R/31_three_predictor_statistics.R` and the Python three-predictor backend | Reference models using age, sex and serum creatinine |
| `R/16_tables.R` | Numerical table summaries |

These paths are relative to `machine_learning/`. `shared/R/` contains data-loading and statistical functions; `docs/` records dependencies and model parameters. `data/reference/` contains only aggregate prognostic effects, descriptive statistics and missingness summaries, not patient-level data.

Training and validation are split 70/30 by patient, so admissions from the same patient do not cross sets. LASSO uses ten-fold cross-validation; model development uses patient-level five-fold cross-validation. Mean training-fold AUC selects the algorithm, with Brier score breaking exact ties. These development scores are conditional on the training-selected LASSO list. TabPFN and TabICL use fixed pretrained configurations, with checkpoint paths under `models/pretrained/`. Calibration and testing thresholds are determined from training out-of-fold predictions. SHAP explains the Platt-calibrated mean positive probability. Fitted inference files for the nine-variable TabM model are in `calculator/models/`.

After fitting, run selection (`04`), training-fold reduction (`06`), calibration (`10`) and recall thresholds (`11`) before the terminal validation scripts (`05`, `08`, `12`–`15`). The development steps do not load validation labels. Reported performance represents post-selection internal evaluation and requires confirmation in independent data.

Probability evaluation (`08`) does not require SHAP outputs. For explanations, run `R/07_full_model_explanation.R prepare`, then `python/calibrated_shap.py --family both` after calibration and the final model fits. Table summaries (`16`) read the resulting `results/shap_full/TabM_OFFICIAL_v2_importance.csv` directly. Reference-model evaluation (`29`, `31`) follows `12` and reads training and validation labels from `results/metrics/Release_metrics.rds`; run `python/three_predictor_reference.py` before `31` to generate the three-predictor TabM probabilities.

## Calculator

The calculator includes 30 fitted nine-variable TabM models and their preprocessors, without training patient records. The inputs and units are age (years), sex (female = 0, male = 1), serum creatinine (μmol/L), HbA1c (%), hemoglobin (g/L), serum uric acid (μmol/L), serum albumin (g/L), alkaline phosphatase (U/L) and adenosine deaminase (U/L).

In a Python 3.12 environment with the dependencies in `calculator/requirements.txt`, run:

```sh
cd calculator
python -m streamlit run app.py
```

For the Chinese interface:

```sh
python -m streamlit run app_zh.py
```

The application supports individual entry, CSV template download, batch import and editing, record selection, and result export. `Patient` is the unique identifier for each record; repeated admissions require different record identifiers. The Chinese interface uses Chinese CSV headers and also accepts the English template. Individual and batch records use the same inference module. The model returns the Platt-calibrated probability of eGFRdiff <−30%; the screening threshold is 16.37%.

For Streamlit Community Cloud, select Python 3.12 and use `calculator/app.py` or `calculator/app_zh.py` as the entry point. Retain the calculator's `models/`, `vendor/`, `assets/` and `LICENSES/` directories.

## License

Project code is licensed under Apache-2.0. The bundled TabM and numerical-embedding modules retain their upstream Apache-2.0 licenses and notices; other dependencies remain subject to their own licenses. Retain `calculator/THIRD_PARTY_NOTICES.txt` and `calculator/LICENSES/` when using or redistributing the software.
