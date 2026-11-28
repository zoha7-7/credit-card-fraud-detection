# Sparkov Fraud Detection: Astra Handoff

## 1. Project Objective

Develop a separate fraud detection pipeline for the Sparkov transaction dataset.

The larger project aims to support multiple transaction schemas through a unified application. Do not force different schemas into one model.

Current schema tracks:

1. ULB: anonymized numerical transaction features.
2. Sparkov: merchant and transaction information.
3. IEEE-CIS: customer and behavioral transaction features (future work).

The immediate task is Sparkov model development and evaluation.

## 2. Repository and Existing ULB Work

Repository:
https://github.com/zoha7-7/credit-card-fraud-detection

Relevant branches:
- model-training: existing ULB model pipeline.
- ulb-data-creation: ULB data preparation and current integration work.
- misc: previous audit work.

The selected ULB model is already committed on model-training.

Model:
artifacts/trained_models/lightgbm_smote_ratio_0_03.joblib

Preprocessing:
- artifacts/preprocessing/validation_imputer.joblib
- artifacts/preprocessing/validation_scaler.joblib

Feature definition:
- artifacts/datasets/feature_columns.json

External evaluation configuration:
- external_evaluation/external_config.json

Selected ULB threshold: 0.85232879.

Do not modify or replace the existing ULB pipeline while developing Sparkov.

## 3. Sparkov Dataset Locations

Local files:

- external_data/Sparkov/fraudTrain.csv
- external_data/Sparkov/fraudTest.csv

These raw files are intentionally excluded from Git because they contain sensitive personal and financial information.

Do not remove the ignore rules or upload these raw files.

The files are available locally in the development environment.

## 4. Dataset Structure

Both files contain 23 columns:

Unnamed: 0
trans_date_trans_time
cc_num
merchant
category
amt
first
last
gender
street
city
state
zip
lat
long
city_pop
job
dob
trans_num
unix_time
merch_lat
merch_long
is_fraud

No missing values were found in either dataset during the initial audit.

### fraudTrain.csv

- Rows: 1,296,675
- Legitimate: 1,289,169
- Fraudulent: 7,506
- Fraud prevalence: approximately 0.5789%
- Date range: 2019-01-01 to 2020-06-21

### fraudTest.csv

- Rows: 555,719
- Legitimate: 553,574
- Fraudulent: 2,145
- Fraud prevalence: approximately 0.3860%
- Date range: 2020-06-21 to 2020-12-31

The files form a chronological split. The training file ends at approximately 12:13:37 on 2020-06-21, and the test file begins at approximately 12:14:25.

## 5. Required Data Split Strategy

Use fraudTrain.csv for:

- Training.
- Chronological validation.
- Feature engineering decisions.
- Model comparison.
- Hyperparameter tuning.
- Threshold selection.

Keep fraudTest.csv untouched until the final evaluation.

Do not use the test set for training, oversampling, feature selection, threshold tuning, or model selection.

A small presentation subset may be extracted from the held-out test data after the final evaluation. Clearly distinguish application demonstration from independent model evaluation.

Investigate and document the chronological validation boundary before splitting.

## 6. Timestamp Audit Findings

The full training and test files were audited for chronological ordering.

One backward transaction-date anomaly was identified in fraudTrain.csv:

- Previous row index: 100531
- Previous timestamp: 2019-02-28 23:59:40
- Current row index: 100532
- Current timestamp: 2019-02-28 00:02:34

The corresponding unix_time values increased by 174 seconds.

The test file showed no backward timestamp anomalies.

The unix_time field does not directly correspond to the 2019-2020 transaction timestamps when interpreted as standard Unix seconds. An approximately seven-year offset was observed, with a one-day variation related to leap-year handling.

Use trans_date_trans_time as the authoritative timestamp for chronological splitting and time-derived features.

Do not use unix_time as a predictive feature until its semantics have been independently established.

Document how the single ordering anomaly is handled.

## 7. Feature Engineering and Leakage Prevention

Potential useful features:

- Transaction amount.
- Merchant.
- Transaction category.
- Hour, day, month and day of week.
- Distance between customer and merchant coordinates using a haversine calculation.
- City population.
- Carefully constructed historical transaction frequency and amount features.

Important exclusions and cautions:

- is_fraud is the target.
- Unnamed: 0 is an index, not a predictive feature.
- trans_num is a transaction identifier and should not be used as an ordinary predictive feature.
- cc_num is a sensitive card identifier. Do not use it as an ordinary categorical predictor.
- first, last, street, zip, city, state, dob, job and gender require careful leakage, privacy and generalization assessment.
- Any customer-history features must use past transactions only.
- Never allow future transactions to influence historical features.
- Do not calculate aggregates using the full dataset before splitting.

Document the final feature list and the reasoning behind each inclusion or exclusion.

## 8. Imbalance Experiments

Compare at least:

1. LightGBM baseline without imbalance correction.
2. LightGBM with class weighting.
3. SMOTENC, if appropriate for the final categorical/numerical feature representation.

Calculate class weights using the training partition only.

If SMOTENC is used:
- Apply it only to the training partition.
- Never oversample validation or test data.
- Preserve categorical feature semantics.
- Do not apply ordinary SMOTE blindly to integer-coded categorical columns.

The goal is better fraud detection, not merely balanced training data.

## 9. Evaluation Requirements

Fraud is highly imbalanced. Accuracy alone is insufficient.

Report:

- Precision.
- Recall.
- F1-score.
- PR-AUC.
- ROC-AUC.
- Confusion matrix.
- False-positive and false-negative counts.
- Fraud prevalence.
- Selected probability threshold.

Use the same validation partition for model comparison and threshold selection.

After selecting the model and threshold:
1. Freeze the complete pipeline.
2. Evaluate once on fraudTest.csv.
3. Report final held-out test results separately.
4. Do not tune the model using test results.

Discuss the effect of the lower fraud prevalence in the test dataset.

## 10. Hardware and Implementation Constraints

Development environment:

- Windows.
- Python virtual environment.
- Intel Core i5-3330S.
- 8 GB RAM.
- HDD storage.

Use memory-efficient processing.

Prefer:
- Chunked CSV reading where useful.
- Efficient categorical dtypes.
- Avoiding repeated full-dataset copies.
- Scripts over notebooks.
- Reproducible random seeds.
- Clear logs and metadata.

Use the existing project environment where possible. Do not replace working dependencies without justification.

## 11. Expected Deliverables

Create a reproducible Sparkov pipeline with:

- Data validation and schema audit.
- Chronological train/validation split.
- Leakage-safe feature engineering.
- Preprocessing pipeline.
- Baseline and imbalance experiments.
- Model comparison.
- Validation threshold selection.
- Final held-out test evaluation.
- Saved model and preprocessing artifacts.
- Feature configuration.
- Evaluation metrics and reports.
- Reproducible scripts.
- Clear README documentation.

Use a clean directory structure and descriptive artifact names.

Do not overwrite ULB artifacts.

## 12. Working Rules

- Inspect the existing repository before modifying it.
- Preserve existing work and untracked files.
- Do not use destructive Git commands.
- Do not upload raw Sparkov or BankSim datasets.
- Do not commit virtual environments, caches or temporary files.
- Do not claim generalization across datasets merely because multiple datasets were evaluated.
- Clearly distinguish compatibility testing, validation and independent held-out evaluation.
- Explain every significant modeling decision.
- Keep the pipeline reproducible and suitable for eventual integration into the unified application.

## 13. Immediate Priority

Start with a final review of the existing Sparkov audits and repository structure.

Then implement the Sparkov pipeline independently, preserving the chronological holdout and preventing data leakage.

Do not start the unified dashboard yet. The individual dataset pipelines must first be completed and evaluated.

## 14. Verified Development Environment and Starting State

### Active development workspace

- Repository: credit-card-fraud-detection
- Development branch: sparkov-development
- Starting commit: bcfb5b4
- Starting commit message: Add fraud detection pipeline and evaluation artifacts
- Development worktree: credit-card-fraud-detection-model-test

The Sparkov development branch is based on the existing model-training commit. The original ulb-data-creation worktree must remain untouched.

### Actual Sparkov data access

The following local files are available:

- external_data/Sparkov/fraudTrain.csv (351,238,196 bytes)
- external_data/Sparkov/fraudTest.csv (150,354,339 bytes)

In the development worktree, external_data/Sparkov is a Windows directory junction to the original dataset directory.

The raw CSV files are excluded from Git by the external_data/ ignore rule. Preserve this exclusion. Do not stage, commit, upload, modify or duplicate the raw datasets.

### Existing implementation status

The ULB pipeline and selected ULB LightGBM model already exist in the repository.

The Sparkov pipeline does NOT yet exist.

At the start of this work:
- No Sparkov-trained model exists.
- No Sparkov preprocessing pipeline exists.
- No Sparkov feature-engineering pipeline exists.
- No Sparkov validation or final evaluation scripts exist.
- No Sparkov model artifacts or evaluation reports exist.

Build these components independently. Do not load the ULB model and represent it as a Sparkov model.

### Verified Python environment

Python: 3.13.15 (64-bit)

Verified installed package versions in the original project virtual environment:

- pandas: 3.1.1
- NumPy: 2.5.5
- scikit-learn: 1.9.1
- imbalanced-learn: 0.14.2
- LightGBM: 4.7.0
- XGBoost: 3.4.1

No requirements.txt, pyproject.toml or environment.yml was found in the development worktree during the initial inspection.

Before implementation, inspect package compatibility and create a reproducible dependency manifest. Do not blindly upgrade the working environment.

### Implementation boundary

Keep all Sparkov-specific source code, configuration, reports and artifacts under the sparkov/ directory.

Keep the existing ULB source code, model artifacts, metrics and evaluation workflow intact.

Do not start dashboard integration until the independent Sparkov pipeline has been implemented and evaluated.

### Initial Git state

The development branch contains an untracked external_evaluation/reports/ directory from a previous synthetic ULB experiment. Preserve it. It is not a Sparkov result and must not be presented as one.

Do not stage or commit files without explicit approval.
