# Credit Card Fraud Detection

A multi-dataset fraud detection project with independent pipelines.

## Project Architecture

| Dataset | Purpose | Status |
|---|---|---|
| ULB Credit Card | Existing anonymized transaction model | Implemented |
| Sparkov | Independent transaction-level fraud pipeline | Development pending |
| IEEE-CIS | Future independent pipeline | Planned |

Different datasets will retain their own feature schemas and preprocessing pipelines.

## Repository Structure

- `artifacts/`: Existing ULB models and preprocessing artifacts.
- `docs/`: Project documentation and implementation handoffs.
- `external_data/Sparkov/`: Local Sparkov raw datasets, excluded from Git.
- `external_evaluation/`: External evaluation reports.
- `sparkov/`: Location for the new Sparkov pipeline.

## ULB Pipeline

The existing ULB pipeline must remain unchanged.

- Selected model: LightGBM with 3% SMOTE training ratio.
- Model: `artifacts/trained_models/lightgbm_smote_ratio_0_03.joblib`
- Imputer: `artifacts/preprocessing/validation_imputer.joblib`
- Scaler: `artifacts/preprocessing/validation_scaler.joblib`
- Feature configuration: `artifacts/datasets/feature_columns.json`
- External evaluation configuration: `external_evaluation/external_config.json`
- Selected threshold: approximately `0.85232879`

## Sparkov Pipeline

**Status: Not yet implemented.**

Raw datasets:

- Training: `external_data/Sparkov/fraudTrain.csv`
- Held-out test: `external_data/Sparkov/fraudTest.csv`

There is currently no Sparkov model, preprocessing pipeline, feature engineering implementation, or evaluation pipeline.

Planned workflow:

1. Validate and audit the datasets.
2. Establish chronological training and validation splits.
3. Engineer leakage-safe features.
4. Build preprocessing and model-training pipelines.
5. Compare models and imbalance strategies.
6. Select a decision threshold using validation data.
7. Freeze the final pipeline.
8. Evaluate once on the untouched `fraudTest.csv`.
9. Save models, preprocessing artifacts, configurations, and reports.

See `docs/SPARKOV_ASTRA_HANDOFF.md` for detailed requirements and data-audit findings.

## Development Environment

Verified versions:

- Python 3.13.15
- pandas 3.1.1
- NumPy 2.5.5
- scikit-learn 1.9.1
- imbalanced-learn 0.14.2
- LightGBM 4.7.0
- XGBoost 3.4.1

A reproducible dependency manifest has not yet been established. Finalize dependencies after implementation compatibility checks.

## Hardware Constraints

- Intel Core i5-3330S
- 8 GB RAM
- HDD storage

Prefer memory-efficient processing, chunked loading where appropriate, and scripts over unnecessarily large notebooks.

## Development Rules

- Keep ULB and Sparkov pipelines independent.
- Do not modify existing ULB models or preprocessing artifacts.
- Never commit raw datasets or large generated intermediate files.
- Keep Sparkov code, configuration, reports, and artifacts inside `sparkov/`.
- Preserve existing audit reports.
- Do not build a unified dashboard until individual pipelines are complete.
- Do not stage, commit, or push without explicit approval.

## Current Development Branch

- Branch: `sparkov-development`
- Base commit: `bcfb5b4`
- Purpose: Develop the independent Sparkov pipeline while preserving ULB.
