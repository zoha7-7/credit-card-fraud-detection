
from pathlib import Path
import json

import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score,
)

# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "artifacts" / "trained_models"
    / "lightgbm_partial_smote_5.joblib"
)

DATA_PATH = (
    ROOT / "artifacts" / "datasets"
    / "test_data_model_scaled.npz"
)

OUTPUT_DIR = (
    ROOT / "artifacts" / "test_evaluation_partial_smote_5"
)

THRESHOLD = 0.90566961

# ============================================================
# 2. SAFETY CHECKS
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

if not DATA_PATH.exists():
    raise FileNotFoundError(DATA_PATH)

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Output directory already exists: {OUTPUT_DIR}\n"
        "Inspect existing results before running again."
    )

# ============================================================
# 3. LOAD EXISTING TEST DATA
# ============================================================

print("\nLoading existing test dataset...")

data = np.load(DATA_PATH)

X_test = data["X_test"]
y_test = data["y_test"]

data.close()

if X_test.shape != (56746, 30):
    raise ValueError(f"Unexpected test shape: {X_test.shape}")

if y_test.shape != (56746,):
    raise ValueError(f"Unexpected label shape: {y_test.shape}")

if not np.isfinite(X_test).all():
    raise ValueError("Test features contain non-finite values.")

if set(np.unique(y_test)) != {0, 1}:
    raise ValueError("Expected binary labels 0 and 1.")

if int(np.sum(y_test == 1)) != 95:
    raise ValueError("Unexpected fraud count.")

print("Test rows:", len(y_test))
print("Legitimate:", int(np.sum(y_test == 0)))
print("Fraudulent:", int(np.sum(y_test == 1)))

# ============================================================
# 4. LOAD SAVED MODEL
# ============================================================

print("\nLoading 5% Partial SMOTE LightGBM model...")

model = joblib.load(MODEL_PATH)

if not hasattr(model, "predict_proba"):
    raise TypeError("Model does not support predict_proba().")

if model.n_features_in_ != X_test.shape[1]:
    raise ValueError("Model and test feature counts do not match.")

classes = list(model.classes_)

if 1 not in classes:
    raise ValueError("Fraud class 1 is missing.")

fraud_column = classes.index(1)

# ============================================================
# 5. GENERATE TEST PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_scores = model.predict_proba(X_test)[:, fraud_column]

if not np.isfinite(y_scores).all():
    raise ValueError("Model generated invalid probability scores.")

# Frozen threshold selected using validation data only
y_pred = (y_scores >= THRESHOLD).astype(np.int8)

# ============================================================
# 6. CALCULATE TEST METRICS
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
).ravel()

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

specificity = tn / (tn + fp)

false_positive_rate = fp / (fp + tn)

balanced_accuracy = balanced_accuracy_score(
    y_test, y_pred
)

roc_auc = roc_auc_score(y_test, y_scores)

pr_auc = average_precision_score(y_test, y_scores)

# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

results = {
    "experiment": "lightgbm_partial_smote_5",
    "evaluation": "holdout_test",
    "threshold": THRESHOLD,
    "threshold_selection": "validation_only",
    "test_used_for_threshold_selection": False,
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "pr_auc_average_precision": float(pr_auc),
    "specificity": float(specificity),
    "false_positive_rate": float(false_positive_rate),
    "balanced_accuracy": float(balanced_accuracy),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),
    "test_rows": int(len(y_test)),
    "test_frauds": int(np.sum(y_test == 1)),
    "test_legitimate": int(np.sum(y_test == 0)),
}

print("\n" + "=" * 65)
print("5% PARTIAL SMOTE - FINAL TEST EVALUATION")
print("=" * 65)

print(f"Threshold          : {THRESHOLD:.8f}")
print(f"Accuracy           : {accuracy:.6f}")
print(f"Precision          : {precision:.6f}")
print(f"Recall             : {recall:.6f}")
print(f"F1-score           : {f1:.6f}")
print(f"ROC-AUC            : {roc_auc:.6f}")
print(f"PR-AUC             : {pr_auc:.6f}")
print(f"Specificity        : {specificity:.6f}")
print(f"False positive rate: {false_positive_rate:.6f}")
print(f"Balanced accuracy  : {balanced_accuracy:.6f}")

print("\nConfusion Matrix:")
print(f"TN: {tn} | FP: {fp}")
print(f"FN: {fn} | TP: {tp}")

# ============================================================
# 8. SAVE RESULTS
# ============================================================

OUTPUT_DIR.mkdir(parents=True)

with open(
    OUTPUT_DIR / "partial_smote_5_test_results.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(results, file, indent=4)

np.savez_compressed(
    OUTPUT_DIR / "partial_smote_5_test_predictions.npz",
    y_true=y_test,
    y_scores=y_scores,
    y_pred=y_pred,
    threshold=np.array(THRESHOLD),
)

print("\n" + "=" * 65)
print("TEST EVALUATION COMPLETED")
print("=" * 65)

print("Saved results:")
print(OUTPUT_DIR / "partial_smote_5_test_results.json")
print(OUTPUT_DIR / "partial_smote_5_test_predictions.npz")

print("\nThreshold was not adjusted using test results.")
print("Model was not retrained.")
