
from pathlib import Path
import json
import csv

import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    balanced_accuracy_score
)


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "artifacts" / "trained_models"
DATA_PATH = ROOT / "artifacts" / "datasets" / "test_data_model_scaled.npz"

OUTPUT_DIR = ROOT / "artifacts" / "test_evaluation"

THRESHOLDS = {
    "logistic_regression": 1.00000000,
    "lightgbm": 0.8830631197603867,
    "xgboost": 0.84469706,
    "random_forest": 0.83503902
}

MODEL_FILES = {
    "logistic_regression": "logistic_regression.joblib",
    "lightgbm": "lightgbm.joblib",
    "xgboost": "xgboost.joblib",
    "random_forest": "random_forest.joblib"
}


# ============================================================
# 2. SAFETY CHECKS
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(DATA_PATH)

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Output directory already exists: {OUTPUT_DIR}\n"
        "Inspect existing results before running again."
    )

for filename in MODEL_FILES.values():
    if not (MODEL_DIR / filename).exists():
        raise FileNotFoundError(MODEL_DIR / filename)

OUTPUT_DIR.mkdir(parents=True)


# ============================================================
# 3. LOAD UNTOUCHED TEST LABELS AND PREPARED FEATURES
# ============================================================

print("\nLoading final test dataset...")

data = np.load(DATA_PATH)

X_test = data["X_test"]
y_test = data["y_test"]

data.close()

if X_test.shape != (56746, 30):
    raise ValueError(f"Unexpected test feature shape: {X_test.shape}")

if y_test.shape != (56746,):
    raise ValueError(f"Unexpected test label shape: {y_test.shape}")

if not np.isfinite(X_test).all():
    raise ValueError("Test data contains non-finite values.")

if set(np.unique(y_test)) != {0, 1}:
    raise ValueError("Expected binary labels 0 and 1.")

if int(np.sum(y_test == 1)) != 95:
    raise ValueError("Unexpected number of fraud cases.")

print("Test shape:", X_test.shape)
print("Legitimate transactions:", int(np.sum(y_test == 0)))
print("Fraudulent transactions:", int(np.sum(y_test == 1)))


# ============================================================
# 4. EVALUATE EACH MODEL
# ============================================================

results = {}
curve_data = {}

for model_name, filename in MODEL_FILES.items():

    print("\n" + "=" * 65)
    print("EVALUATING:", model_name.upper())
    print("=" * 65)

    model_path = MODEL_DIR / filename

    model = joblib.load(model_path)

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"{model_name} does not support predict_proba()."
        )

    if getattr(model, "n_features_in_", 30) != X_test.shape[1]:
        raise ValueError(
            f"Feature count mismatch for {model_name}."
        )

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            f"Fraud class 1 missing from {model_name}."
        )

    fraud_column = classes.index(1)

    # Obtain fraud probabilities
    y_scores = model.predict_proba(X_test)[:, fraud_column]

    if not np.isfinite(y_scores).all():
        raise ValueError(
            f"Invalid probability scores from {model_name}."
        )

    threshold = THRESHOLDS[model_name]

    # Apply frozen validation threshold
    y_pred = (y_scores >= threshold).astype(np.int8)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    ).ravel()

    # Metrics
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

    specificity = tn / (tn + fp) if (tn + fp) else 0

    fpr = fp / (fp + tn) if (fp + tn) else 0

    balanced_accuracy = balanced_accuracy_score(
        y_test, y_pred
    )

    roc_auc = roc_auc_score(y_test, y_scores)

    pr_auc = average_precision_score(y_test, y_scores)

    # Save metrics
    results[model_name] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc_average_precision": float(pr_auc),
        "specificity": float(specificity),
        "false_positive_rate": float(fpr),
        "balanced_accuracy": float(balanced_accuracy),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "test_rows": int(len(y_test)),
        "test_frauds": int(np.sum(y_test == 1))
    }

    curve_data[model_name] = {
        "scores": y_scores,
        "predictions": y_pred
    }

    print(f"Threshold          : {threshold:.8f}")
    print(f"Accuracy           : {accuracy:.6f}")
    print(f"Precision          : {precision:.6f}")
    print(f"Recall             : {recall:.6f}")
    print(f"F1-score           : {f1:.6f}")
    print(f"ROC-AUC            : {roc_auc:.6f}")
    print(f"PR-AUC             : {pr_auc:.6f}")
    print(f"Specificity        : {specificity:.6f}")
    print(f"False positive rate: {fpr:.6f}")

    print("\nConfusion Matrix:")
    print(f"TN: {tn} | FP: {fp}")
    print(f"FN: {fn} | TP: {tp}")

    del model


# ============================================================
# 5. RANK MODELS BY TEST F1-SCORE
# ============================================================

ranking = sorted(
    results.items(),
    key=lambda item: item[1]["f1_score"],
    reverse=True
)

print("\n" + "=" * 75)
print("FINAL TEST MODEL RANKING")
print("=" * 75)

for rank, (name, metrics) in enumerate(ranking, start=1):

    print(
        f"{rank}. {name:<22}"
        f"Precision={metrics['precision']:.4%}  "
        f"Recall={metrics['recall']:.4%}  "
        f"F1={metrics['f1_score']:.4f}"
    )

best_model = ranking[0][0]

print("\nBEST TEST MODEL:", best_model.upper())


# ============================================================
# 6. SAVE JSON RESULTS
# ============================================================

json_output = {
    "evaluation": "final_holdout_test",
    "test_rows": int(len(y_test)),
    "test_frauds": int(np.sum(y_test == 1)),
    "test_legitimate": int(np.sum(y_test == 0)),
    "threshold_selection": "validation_only",
    "test_used_for_threshold_selection": False,
    "ranking_metric": "F1-score",
    "best_test_model": best_model,
    "models": results
}

with open(
    OUTPUT_DIR / "final_test_results.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(json_output, f, indent=4)


# ============================================================
# 7. SAVE CSV RESULTS
# ============================================================

with open(
    OUTPUT_DIR / "final_test_results.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=["model"] + list(results[best_model].keys())
    )

    writer.writeheader()

    for name, metrics in ranking:
        writer.writerow({
            "model": name,
            **metrics
        })


# ============================================================
# 8. GENERATE ROC CURVES
# ============================================================

plt.figure(figsize=(9, 7))

for name, values in curve_data.items():

    fpr_values, tpr_values, _ = roc_curve(
        y_test, values["scores"]
    )

    plt.plot(
        fpr_values,
        tpr_values,
        label=f"{name} (AUC={results[name]['roc_auc']:.4f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Final Test Evaluation")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "roc_curves.png",
    dpi=300
)

plt.close()


# ============================================================
# 9. GENERATE PRECISION-RECALL CURVES
# ============================================================

plt.figure(figsize=(9, 7))

for name, values in curve_data.items():

    precision_values, recall_values, _ = precision_recall_curve(
        y_test, values["scores"]
    )

    plt.plot(
        recall_values,
        precision_values,
        label=f"{name} (AP={results[name]['pr_auc_average_precision']:.4f})"
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves - Final Test Evaluation")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "precision_recall_curves.png",
    dpi=300
)

plt.close()


# ============================================================
# 10. GENERATE CONFUSION MATRICES
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(11, 9))

for ax, (name, values) in zip(axes.flat, curve_data.items()):

    cm = confusion_matrix(
        y_test,
        values["predictions"],
        labels=[0, 1]
    )

    ax.imshow(cm)

    ax.set_title(name.replace("_", " ").title())

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center"
            )

plt.suptitle("Confusion Matrices - Final Test Evaluation")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrices.png",
    dpi=300
)

plt.close()


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FINAL TEST EVALUATION COMPLETED")
print("=" * 75)

print("Best model:", best_model.upper())

print("\nSaved files:")

for path in sorted(OUTPUT_DIR.iterdir()):
    print(path.name)

print("\nTest data was not used for threshold selection.")
print("All four trained models were evaluated.")
print("No model retraining was performed.")
