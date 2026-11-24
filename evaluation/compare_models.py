
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[1]

METRICS_DIR = ROOT / "artifacts" / "metrics"
PREDICTIONS_DIR = ROOT / "artifacts" / "validation_predictions"

REPORT_DIR = ROOT / "reports"
EVALUATION_DIR = ROOT / "evaluation"

COMPARISON_PATH = REPORT_DIR / "model_comparison.csv"
THRESHOLD_PATH = EVALUATION_DIR / "threshold_analysis.csv"

MODELS = [
    "logistic_regression",
    "lightgbm",
    "xgboost",
    "random_forest",
]

# Include the original comparison threshold.
THRESHOLDS = np.round(
    np.arange(0.01, 1.00, 0.01), 2
)
THRESHOLDS = np.unique(
    np.append(THRESHOLDS, 0.50)
)


# --------------------------------------------------
# 1. CHECK OUTPUT FILES
# --------------------------------------------------

for path in [COMPARISON_PATH, THRESHOLD_PATH]:
    if path.exists():
        raise FileExistsError(
            f"Report already exists. Move it before rerunning:\n{path}"
        )


# --------------------------------------------------
# 2. LOAD AND VERIFY PREDICTIONS
# --------------------------------------------------

print("\nLoading validation predictions...")

model_data = {}
reference_y = None

for name in MODELS:

    prediction_path = (
        PREDICTIONS_DIR / f"{name}_predictions.npz"
    )

    metrics_path = (
        METRICS_DIR / f"{name}_metrics.json"
    )

    with np.load(prediction_path) as data:
        y_true = data["y_true"].copy()
        y_scores = data["y_scores"].copy()

    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    if reference_y is None:
        reference_y = y_true
    elif not np.array_equal(reference_y, y_true):
        raise ValueError(
            f"Validation labels do not match for {name}"
        )

    if len(y_true) != len(y_scores):
        raise ValueError(
            f"Prediction length mismatch for {name}"
        )

    model_data[name] = {
        "y_true": y_true,
        "y_scores": y_scores,
        "metrics": metrics,
    }

    print(f"Verified: {name}")

print("\nAll four models use identical validation labels.")
print("Validation rows:", len(reference_y))


# --------------------------------------------------
# 3. CREATE MODEL COMPARISON REPORT
# --------------------------------------------------

print("\nCreating model comparison report...")

comparison_rows = []

for name in MODELS:

    metrics = model_data[name]["metrics"]

    comparison_rows.append({
        "model": metrics["model"],
        "average_precision": metrics["average_precision"],
        "roc_auc": metrics["roc_auc"],
        "precision_at_0.5": metrics["precision"],
        "recall_at_0.5": metrics["recall"],
        "f1_at_0.5": metrics["f1_score"],
        "false_positives_at_0.5": metrics["false_positives"],
        "false_negatives_at_0.5": metrics["false_negatives"],
        "training_time_seconds": metrics["training_time_seconds"],
    })

comparison_rows.sort(
    key=lambda row: row["average_precision"],
    reverse=True
)

with open(
    COMPARISON_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=comparison_rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(comparison_rows)


# --------------------------------------------------
# 4. ANALYZE CLASSIFICATION THRESHOLDS
# --------------------------------------------------

print("\nAnalyzing thresholds...")

threshold_rows = []

for name in MODELS:

    y_true = model_data[name]["y_true"]
    y_scores = model_data[name]["y_scores"]

    for threshold in THRESHOLDS:

        y_pred = (y_scores >= threshold).astype(np.int8)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        ).ravel()

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        threshold_rows.append({
            "model": name,
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        })


with open(
    THRESHOLD_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=threshold_rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(threshold_rows)


# --------------------------------------------------
# 5. DISPLAY COMPARISON
# --------------------------------------------------

print("\n" + "=" * 65)
print("MODEL COMPARISON: VALIDATION RESULTS AT THRESHOLD 0.50")
print("=" * 65)

for row in comparison_rows:

    print(f"\n{row['model']}")

    print(f"Average Precision: {row['average_precision']:.5f}")
    print(f"ROC-AUC: {row['roc_auc']:.5f}")
    print(f"Precision: {row['precision_at_0.5']:.5f}")
    print(f"Recall: {row['recall_at_0.5']:.5f}")
    print(f"F1-score: {row['f1_at_0.5']:.5f}")
    print(f"False positives: {row['false_positives_at_0.5']}")
    print(f"False negatives: {row['false_negatives_at_0.5']}")


# --------------------------------------------------
# 6. DISPLAY HIGH-RECALL THRESHOLD OPTIONS
# --------------------------------------------------

print("\n" + "=" * 65)
print("BEST PRECISION AT SELECTED MINIMUM RECALL LEVELS")
print("=" * 65)

for name in MODELS:

    print(f"\n{name.upper()}")

    rows = [
        row for row in threshold_rows
        if row["model"] == name
    ]

    for target_recall in [0.90, 0.95]:

        eligible = [
            row for row in rows
            if row["recall"] >= target_recall
        ]

        if not eligible:
            print(
                f"Recall >= {target_recall:.0%}: "
                "Not achieved in tested thresholds"
            )
            continue

        best = max(
            eligible,
            key=lambda row: (
                row["precision"],
                row["f1_score"]
            )
        )

        print(
            f"Recall >= {target_recall:.0%}: "
            f"threshold={best['threshold']:.2f}, "
            f"precision={best['precision']:.4f}, "
            f"recall={best['recall']:.4f}, "
            f"FP={best['false_positives']}, "
            f"FN={best['false_negatives']}"
        )


# --------------------------------------------------
# 7. FINAL STATUS
# --------------------------------------------------

print("\n" + "=" * 65)
print("EVALUATION REPORTS CREATED SUCCESSFULLY")
print("=" * 65)

print(COMPARISON_PATH)
print(THRESHOLD_PATH)

print("\nNo test data was loaded or evaluated.")
