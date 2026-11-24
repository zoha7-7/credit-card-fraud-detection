
from pathlib import Path
import numpy as np
import json
from sklearn.metrics import precision_recall_curve, f1_score

ROOT = Path(__file__).resolve().parents[1]

PRED_DIR = ROOT / "artifacts/validation_predictions"
METRIC_DIR = ROOT / "artifacts/metrics"

weights = [10, 25, 50]

print("\nWEIGHTED LIGHTGBM THRESHOLD SELECTION")
print("=" * 75)
print("Criterion: Precision strictly above 90%, maximum F1")
print()

for weight in weights:

    name = f"lightgbm_weight_{weight}"
    pred_path = PRED_DIR / f"{name}_predictions.npz"

    data = np.load(pred_path)
    y_true = data["y_true"]
    y_prob = data["y_prob"]

    precision, recall, thresholds = precision_recall_curve(
        y_true, y_prob
    )

    # The final precision/recall element has no corresponding threshold.
    precision = precision[:-1]
    recall = recall[:-1]

    f1 = (
        2 * precision * recall /
        (precision + recall + 1e-12)
    )

    eligible = np.where(precision > 0.90)[0]

    print(f"WEIGHT: {weight}")
    print("-" * 50)

    if len(eligible) == 0:
        print("No threshold achieves precision above 90%.\n")
        continue

    best_idx = eligible[np.argmax(f1[eligible])]
    threshold = float(thresholds[best_idx])

    predictions = (y_prob >= threshold).astype(int)

    result = {
        "weight": weight,
        "threshold": threshold,
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "f1": float(f1[best_idx]),
        "true_positives": int(np.sum((y_true == 1) & (predictions == 1))),
        "false_positives": int(np.sum((y_true == 0) & (predictions == 1))),
        "false_negatives": int(np.sum((y_true == 1) & (predictions == 0))),
    }

    print("Threshold:", round(result["threshold"], 8))
    print("Precision:", round(result["precision"], 6))
    print("Recall:", round(result["recall"], 6))
    print("F1:", round(result["f1"], 6))
    print("TP:", result["true_positives"])
    print("FP:", result["false_positives"])
    print("FN:", result["false_negatives"])
    print()
