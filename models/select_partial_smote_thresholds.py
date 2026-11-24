
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]

PRED_DIR = ROOT / "artifacts/validation_predictions"

experiments = [
    ("5%", "lightgbm_partial_smote_5_predictions.npz"),
    ("10%", "lightgbm_partial_smote_10_predictions.npz"),
    ("20%", "lightgbm_partial_smote_20_predictions.npz"),
]

print("\nPARTIAL SMOTE THRESHOLD SELECTION")
print("=" * 75)
print("Criterion: Precision strictly above 90%, maximum F1\n")

for label, filename in experiments:

    data = np.load(PRED_DIR / filename)
    y_true = data["y_true"]
    y_prob = data["y_prob"]

    precision, recall, thresholds = precision_recall_curve(
        y_true, y_prob
    )

    precision = precision[:-1]
    recall = recall[:-1]

    f1 = (
        2 * precision * recall /
        (precision + recall + 1e-12)
    )

    eligible = np.where(precision > 0.90)[0]

    print(f"SMOTE RATIO: {label}")
    print("-" * 50)

    if len(eligible) == 0:
        print("No threshold achieves precision above 90%.\n")
        continue

    best_idx = eligible[np.argmax(f1[eligible])]
    threshold = float(thresholds[best_idx])

    predictions = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (predictions == 1)))
    fp = int(np.sum((y_true == 0) & (predictions == 1)))
    fn = int(np.sum((y_true == 1) & (predictions == 0)))

    print("Threshold:", round(threshold, 8))
    print("Precision:", round(float(precision[best_idx]), 6))
    print("Recall:", round(float(recall[best_idx]), 6))
    print("F1:", round(float(f1[best_idx]), 6))
    print("True positives:", tp)
    print("False positives:", fp)
    print("False negatives:", fn)
    print()
