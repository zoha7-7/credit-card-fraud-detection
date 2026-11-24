
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "artifacts/validation_predictions"

EXPERIMENTS = {
    "2%": "lightgbm_smote_ratio_0_02_predictions.npz",
    "2.5%": "lightgbm_smote_ratio_0_025_predictions.npz",
    "3%": "lightgbm_smote_ratio_0_03_predictions.npz",
    "3.25%": "lightgbm_smote_ratio_0_0325_predictions.npz",
    "3.5%": "lightgbm_smote_ratio_0_035_predictions.npz",
    "3.75%": "lightgbm_smote_ratio_0_0375_predictions.npz",
    "4%": "lightgbm_smote_ratio_0_04_predictions.npz",
    "5%": "lightgbm_partial_smote_5_predictions.npz",
    "6%": "lightgbm_smote_ratio_0_06_predictions.npz",
    "8%": "lightgbm_smote_ratio_0_08_predictions.npz",
    "10%": "lightgbm_partial_smote_10_predictions.npz",
    "12.5%": "lightgbm_smote_ratio_0_125_predictions.npz",
    "15%": "lightgbm_smote_ratio_0_15_predictions.npz",
    "20%": "lightgbm_partial_smote_20_predictions.npz",
    "3% A_simple_trees": "lightgbm_smote_3_A_simple_trees_predictions.npz",
"3% B_complex_trees": "lightgbm_smote_3_B_complex_trees_predictions.npz",
"3% C_stronger_regularization": "lightgbm_smote_3_C_stronger_regularization_predictions.npz",
"3% D_min_child_75": "lightgbm_smote_3_D_min_child_75_predictions.npz",
"3% E_min_child_125": "lightgbm_smote_3_E_min_child_125_predictions.npz",
}

MIN_PRECISION = 0.90
reference_y = None
results = []

for ratio, filename in EXPERIMENTS.items():
    path = PRED_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")

    data = np.load(path)
    y_true = data["y_true"]
    y_prob = data["y_prob"]

    if reference_y is None:
        reference_y = y_true.copy()
    elif not np.array_equal(reference_y, y_true):
        raise ValueError(f"Validation labels differ for {ratio}")

    precision, recall, thresholds = precision_recall_curve(
        y_true, y_prob
    )

    precision = precision[:-1]
    recall = recall[:-1]

    f1 = (
        2 * precision * recall
        / np.maximum(precision + recall, 1e-12)
    )

    eligible = np.where(precision > MIN_PRECISION)[0]

    if len(eligible) == 0:
        print(f"\n{ratio}: No threshold achieves precision > 90%.")
        continue

    # Maximize F1; break ties using higher precision.
    best_idx = max(
        eligible,
        key=lambda i: (f1[i], precision[i], thresholds[i])
    )

    threshold = float(thresholds[best_idx])
    predictions = (y_prob >= threshold).astype(int)

    tp = int(np.sum((predictions == 1) & (y_true == 1)))
    fp = int(np.sum((predictions == 1) & (y_true == 0)))
    fn = int(np.sum((predictions == 0) & (y_true == 1)))

    result = {
        "ratio": ratio,
        "threshold": threshold,
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "f1": float(f1[best_idx]),
        "average_precision": float(
            average_precision_score(y_true, y_prob)
        ),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }

    results.append(result)

    print("\n" + "=" * 55)
    print(f"SMOTE RATIO: {ratio}")
    print("=" * 55)
    print(f"Threshold: {threshold:.8f}")
    print(f"Precision: {precision[best_idx]:.6f}")
    print(f"Recall:    {recall[best_idx]:.6f}")
    print(f"F1:        {f1[best_idx]:.6f}")
    print(f"PR-AUC:    {result['average_precision']:.6f}")
    print(f"TP: {tp} | FP: {fp} | FN: {fn}")

print("\n\nFINAL RANKING: PRECISION > 90%, MAXIMUM F1")
print("=" * 75)

for result in sorted(
    results,
    key=lambda x: (x["f1"], x["precision"]),
    reverse=True
):
    print(
        f"{result['ratio']:>6} | "
        f"F1: {result['f1']:.6f} | "
        f"P: {result['precision']:.4%} | "
        f"R: {result['recall']:.4%} | "
        f"TP: {result['tp']:2} | "
        f"FP: {result['fp']:2} | "
        f"FN: {result['fn']:2} | "
        f"Threshold: {result['threshold']:.6f}"
    )

