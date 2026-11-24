
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]

PREDICTION_DIR = ROOT / "artifacts" / "validation_predictions"
OUTPUT_DIR = ROOT / "artifacts" / "metrics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "logistic_regression",
    "lightgbm",
    "xgboost",
    "random_forest",
]

MIN_PRECISION = 0.90

results = []

print("\nVALIDATION THRESHOLD SELECTION")
print("=" * 75)
print(f"Required precision: strictly above {MIN_PRECISION:.0%}")
print("Selection objective: highest F1-score among qualifying thresholds")
print("=" * 75)

for model_name in MODELS:
    prediction_file = (
        PREDICTION_DIR / f"{model_name}_predictions.npz"
    )

    if not prediction_file.exists():
        print(f"\nSkipping {model_name}: prediction file not found.")
        continue

    data = np.load(prediction_file)

    y_true = data["y_true"].astype(int)
    y_scores = data["y_scores"].astype(float)

    precision, recall, thresholds = precision_recall_curve(
        y_true, y_scores
    )

    # The final precision/recall values have no corresponding threshold.
    precision = precision[:-1]
    recall = recall[:-1]

    f1_values = (
        2 * precision * recall
        / (precision + recall + 1e-12)
    )

    eligible = np.where(precision > MIN_PRECISION)[0]

    print(f"\nMODEL: {model_name.upper()}")
    print("-" * 75)

    if len(eligible) == 0:
        print("No threshold achieved precision above 90%.")
        continue

    # Choose the highest F1 among thresholds meeting the requirement.
    best_index = eligible[np.argmax(f1_values[eligible])]

    selected_threshold = float(thresholds[best_index])

    y_pred = (y_scores >= selected_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    selected_precision = float(precision[best_index])
    selected_recall = float(recall[best_index])
    selected_f1 = float(f1_values[best_index])

    result = {
        "model": model_name,
        "threshold": selected_threshold,
        "precision": selected_precision,
        "recall": selected_recall,
        "f1_score": selected_f1,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "validation_rows": int(len(y_true)),
        "validation_frauds": int(np.sum(y_true == 1)),
    }

    results.append(result)

    print(f"Selected threshold : {selected_threshold:.8f}")
    print(f"Precision          : {selected_precision:.4%}")
    print(f"Recall             : {selected_recall:.4%}")
    print(f"F1-score           : {selected_f1:.4f}")
    print(f"True positives     : {tp}")
    print(f"False positives    : {fp}")
    print(f"False negatives    : {fn}")
    print(f"True negatives     : {tn}")

if not results:
    raise RuntimeError(
        "No model achieved precision above 90% on validation."
    )

# Rank qualifying model-threshold combinations by F1.
results.sort(
    key=lambda item: item["f1_score"],
    reverse=True
)

print("\n\nFINAL CANDIDATE RANKING")
print("=" * 90)

for rank, result in enumerate(results, start=1):
    print(
        f"{rank}. {result['model']:<22}"
        f"Threshold={result['threshold']:.6f}  "
        f"Precision={result['precision']:.2%}  "
        f"Recall={result['recall']:.2%}  "
        f"F1={result['f1_score']:.4f}"
    )

best_candidate = results[0]

print("\nBEST VALIDATION CANDIDATE")
print("=" * 75)
print(json.dumps(best_candidate, indent=4))

output_file = OUTPUT_DIR / "threshold_selection_candidates.json"

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(
        {
            "selection_rule": (
                "Highest F1-score among thresholds with "
                "precision strictly greater than 0.90"
            ),
            "minimum_precision_exclusive": MIN_PRECISION,
            "ranked_candidates": results,
            "best_candidate": best_candidate,
        },
        file,
        indent=4,
    )

print(f"\nCandidate results saved to: {output_file}")
print("\nNo test data was loaded or evaluated.")
