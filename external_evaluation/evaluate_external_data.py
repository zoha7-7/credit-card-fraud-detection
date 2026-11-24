
from pathlib import Path
import json
import sys
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
    classification_report
)

from prepare_external_data import prepare_external_data


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "external_evaluation" / "external_config.json"


def evaluate(csv_path):

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    model_path = BASE_DIR / config["model"]
    threshold = config["threshold"]

    # Prepare external data
    X, y_true = prepare_external_data(csv_path)

    # Load frozen model
    model = joblib.load(model_path)

    # Generate predictions
    probabilities = model.predict_proba(
        X,
        num_iteration=model.best_iteration_
    )[:, 1]

    predictions = (probabilities >= threshold).astype(int)

    print("\n========== EXTERNAL EVALUATION ==========")
    print("Dataset:", Path(csv_path).name)
    print("Transactions:", len(X))
    print("Threshold:", threshold)

    results = {
        "dataset": Path(csv_path).name,
        "transactions": int(len(X)),
        "threshold": float(threshold),
        "model": "LightGBM 3% SMOTE",
        "ground_truth_available": y_true is not None,
        "fraud_predictions": int(predictions.sum()),
        "legitimate_predictions": int(len(predictions) - predictions.sum())
    }

    # Evaluate if ground-truth labels exist
    if y_true is not None:

        y_true = np.asarray(y_true, dtype=int)

        if len(np.unique(y_true)) == 2:

            tn, fp, fn, tp = confusion_matrix(
                y_true,
                predictions,
                labels=[0, 1]
            ).ravel()

            results.update({
                "accuracy": float(accuracy_score(y_true, predictions)),
                "precision": float(precision_score(
                    y_true, predictions, zero_division=0
                )),
                "recall": float(recall_score(
                    y_true, predictions, zero_division=0
                )),
                "f1_score": float(f1_score(
                    y_true, predictions, zero_division=0
                )),
                "roc_auc": float(roc_auc_score(
                    y_true, probabilities
                )),
                "pr_auc": float(average_precision_score(
                    y_true, probabilities
                )),
                "confusion_matrix": {
                    "TN": int(tn),
                    "FP": int(fp),
                    "FN": int(fn),
                    "TP": int(tp)
                }
            })

            print("\nClassification report:")
            print(classification_report(
                y_true, predictions, zero_division=0
            ))

            print("ROC-AUC:", results["roc_auc"])
            print("PR-AUC:", results["pr_auc"])
            print("Confusion matrix:", results["confusion_matrix"])

        else:

            results["evaluation_note"] = (
                "Only one ground-truth class is present. "
                "Full binary classification metrics are unavailable."
            )

            print("\nOnly one class is present.")
            print("Full fraud-detection metrics cannot be calculated.")
            print("Actual class:", int(y_true[0]))
            print("Accuracy:", accuracy_score(y_true, predictions))

    else:

        print("\nNo ground-truth labels.")
        print("Only prediction counts and probabilities are available.")

    # Save report without overwriting existing reports
    output_dir = BASE_DIR / "external_evaluation" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (
        Path(csv_path).stem + "_report.json"
    )

    if output_path.exists():
        raise FileExistsError(
            f"Report already exists: {output_path}"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("\nReport saved:", output_path)

    return results


if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python evaluate_external_data.py path_to_csv"
        )

    evaluate(sys.argv[1])
