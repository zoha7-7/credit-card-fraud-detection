
from pathlib import Path
import json

import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
)

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "artifacts/trained_models/lightgbm_smote_ratio_0_03.joblib"
)
TEST_PATH = ROOT / "artifacts/datasets/test_data_model_scaled.npz"

OUTPUT_DIR = ROOT / "artifacts/test_evaluation_baseline_3smote"
OUTPUT_PATH = OUTPUT_DIR / "final_baseline_test_metrics.json"

THRESHOLD = 0.85232879
EXPECTED_ITERATION = 175


def main():

    print("Loading final baseline model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    if not TEST_PATH.exists():
        raise FileNotFoundError(TEST_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing results: {OUTPUT_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    best_iteration = getattr(model, "best_iteration_", None)

    print("Model best iteration:", best_iteration)

    if best_iteration != EXPECTED_ITERATION:
        raise ValueError(
            f"Expected best iteration {EXPECTED_ITERATION}, "
            f"but found {best_iteration}. Stop and verify the model."
        )

    print("\nLoading frozen test dataset...")

    with np.load(TEST_PATH) as data:
        print("Available test arrays:", data.files)

        X_test = data["X_test"]
        y_test = data["y_test"]

    y_test = y_test.astype(np.int8)

    print("Test shape:", X_test.shape)
    print("Test class distribution:", np.bincount(y_test))

    if X_test.shape[1] != 30:
        raise ValueError("Unexpected test feature count.")

    # No retraining. Use the frozen model and iteration.
    print("\nGenerating test predictions...")

    y_prob = model.predict_proba(
        X_test,
        num_iteration=EXPECTED_ITERATION,
    )[:, 1]

    # Frozen threshold selected on validation data.
    y_pred = (y_prob >= THRESHOLD).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    ).ravel()

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
    fpr = fp / (fp + tn)

    metrics = {
        "model": "LightGBM 3% SMOTE baseline",
        "smote_ratio": 0.03,
        "min_child_samples": 50,
        "n_estimators_used": EXPECTED_ITERATION,
        "threshold": THRESHOLD,
        "test_rows": int(len(y_test)),
        "test_fraud_count": int(np.sum(y_test == 1)),
        "test_legitimate_count": int(np.sum(y_test == 0)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "specificity": float(specificity),
        "false_positive_rate": float(fpr),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, y_pred)
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "test_set_prior_exposure_note": (
            "The outer test set was previously used to evaluate "
            "another SMOTE candidate in this project."
        ),
    }

    print("\n" + "=" * 60)
    print("FINAL BASELINE TEST RESULTS")
    print("=" * 60)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nResults saved to:", OUTPUT_PATH)
    print("\nTEST EVALUATION COMPLETED.")


if __name__ == "__main__":
    main()
