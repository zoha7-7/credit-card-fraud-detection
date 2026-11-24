
from pathlib import Path
import json

import joblib
import numpy as np
import lightgbm as lgb

from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "artifacts/datasets/original_training_data_scaled.npz"
VAL_PATH = ROOT / "artifacts/datasets/training_validation_data.npz"

MODEL_DIR = ROOT / "artifacts/trained_models"
PRED_DIR = ROOT / "artifacts/validation_predictions"
METRIC_DIR = ROOT / "artifacts/metrics"

RATIOS = [0.025, 0.0325, 0.035, 0.0375]

RANDOM_STATE = 42
K_NEIGHBORS = 5

# Keep the existing LightGBM configuration unchanged.
MODEL_PARAMS = {
    "n_estimators": 200,
    "num_leaves": 15,
    "max_depth": 5,
    "learning_rate": 0.05,
    "min_child_samples": 50,
    "max_bin": 127,
    "random_state": 42,
    "n_jobs": 2,
    "verbosity": -1,
}


def main():
    for folder in [MODEL_DIR, PRED_DIR, METRIC_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    data = np.load(DATA_PATH)
    val_data = np.load(VAL_PATH)

    X_train = data["X_train_scaled"]
    y_train = data["y_train"]

    X_val = val_data["X_val_scaled"]
    y_val = val_data["y_val"]

    print("Training data:", X_train.shape)
    print("Validation data:", X_val.shape)
    print("Original class counts:", np.bincount(y_train))

    # Check every output path before starting.
    for ratio in RATIOS:
        tag = f"lightgbm_smote_ratio_{str(ratio).replace('.', '_')}"

        paths = [
            MODEL_DIR / f"{tag}.joblib",
            PRED_DIR / f"{tag}_predictions.npz",
            METRIC_DIR / f"{tag}_metrics.json",
        ]

        existing = [p for p in paths if p.exists()]
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite existing artifacts:\n"
                + "\n".join(str(p) for p in existing)
            )

    results = []

    for ratio in RATIOS:
        tag = f"lightgbm_smote_ratio_{str(ratio).replace('.', '_')}"

        print("\n" + "=" * 70)
        print(f"SMOTE RATIO: {ratio:.3%}")
        print("=" * 70)

        smote = SMOTE(
            sampling_strategy=ratio,
            random_state=RANDOM_STATE,
            k_neighbors=K_NEIGHBORS,
        )

        X_resampled, y_resampled = smote.fit_resample(
            X_train, y_train
        )

        print("Resampled shape:", X_resampled.shape)
        print("Resampled class counts:", np.bincount(y_resampled))

        model = lgb.LGBMClassifier(**MODEL_PARAMS)

        model.fit(
            X_resampled,
            y_resampled,
            eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
            callbacks=[
                lgb.early_stopping(25, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred_050 = (y_prob >= 0.50).astype(int)

        metrics = {
            "experiment": tag,
            "sampling_strategy": ratio,
            "original_training_rows": int(len(y_train)),
            "resampled_training_rows": int(len(y_resampled)),
            "resampled_class_counts": np.bincount(
                y_resampled
            ).tolist(),
            "best_iteration": int(model.best_iteration_),
            "validation_average_precision": float(
                average_precision_score(y_val, y_prob)
            ),
            "validation_roc_auc": float(
                roc_auc_score(y_val, y_prob)
            ),
            "precision_at_0.50": float(
                precision_score(y_val, y_pred_050, zero_division=0)
            ),
            "recall_at_0.50": float(
                recall_score(y_val, y_pred_050, zero_division=0)
            ),
            "f1_at_0.50": float(
                f1_score(y_val, y_pred_050, zero_division=0)
            ),
        }

        joblib.dump(model, MODEL_DIR / f"{tag}.joblib")

        np.savez_compressed(
            PRED_DIR / f"{tag}_predictions.npz",
            y_true=y_val,
            y_prob=y_prob,
        )

        with open(METRIC_DIR / f"{tag}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        results.append(metrics)

        print("Best iteration:", metrics["best_iteration"])
        print("Validation PR-AUC:", round(
            metrics["validation_average_precision"], 6
        ))
        print("Validation ROC-AUC:", round(
            metrics["validation_roc_auc"], 6
        ))
        print("Precision @ 0.50:", round(
            metrics["precision_at_0.50"], 4
        ))
        print("Recall @ 0.50:", round(
            metrics["recall_at_0.50"], 4
        ))
        print("F1 @ 0.50:", round(
            metrics["f1_at_0.50"], 4
        ))

    print("\n" + "=" * 70)
    print("SMOTE RATIO EXPERIMENT SUMMARY")
    print("=" * 70)

    for item in results:
        print(
            f"Ratio: {item['sampling_strategy']:.3%} | "
            f"PR-AUC: {item['validation_average_precision']:.6f} | "
            f"ROC-AUC: {item['validation_roc_auc']:.6f} | "
            f"Best iteration: {item['best_iteration']}"
        )

    print("\nAll experiments completed.")
    print("Existing models and artifacts were not modified.")


if __name__ == "__main__":
    main()
