
from pathlib import Path
import json

import numpy as np
import lightgbm as lgb
import joblib

from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "artifacts/datasets/original_training_data_scaled.npz"
VAL_PATH = ROOT / "artifacts/datasets/training_validation_data.npz"

MODEL_DIR = ROOT / "artifacts/trained_models"
PRED_DIR = ROOT / "artifacts/validation_predictions"
METRIC_DIR = ROOT / "artifacts/metrics"

BASE_PARAMS = {
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

EXPERIMENTS = {
    "A_simple_trees": {
        **BASE_PARAMS,
        "num_leaves": 7,
        "max_depth": 4,
    },
    "B_complex_trees": {
        **BASE_PARAMS,
        "num_leaves": 31,
        "max_depth": 6,
    },
    "C_stronger_regularization": {
        **BASE_PARAMS,
        "min_child_samples": 100,
    },
}

SMOTE_RATIO = 0.03
RANDOM_STATE = 42
K_NEIGHBORS = 5


def main():
    for folder in [MODEL_DIR, PRED_DIR, METRIC_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    # Load the exact original training and validation datasets.
    train = np.load(DATA_PATH)
    validation = np.load(VAL_PATH)

    X_train = train["X_train_scaled"]
    y_train = train["y_train"]

    X_val = validation["X_val_scaled"]
    y_val = validation["y_val"]

    # Preserve the existing 3% SMOTE experiment.
    smote = SMOTE(
        sampling_strategy=SMOTE_RATIO,
        random_state=RANDOM_STATE,
        k_neighbors=K_NEIGHBORS,
    )

    X_resampled, y_resampled = smote.fit_resample(
        X_train, y_train
    )

    print("Training shape:", X_train.shape)
    print("Resampled shape:", X_resampled.shape)
    print("Validation shape:", X_val.shape)
    print("Class counts:", np.bincount(y_resampled))

    # Refuse to overwrite any previous experiment.
    for name in EXPERIMENTS:
        tag = f"lightgbm_smote_3_{name}"

        paths = [
            MODEL_DIR / f"{tag}.joblib",
            PRED_DIR / f"{tag}_predictions.npz",
            METRIC_DIR / f"{tag}_metrics.json",
        ]

        existing = [p for p in paths if p.exists()]
        if existing:
            raise FileExistsError(
                "Existing artifacts found. Refusing to overwrite:\n"
                + "\n".join(str(p) for p in existing)
            )

    results = []

    for name, params in EXPERIMENTS.items():
        tag = f"lightgbm_smote_3_{name}"

        print("\n" + "=" * 65)
        print("EXPERIMENT:", name)
        print("=" * 65)

        model = lgb.LGBMClassifier(**params)

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

        metrics = {
            "experiment": name,
            "sampling_strategy": SMOTE_RATIO,
            "parameters": params,
            "best_iteration": int(model.best_iteration_),
            "validation_average_precision": float(
                average_precision_score(y_val, y_prob)
            ),
            "validation_roc_auc": float(
                roc_auc_score(y_val, y_prob)
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
        print("PR-AUC:", round(
            metrics["validation_average_precision"], 6
        ))
        print("ROC-AUC:", round(
            metrics["validation_roc_auc"], 6
        ))

    print("\n" + "=" * 65)
    print("EXPERIMENT SUMMARY")
    print("=" * 65)

    for result in results:
        print(
            f"{result['experiment']}: "
            f"PR-AUC={result['validation_average_precision']:.6f}, "
            f"ROC-AUC={result['validation_roc_auc']:.6f}, "
            f"Best iteration={result['best_iteration']}"
        )

    print("\nAll experiments completed.")
    print("Existing candidate artifacts were preserved.")


if __name__ == "__main__":
    main()
