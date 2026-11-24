
from pathlib import Path
import json
import numpy as np
import lightgbm as lgb
import joblib

from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "artifacts/datasets/original_training_data_scaled.npz"
VAL_PATH = ROOT / "artifacts/datasets/training_validation_data.npz"

MODEL_DIR = ROOT / "artifacts/trained_models"
PRED_DIR = ROOT / "artifacts/validation_predictions"
METRIC_DIR = ROOT / "artifacts/metrics"

RATIOS = [0.05, 0.10, 0.20]

train = np.load(TRAIN_PATH)
val = np.load(VAL_PATH)

X_train = train["X_train_scaled"]
y_train = train["y_train"]
X_val = val["X_val_scaled"]
y_val = val["y_val"]

assert X_train.shape == (181584, 30)
assert np.sum(y_train == 0) == 181282
assert np.sum(y_train == 1) == 302
assert len(X_val) == 45396
assert np.sum(y_val == 1) == 76

results = []

for ratio in RATIOS:

    ratio_name = str(int(ratio * 100))
    name = f"lightgbm_partial_smote_{ratio_name}"

    model_path = MODEL_DIR / f"{name}.joblib"
    pred_path = PRED_DIR / f"{name}_predictions.npz"
    metric_path = METRIC_DIR / f"{name}_metrics.json"

    for path in [model_path, pred_path, metric_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")

    print("\n" + "=" * 65)
    print(f"PARTIAL SMOTE EXPERIMENT: {ratio:.0%}")
    print("=" * 65)

    smote = SMOTE(
        sampling_strategy=ratio,
        random_state=42,
        k_neighbors=5,
    )

    X_resampled, y_resampled = smote.fit_resample(
        X_train, y_train
    )

    print("Original training shape:", X_train.shape)
    print("Resampled training shape:", X_resampled.shape)
    print(
        "Resampled classes:",
        np.unique(y_resampled, return_counts=True)
    )

    model = lgb.LGBMClassifier(
        n_estimators=200,
        num_leaves=15,
        max_depth=5,
        learning_rate=0.05,
        min_child_samples=50,
        max_bin=127,
        random_state=42,
        n_jobs=2,
        verbosity=-1,
    )

    print("\nTraining LightGBM...")

    model.fit(
        X_resampled,
        y_resampled,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(stopping_rounds=25, verbose=True),
            lgb.log_evaluation(period=0),
        ],
    )

    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    metrics = {
        "experiment": name,
        "sampling_strategy": ratio,
        "original_training_rows": int(len(y_train)),
        "resampled_training_rows": int(len(y_resampled)),
        "resampled_fraud_count": int(np.sum(y_resampled == 1)),
        "best_iteration": int(model.best_iteration_),
        "validation_average_precision": float(
            average_precision_score(y_val, probabilities)
        ),
        "validation_roc_auc": float(
            roc_auc_score(y_val, probabilities)
        ),
        "precision_at_0.50": float(
            precision_score(y_val, predictions, zero_division=0)
        ),
        "recall_at_0.50": float(
            recall_score(y_val, predictions, zero_division=0)
        ),
        "f1_at_0.50": float(
            f1_score(y_val, predictions, zero_division=0)
        ),
        "true_positives": int(
            np.sum((y_val == 1) & (predictions == 1))
        ),
        "false_positives": int(
            np.sum((y_val == 0) & (predictions == 1))
        ),
        "false_negatives": int(
            np.sum((y_val == 1) & (predictions == 0))
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)

    np.savez_compressed(
        pred_path,
        y_true=y_val,
        y_prob=probabilities,
    )

    with open(metric_path, "w") as f:
        json.dump(metrics, f, indent=4)

    results.append(metrics)

    print("\nRESULTS")
    print("Best iteration:", metrics["best_iteration"])
    print("PR-AUC:", round(metrics["validation_average_precision"], 6))
    print("ROC-AUC:", round(metrics["validation_roc_auc"], 6))
    print("Precision:", round(metrics["precision_at_0.50"], 6))
    print("Recall:", round(metrics["recall_at_0.50"], 6))
    print("F1:", round(metrics["f1_at_0.50"], 6))

print("\n" + "=" * 90)
print("PARTIAL SMOTE COMPARISON")
print("=" * 90)

print(
    f"{'Ratio':<10}{'Fraud rows':<14}{'Iteration':<12}"
    f"{'PR-AUC':<14}{'ROC-AUC':<14}{'Precision':<14}"
    f"{'Recall':<12}{'F1':<12}"
)

for r in results:
    print(
        f"{r['sampling_strategy']:<10.0%}"
        f"{r['resampled_fraud_count']:<14}"
        f"{r['best_iteration']:<12}"
        f"{r['validation_average_precision']:<14.6f}"
        f"{r['validation_roc_auc']:<14.6f}"
        f"{r['precision_at_0.50']:<14.6f}"
        f"{r['recall_at_0.50']:<12.6f}"
        f"{r['f1_at_0.50']:<12.6f}"
    )

print("\nAll partial SMOTE experiments completed.")
