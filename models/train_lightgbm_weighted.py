
from pathlib import Path
import json
import numpy as np
import joblib
import lightgbm as lgb

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

MODEL_PATH = ROOT / "artifacts/trained_models/lightgbm_weighted.joblib"
PRED_PATH = ROOT / "artifacts/validation_predictions/lightgbm_weighted_predictions.npz"
METRICS_PATH = ROOT / "artifacts/metrics/lightgbm_weighted_metrics.json"

# Protect all existing artifacts.
for path in [MODEL_PATH, PRED_PATH, METRICS_PATH]:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")

print("Loading original imbalanced training data...")
train = np.load(DATA_PATH)
X_train = train["X_train_scaled"]
y_train = train["y_train"]

print("Loading existing validation data...")
val = np.load(VAL_PATH)
X_val = val["X_val_scaled"]
y_val = val["y_val"]

# Verify data dimensions and class counts.
assert X_train.shape == (181584, 30)
assert np.sum(y_train == 0) == 181282
assert np.sum(y_train == 1) == 302
assert len(X_val) == 45396
assert np.sum(y_val == 1) == 76

scale_pos_weight = float(
    np.sum(y_train == 0) / np.sum(y_train == 1)
)

print("\nEXPERIMENT B: CLASS-WEIGHTED LIGHTGBM")
print("=" * 55)
print("Training rows:", len(y_train))
print("Legitimate:", int(np.sum(y_train == 0)))
print("Fraud:", int(np.sum(y_train == 1)))
print("Scale positive weight:", round(scale_pos_weight, 4))

model = lgb.LGBMClassifier(
    n_estimators=1000,
    num_leaves=15,
    max_depth=5,
    learning_rate=0.05,
    min_child_samples=50,
    max_bin=127,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=2,
    verbosity=-1,
)

print("\nTraining model...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="average_precision",
    callbacks=[
        lgb.early_stopping(stopping_rounds=25, verbose=True),
        lgb.log_evaluation(period=50),
    ],
)

print("\nGenerating validation predictions...")
probabilities = model.predict_proba(X_val)[:, 1]

# Preliminary metrics at 0.50, for reference only.
predictions = (probabilities >= 0.50).astype(int)

metrics = {
    "experiment": "LightGBM original imbalanced training with class weighting",
    "scale_pos_weight": scale_pos_weight,
    "best_iteration": int(model.best_iteration_),
    "validation_average_precision": float(
        average_precision_score(y_val, probabilities)
    ),
    "validation_roc_auc": float(
        roc_auc_score(y_val, probabilities)
    ),
    "threshold_0.50": {
        "precision": float(precision_score(
            y_val, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_val, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_val, predictions, zero_division=0
        )),
        "true_positives": int(np.sum((y_val == 1) & (predictions == 1))),
        "false_positives": int(np.sum((y_val == 0) & (predictions == 1))),
        "false_negatives": int(np.sum((y_val == 1) & (predictions == 0))),
    },
}

# Save to new, separate files.
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)

np.savez_compressed(
    PRED_PATH,
    y_true=y_val,
    y_prob=probabilities,
)

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print("\n" + "=" * 55)
print("EXPERIMENT B RESULTS")
print("=" * 55)
print("Best iteration:", model.best_iteration_)
print("Validation PR-AUC:", round(metrics["validation_average_precision"], 6))
print("Validation ROC-AUC:", round(metrics["validation_roc_auc"], 6))
print("Precision at 0.50:", round(metrics["threshold_0.50"]["precision"], 6))
print("Recall at 0.50:", round(metrics["threshold_0.50"]["recall"], 6))
print("F1 at 0.50:", round(metrics["threshold_0.50"]["f1"], 6))

print("\nSaved model:", MODEL_PATH)
print("Saved predictions:", PRED_PATH)
print("Saved metrics:", METRICS_PATH)
