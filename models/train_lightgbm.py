
import gc
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "artifacts" / "datasets" / "training_validation_data.npz"

MODEL_PATH = ROOT / "artifacts" / "trained_models" / "lightgbm.joblib"
PREDICTION_PATH = (
    ROOT / "artifacts" / "validation_predictions" / "lightgbm_predictions.npz"
)
METRICS_PATH = ROOT / "artifacts" / "metrics" / "lightgbm_metrics.json"

THRESHOLD = 0.50
RANDOM_STATE = 42


# --------------------------------------------------
# 2. PREVENT ACCIDENTAL OVERWRITING
# --------------------------------------------------

outputs = [MODEL_PATH, PREDICTION_PATH, METRICS_PATH]

for path in outputs:
    if path.exists():
        raise FileExistsError(
            f"Output already exists. Move or rename it before running again:\n{path}"
        )


# --------------------------------------------------
# 3. LOAD PREPARED DATA
# --------------------------------------------------

print("\nLoading prepared datasets...")

data = np.load(DATA_PATH)

X_train_balanced = data["X_train_balanced"]
y_train_balanced = data["y_train_balanced"]
X_val_scaled = data["X_val_scaled"]
y_val = data["y_val"]

data.close()

print("Training shape:", X_train_balanced.shape)
print("Validation shape:", X_val_scaled.shape)
print("Training labels:", np.unique(y_train_balanced, return_counts=True))
print("Validation labels:", np.unique(y_val, return_counts=True))


# --------------------------------------------------
# 4. INITIALIZE LIGHTGBM
# --------------------------------------------------

model = lgb.LGBMClassifier(
    n_estimators=200,
    num_leaves=15,
    max_depth=5,
    learning_rate=0.05,
    min_child_samples=50,
    max_bin=127,
    random_state=RANDOM_STATE,
    n_jobs=2,
    verbosity=-1,
)


# --------------------------------------------------
# 5. TRAIN MODEL WITH EARLY STOPPING
# --------------------------------------------------

print("\nTraining LightGBM...")

start_time = time.time()

model.fit(
    X_train_balanced,
    y_train_balanced,
    eval_set=[(X_val_scaled, y_val)],
    eval_metric="average_precision",
    callbacks=[
        lgb.early_stopping(stopping_rounds=25, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)

training_time = time.time() - start_time

print(f"Training completed in {training_time:.2f} seconds.")
print("Best iteration:", model.best_iteration_)


# --------------------------------------------------
# 6. GENERATE VALIDATION PREDICTIONS
# --------------------------------------------------

print("\nGenerating validation predictions...")

y_scores = model.predict_proba(X_val_scaled)[:, 1]
y_pred = (y_scores >= THRESHOLD).astype(np.int8)

ap = average_precision_score(y_val, y_scores)
roc_auc = roc_auc_score(y_val, y_scores)

precision = precision_score(y_val, y_pred, zero_division=0)
recall = recall_score(y_val, y_pred, zero_division=0)
f1 = f1_score(y_val, y_pred, zero_division=0)

tn, fp, fn, tp = confusion_matrix(
    y_val, y_pred, labels=[0, 1]
).ravel()


# --------------------------------------------------
# 7. COLLECT METRICS
# --------------------------------------------------

metrics = {
    "model": "LightGBM",
    "average_precision": float(ap),
    "roc_auc": float(roc_auc),
    "threshold": THRESHOLD,
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),
    "training_time_seconds": float(training_time),
    "best_iteration": int(model.best_iteration_),
}


# --------------------------------------------------
# 8. SAVE MODEL AND RESULTS
# --------------------------------------------------

print("\nSaving model and results...")

joblib.dump(model, MODEL_PATH)

np.savez_compressed(
    PREDICTION_PATH,
    y_true=y_val,
    y_scores=y_scores,
    y_pred=y_pred,
)

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)


# --------------------------------------------------
# 9. DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 55)
print("LIGHTGBM RESULTS")
print("=" * 55)

for key, value in metrics.items():
    print(f"{key}: {value}")

print("\nConfusion Matrix:")
print(np.array([[tn, fp], [fn, tp]]))

print("\nSaved files:")
print(MODEL_PATH)
print(PREDICTION_PATH)
print(METRICS_PATH)

print("\nLightGBM completed successfully.")

del X_train_balanced, y_train_balanced, X_val_scaled
del y_val, y_scores, y_pred, model, data

gc.collect()
