
from pathlib import Path
import json
import time
import gc

import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "artifacts" / "datasets" / "training_validation_data.npz"

MODEL_DIR = BASE_DIR / "artifacts" / "trained_models"
PREDICTION_DIR = BASE_DIR / "artifacts" / "validation_predictions"
METRICS_DIR = BASE_DIR / "artifacts" / "metrics"

MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"
PREDICTION_PATH = PREDICTION_DIR / "logistic_regression_predictions.npz"
METRICS_PATH = METRICS_DIR / "logistic_regression_metrics.json"

RANDOM_STATE = 42
THRESHOLD = 0.50


# ============================================================
# 2. CHECK FILES AND PREVENT OVERWRITING
# ============================================================

for directory in [MODEL_DIR, PREDICTION_DIR, METRICS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

for path in [MODEL_PATH, PREDICTION_PATH, METRICS_PATH]:
    if path.exists():
        raise FileExistsError(
            f"Output already exists: {path}\n"
            "Inspect or back it up before rerunning."
        )

if not DATA_PATH.exists():
    raise FileNotFoundError(DATA_PATH)


# ============================================================
# 3. LOAD PREPARED DATA
# ============================================================

print("\nLoading prepared datasets...")

data = np.load(DATA_PATH)

X_train = data["X_train_balanced"]
y_train = data["y_train_balanced"]

X_val = data["X_val_scaled"]
y_val = data["y_val"]

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)

print("Training labels:", np.unique(y_train, return_counts=True))
print("Validation labels:", np.unique(y_val, return_counts=True))


# ============================================================
# 4. INITIALISE MODEL
# ============================================================

model = LogisticRegression(
    solver="lbfgs",
    C=0.1,
    max_iter=300,
    class_weight=None,
    random_state=RANDOM_STATE
)


# ============================================================
# 5. TRAIN MODEL
# ============================================================

print("\nTraining Logistic Regression...")

start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print(f"Training completed in {training_time:.2f} seconds.")


# ============================================================
# 6. VALIDATION PREDICTIONS
# ============================================================

print("\nGenerating validation predictions...")

y_scores = model.predict_proba(X_val)[:, 1]
y_pred = (y_scores >= THRESHOLD).astype(np.int8)


# ============================================================
# 7. EVALUATION
# ============================================================

average_precision = average_precision_score(y_val, y_scores)
roc_auc = roc_auc_score(y_val, y_scores)

precision = precision_score(y_val, y_pred, zero_division=0)
recall = recall_score(y_val, y_pred, zero_division=0)
f1 = f1_score(y_val, y_pred, zero_division=0)

cm = confusion_matrix(y_val, y_pred, labels=[0, 1])

tn, fp, fn, tp = cm.ravel()

metrics = {
    "model": "Logistic Regression",
    "average_precision": float(average_precision),
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
    "best_iteration": None
}


# ============================================================
# 8. SAVE MODEL, PREDICTIONS AND METRICS
# ============================================================

print("\nSaving model and results...")

joblib.dump(model, MODEL_PATH)

np.savez_compressed(
    PREDICTION_PATH,
    y_true=y_val,
    y_scores=y_scores,
    y_pred=y_pred
)

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 55)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 55)

for key, value in metrics.items():
    print(f"{key}: {value}")

print("\nConfusion Matrix:")
print(cm)

print("\nSaved files:")
print(MODEL_PATH)
print(PREDICTION_PATH)
print(METRICS_PATH)

print("\nLogistic Regression completed successfully.")

del data, X_train, y_train, X_val, y_val
del y_scores, y_pred, model

gc.collect()
