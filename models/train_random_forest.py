
import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
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

MODEL_PATH = ROOT / "artifacts" / "trained_models" / "random_forest.joblib"

PREDICTION_PATH = (
    ROOT / "artifacts" / "validation_predictions" / "random_forest_predictions.npz"
)

METRICS_PATH = ROOT / "artifacts" / "metrics" / "random_forest_metrics.json"

THRESHOLD = 0.50
RANDOM_STATE = 42


# --------------------------------------------------
# 2. PREVENT ACCIDENTAL OVERWRITING
# --------------------------------------------------

for path in [MODEL_PATH, PREDICTION_PATH, METRICS_PATH]:
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

print(
    "Training labels:",
    np.unique(y_train_balanced, return_counts=True)
)

print(
    "Validation labels:",
    np.unique(y_val, return_counts=True)
)


# --------------------------------------------------
# 4. INITIALIZE RANDOM FOREST
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight=None,
    random_state=RANDOM_STATE,
    n_jobs=2,
)


# --------------------------------------------------
# 5. TRAIN MODEL
# --------------------------------------------------

print("\nTraining Random Forest...")
print("This model may take longer than the previous models.")

start_time = time.time()

model.fit(
    X_train_balanced,
    y_train_balanced
)

training_time = time.time() - start_time

print(f"Training completed in {training_time:.2f} seconds.")


# --------------------------------------------------
# 6. GENERATE VALIDATION PREDICTIONS
# --------------------------------------------------

print("\nGenerating validation predictions...")

y_scores = model.predict_proba(X_val_scaled)[:, 1]

y_pred = (y_scores >= THRESHOLD).astype(np.int8)


# --------------------------------------------------
# 7. CALCULATE METRICS
# --------------------------------------------------

ap = average_precision_score(y_val, y_scores)

roc_auc = roc_auc_score(y_val, y_scores)

precision = precision_score(
    y_val, y_pred, zero_division=0
)

recall = recall_score(
    y_val, y_pred, zero_division=0
)

f1 = f1_score(
    y_val, y_pred, zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    y_val,
    y_pred,
    labels=[0, 1]
).ravel()


# --------------------------------------------------
# 8. COLLECT RESULTS
# --------------------------------------------------

metrics = {
    "model": "Random Forest",
    "n_estimators": 100,
    "max_depth": 12,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "threshold": THRESHOLD,
    "average_precision": float(ap),
    "roc_auc": float(roc_auc),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),
    "training_time_seconds": float(training_time),
    "best_iteration": None,
}


# --------------------------------------------------
# 9. SAVE MODEL AND RESULTS
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
# 10. DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 55)
print("RANDOM FOREST RESULTS")
print("=" * 55)

for key, value in metrics.items():
    print(f"{key}: {value}")

print("\nConfusion Matrix:")
print(np.array([[tn, fp], [fn, tp]]))

print("\nSaved files:")
print(MODEL_PATH)
print(PREDICTION_PATH)
print(METRICS_PATH)

print("\nRandom Forest completed successfully.")


# --------------------------------------------------
# 11. RELEASE MEMORY
# --------------------------------------------------

del X_train_balanced
del y_train_balanced
del X_val_scaled
del y_val
del y_scores
del y_pred
del model
del data

gc.collect()
