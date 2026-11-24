
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "creditcard.csv"
DATA_DIR = ROOT / "artifacts" / "datasets"
PREPROCESS_DIR = ROOT / "artifacts" / "preprocessing"

OUTPUT_PATH = DATA_DIR / "original_training_data_scaled.npz"

# Never overwrite an existing artifact.
if OUTPUT_PATH.exists():
    raise FileExistsError(f"Output already exists: {OUTPUT_PATH}")

print("Loading original dataset...")
df = pd.read_csv(RAW_PATH)

# Reproduce the exact cleaning used by prepare_data.py.
df = (
    df.dropna(subset=["Class"])
      .drop_duplicates()
      .reset_index(drop=True)
)

with open(DATA_DIR / "feature_columns.json", "r") as f:
    feature_columns = json.load(f)

splits = np.load(DATA_DIR / "split_indices.npz")
train_idx = splits["inner_train_indices"]
val_idx = splits["validation_indices"]

# Extract the exact original, naturally imbalanced rows.
train_df = df.iloc[train_idx]
val_df = df.iloc[val_idx]

X_train_raw = train_df[feature_columns]
y_train = train_df["Class"].to_numpy(dtype=np.int8)

X_val_raw = val_df[feature_columns]
y_val = val_df["Class"].to_numpy(dtype=np.int8)

# Verify expected split sizes and class counts.
assert len(train_df) == 181584
assert np.sum(y_train == 0) == 181282
assert np.sum(y_train == 1) == 302

assert len(val_df) == 45396
assert np.sum(y_val == 0) == 45320
assert np.sum(y_val == 1) == 76

print("Training rows:", len(train_df))
print("Training classes:", np.unique(y_train, return_counts=True))

# Load the existing training-fitted preprocessing objects.
imputer = joblib.load(PREPROCESS_DIR / "validation_imputer.joblib")
scaler = joblib.load(PREPROCESS_DIR / "validation_scaler.joblib")

X_train_scaled = scaler.transform(
    imputer.transform(X_train_raw)
).astype(np.float32)

# Verify validation transformation against the existing artifact.
X_val_scaled_check = scaler.transform(
    imputer.transform(X_val_raw)
).astype(np.float32)

existing = np.load(DATA_DIR / "training_validation_data.npz")
X_val_existing = existing["X_val_scaled"]

assert np.array_equal(y_val, existing["y_val"])

max_diff = float(
    np.max(np.abs(X_val_scaled_check - X_val_existing))
)

print("Validation maximum difference:", max_diff)

if not np.allclose(
    X_val_scaled_check,
    X_val_existing,
    rtol=0,
    atol=1e-6
):
    raise ValueError(
        "Validation preprocessing mismatch. Output was not saved."
    )

# Save only the newly extracted training data.
np.savez_compressed(
    OUTPUT_PATH,
    X_train_scaled=X_train_scaled,
    y_train=y_train
)

print("\nSUCCESS: Original imbalanced training data saved.")
print("Output:", OUTPUT_PATH)
print("Training feature shape:", X_train_scaled.shape)
print("Training label shape:", y_train.shape)
