
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import json


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DATA = BASE_DIR / "creditcard.csv"

DATA_DIR = BASE_DIR / "artifacts" / "datasets"
PREPROCESS_DIR = BASE_DIR / "artifacts" / "preprocessing"

SPLIT_FILE = DATA_DIR / "split_indices.npz"
OLD_TEST_FILE = DATA_DIR / "test_data.npz"
FEATURE_FILE = DATA_DIR / "feature_columns.json"

IMPUTER_FILE = PREPROCESS_DIR / "validation_imputer.joblib"
SCALER_FILE = PREPROCESS_DIR / "validation_scaler.joblib"

OUTPUT_FILE = DATA_DIR / "test_data_model_scaled.npz"


# ============================================================
# SAFETY CHECKS
# ============================================================

required_files = [
    RAW_DATA,
    SPLIT_FILE,
    OLD_TEST_FILE,
    FEATURE_FILE,
    IMPUTER_FILE,
    SCALER_FILE
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(path)

if OUTPUT_FILE.exists():
    raise FileExistsError(
        f"{OUTPUT_FILE} already exists. "
        "Inspect it before rerunning."
    )


# ============================================================
# LOAD DATA
# ============================================================

print("Loading original dataset...")

df = pd.read_csv(RAW_DATA)

df = df.dropna(subset=["Class"])
df = df.drop_duplicates()
df = df.reset_index(drop=True)

with open(FEATURE_FILE, "r") as f:
    feature_columns = json.load(f)

split = np.load(SPLIT_FILE)

test_indices = split["test_indices"]

old_test = np.load(OLD_TEST_FILE)

old_y_test = old_test["y_test"]

if len(df) != 283726:
    raise ValueError("Cleaned dataset row count mismatch.")

if len(test_indices) != len(old_y_test):
    raise ValueError("Test index count mismatch.")


# ============================================================
# RECONSTRUCT ORIGINAL TEST DATA
# ============================================================

print("Reconstructing raw test transactions...")

X_test_raw = df.iloc[test_indices][feature_columns]

y_test = df.iloc[test_indices]["Class"].to_numpy(
    dtype=np.int8
)

if not np.array_equal(y_test, old_y_test):
    raise ValueError("Test labels do not match existing export.")


# ============================================================
# LOAD TRAINING PREPROCESSING OBJECTS
# ============================================================

print("Loading training preprocessing objects...")

imputer = joblib.load(IMPUTER_FILE)
scaler = joblib.load(SCALER_FILE)


# ============================================================
# TRANSFORM TEST DATA
# ============================================================

print("Transforming test data...")

X_test_scaled = scaler.transform(
    imputer.transform(X_test_raw)
).astype(np.float32)

if not np.isfinite(X_test_scaled).all():
    raise ValueError("Non-finite values found in test features.")

if X_test_scaled.shape != (56746, 30):
    raise ValueError("Unexpected test feature shape.")


# ============================================================
# SAVE CORRECTED TEST DATA
# ============================================================

np.savez_compressed(
    OUTPUT_FILE,
    X_test=X_test_scaled,
    y_test=y_test
)

print("\n" + "=" * 55)
print("TEST PREPROCESSING COMPLETED")
print("=" * 55)

print("Test shape:", X_test_scaled.shape)
print("Labels:", y_test.shape)
print("Fraud cases:", int(y_test.sum()))
print("Legitimate cases:", int((y_test == 0).sum()))

print("Saved:", OUTPUT_FILE)
