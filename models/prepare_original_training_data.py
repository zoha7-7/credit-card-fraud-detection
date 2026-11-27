from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "creditcard.csv"
DATA_DIR = ROOT / "artifacts" / "datasets"
METADATA_PATH = ROOT / "artifacts" / "preprocessing" / "preparation_metadata.json"

OUTPUT_PATH = DATA_DIR / "original_training_data_raw.npz"

if OUTPUT_PATH.exists():
    raise FileExistsError(f"Output already exists: {OUTPUT_PATH}")

if not RAW_PATH.exists():
    raise FileNotFoundError(RAW_PATH)

if not METADATA_PATH.exists():
    raise FileNotFoundError(METADATA_PATH)

print("Loading preprocessing metadata...")
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

feature_columns = metadata["feature_columns"]

if len(feature_columns) != 30:
    raise ValueError("Expected exactly 30 ULB feature columns.")

if metadata["source_partition"] != "inner_train_indices only":
    raise ValueError("Unexpected source partition in metadata.")

print("Loading original ULB dataset...")
df = pd.read_csv(RAW_PATH)

# Reproduce the original cleaning procedure exactly.
df = (
    df.dropna(subset=["Class"])
      .drop_duplicates()
      .reset_index(drop=True)
)

required_columns = feature_columns + ["Class"]

missing_columns = set(required_columns) - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing source columns: {sorted(missing_columns)}")

print("Cleaned dataset shape:", df.shape)

splits_path = DATA_DIR / "split_indices.npz"
if not splits_path.exists():
    raise FileNotFoundError(splits_path)

with np.load(splits_path) as splits:
    train_idx = splits["inner_train_indices"].copy()
    val_idx = splits["validation_indices"].copy()
    test_idx = splits["test_indices"].copy()

# Validate indices without accessing validation/test feature values.
if len(np.unique(train_idx)) != len(train_idx):
    raise ValueError("Duplicate inner-training indices.")

if np.intersect1d(train_idx, val_idx).size:
    raise ValueError("Training and validation indices overlap.")

if np.intersect1d(train_idx, test_idx).size:
    raise ValueError("Training and test indices overlap.")

if train_idx.min() < 0 or train_idx.max() >= len(df):
    raise ValueError("Training indices are outside the cleaned dataset.")

if len(train_idx) != int(metadata["training_rows"]):
    raise ValueError("Training row count does not match metadata.")

# Extract only the original, naturally imbalanced inner-training partition.
train_df = df.iloc[train_idx]

X_train_raw = train_df[feature_columns].to_numpy(dtype=np.float64)
y_train = train_df["Class"].to_numpy(dtype=np.int8)

expected_counts = {0: 181282, 1: 302}
actual_counts = {
    int(label): int(count)
    for label, count in zip(*np.unique(y_train, return_counts=True))
}

if actual_counts != expected_counts:
    raise ValueError(
        f"Unexpected training class distribution: {actual_counts}"
    )

if not np.isfinite(X_train_raw).all():
    raise ValueError("Non-finite values found in raw training features.")

DATA_DIR.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    OUTPUT_PATH,
    X_train_raw=X_train_raw,
    y_train=y_train,
    feature_columns=np.asarray(feature_columns, dtype=str)
)

print("\nSUCCESS: Raw inner-training data extracted.")
print("Output:", OUTPUT_PATH)
print("Feature shape:", X_train_raw.shape)
print("Label shape:", y_train.shape)
print("Class counts:", actual_counts)
print("Feature order:", feature_columns)
print("Scaling applied: NO")
print("SMOTE applied: NO")
print("Validation/test feature values accessed: NO")
