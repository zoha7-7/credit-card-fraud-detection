
from pathlib import Path
import json
import gc

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = BASE_DIR / "creditcard.csv"

PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
SOURCE_ARTIFACT_DIR = PROCESSED_DATA_DIR / "preprocessing_artifacts"

TEST_EXPORT_PATH = PROCESSED_DATA_DIR / "test_transactions.csv"

OUTPUT_DATA_DIR = BASE_DIR / "artifacts" / "datasets"
OUTPUT_PREPROCESSING_DIR = BASE_DIR / "artifacts" / "preprocessing"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

TARGET = "Class"


# ============================================================
# 2. VERIFY INPUT FILES
# ============================================================

required_files = [
    RAW_DATA_PATH,
    SOURCE_ARTIFACT_DIR / "imputer.joblib",
    SOURCE_ARTIFACT_DIR / "scaler.joblib",
    SOURCE_ARTIFACT_DIR / "preprocessing_metadata.json",
    TEST_EXPORT_PATH,
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(f"Required file missing: {file_path}")

OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)

output_files = [
    OUTPUT_DATA_DIR / "training_validation_data.npz",
    OUTPUT_DATA_DIR / "test_data.npz",
    OUTPUT_DATA_DIR / "split_indices.npz",
    OUTPUT_DATA_DIR / "feature_columns.json",
    OUTPUT_PREPROCESSING_DIR / "validation_imputer.joblib",
    OUTPUT_PREPROCESSING_DIR / "validation_scaler.joblib",
    OUTPUT_PREPROCESSING_DIR / "preparation_metadata.json",
]

existing_outputs = [p for p in output_files if p.exists()]

if existing_outputs:
    raise FileExistsError(
        "Output files already exist. No files were overwritten:\n"
        + "\n".join(str(p) for p in existing_outputs)
        + "\nInspect or back up these files before rerunning."
    )


# ============================================================
# 3. LOAD METADATA AND ORIGINAL DATASET
# ============================================================

print("\n[1/8] Loading preprocessing metadata...")

with open(
    SOURCE_ARTIFACT_DIR / "preprocessing_metadata.json",
    "r",
    encoding="utf-8"
) as file:
    source_metadata = json.load(file)

feature_columns = source_metadata["feature_columns"]

if source_metadata["random_state"] != RANDOM_STATE:
    raise ValueError("Random-state mismatch with preprocessing metadata.")

if source_metadata["test_size"] != TEST_SIZE:
    raise ValueError("Test-size mismatch with preprocessing metadata.")

print("[2/8] Loading original dataset...")

df = pd.read_csv(RAW_DATA_PATH)

if TARGET not in df.columns:
    raise ValueError("Target column 'Class' is missing.")

if not set(feature_columns).issubset(df.columns):
    raise ValueError("One or more expected feature columns are missing.")

print("Original shape:", df.shape)


# ============================================================
# 4. REPRODUCE ORIGINAL CLEANING
# ============================================================

print("\n[3/8] Cleaning original dataset...")

df = df.dropna(subset=[TARGET])
df = df.drop_duplicates()
df = df.reset_index(drop=True)

if df[TARGET].nunique() != 2:
    raise ValueError("Expected exactly two target classes.")

print("Cleaned shape:", df.shape)
print("Class distribution:")
print(df[TARGET].value_counts().sort_index())


# ============================================================
# 5. RECONSTRUCT OUTER TRAIN/TEST SPLIT
# ============================================================

print("\n[4/8] Reconstructing original train/test split...")

all_indices = np.arange(len(df))
all_labels = df[TARGET].to_numpy()

outer_train_indices, test_indices = train_test_split(
    all_indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=all_labels
)

X_test_raw = df.iloc[test_indices][feature_columns]
y_test = df.iloc[test_indices][TARGET].to_numpy()

print("Outer training rows:", len(outer_train_indices))
print("Test rows:", len(test_indices))


# ============================================================
# 6. VERIFY MEMBER 2 TEST EXPORT
# ============================================================

print("\n[5/8] Verifying exported test dataset...")

member2_imputer = joblib.load(
    SOURCE_ARTIFACT_DIR / "imputer.joblib"
)

member2_scaler = joblib.load(
    SOURCE_ARTIFACT_DIR / "scaler.joblib"
)

test_export = pd.read_csv(TEST_EXPORT_PATH)

if TARGET not in test_export.columns:
    raise ValueError("Test export does not contain Class.")

if list(test_export.columns) != feature_columns + [TARGET]:
    raise ValueError("Test export column order does not match metadata.")

if len(test_export) != len(test_indices):
    raise ValueError("Test export row count does not match reconstructed split.")

if not np.array_equal(
    y_test,
    test_export[TARGET].to_numpy()
):
    raise ValueError("Test labels or row order do not match.")

member2_test_transformed = member2_scaler.transform(
    member2_imputer.transform(X_test_raw)
)

exported_test_features = test_export[feature_columns].to_numpy(
    dtype=np.float64
)

max_difference = float(
    np.max(np.abs(member2_test_transformed - exported_test_features))
)

print("Test export shape:", test_export.shape)
print("Maximum absolute difference:", max_difference)

if not np.allclose(
    member2_test_transformed,
    exported_test_features,
    rtol=0,
    atol=0
):
    raise ValueError("Test transformation verification failed.")

print("TEST EXPORT VERIFIED.")


# ============================================================
# 7. CREATE LEAKAGE-FREE INNER TRAIN/VALIDATION SPLIT
# ============================================================

print("\n[6/8] Creating inner training/validation split...")

inner_train_indices, validation_indices = train_test_split(
    outer_train_indices,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=df.iloc[outer_train_indices][TARGET].to_numpy()
)

X_inner_train_raw = df.iloc[inner_train_indices][feature_columns]
y_inner_train = df.iloc[inner_train_indices][TARGET].to_numpy()

X_validation_raw = df.iloc[validation_indices][feature_columns]
y_validation = df.iloc[validation_indices][TARGET].to_numpy()

print("Inner training shape:", X_inner_train_raw.shape)
print("Validation shape:", X_validation_raw.shape)

print("Validation class distribution:")
print(pd.Series(y_validation).value_counts().sort_index())


# ============================================================
# 8. FIT PREPROCESSING ONLY ON INNER TRAINING DATA
# ============================================================

print("\n[7/8] Fitting validation preprocessing pipeline...")

validation_imputer = SimpleImputer(
    strategy="median",
    keep_empty_features=True
)

validation_scaler = RobustScaler()

X_inner_train_imputed = validation_imputer.fit_transform(
    X_inner_train_raw
)

X_validation_imputed = validation_imputer.transform(
    X_validation_raw
)

X_inner_train_scaled = validation_scaler.fit_transform(
    X_inner_train_imputed
).astype(np.float32)

X_validation_scaled = validation_scaler.transform(
    X_validation_imputed
).astype(np.float32)

del X_inner_train_imputed, X_validation_imputed
gc.collect()


# ============================================================
# 9. APPLY SMOTE ONLY TO INNER TRAINING DATA
# ============================================================

print("\nApplying SMOTE to inner training data only...")

smote = SMOTE(random_state=RANDOM_STATE)

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_inner_train_scaled,
    y_inner_train
)

X_train_balanced = np.asarray(
    X_train_balanced,
    dtype=np.float32
)

y_train_balanced = np.asarray(
    y_train_balanced,
    dtype=np.int8
)

y_validation = np.asarray(y_validation, dtype=np.int8)

print("Balanced training shape:", X_train_balanced.shape)
print("Balanced training class counts:")
print(pd.Series(y_train_balanced).value_counts().sort_index())

print("Validation shape:", X_validation_scaled.shape)


# ============================================================
# 10. PREPARE VERIFIED TEST FEATURES
# ============================================================

X_test_export = exported_test_features.astype(np.float32)
y_test = np.asarray(y_test, dtype=np.int8)


# ============================================================
# 11. SAVE REUSABLE DATASETS AND PREPROCESSING OBJECTS
# ============================================================

print("\n[8/8] Saving datasets and preprocessing objects...")

np.savez_compressed(
    OUTPUT_DATA_DIR / "training_validation_data.npz",
    X_train_balanced=X_train_balanced,
    y_train_balanced=y_train_balanced,
    X_val_scaled=X_validation_scaled,
    y_val=y_validation
)

np.savez_compressed(
    OUTPUT_DATA_DIR / "test_data.npz",
    X_test=X_test_export,
    y_test=y_test
)

np.savez_compressed(
    OUTPUT_DATA_DIR / "split_indices.npz",
    inner_train_indices=inner_train_indices,
    validation_indices=validation_indices,
    test_indices=test_indices
)

with open(
    OUTPUT_DATA_DIR / "feature_columns.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(feature_columns, file, indent=4)

joblib.dump(
    validation_imputer,
    OUTPUT_PREPROCESSING_DIR / "validation_imputer.joblib"
)

joblib.dump(
    validation_scaler,
    OUTPUT_PREPROCESSING_DIR / "validation_scaler.joblib"
)


# ============================================================
# 12. SAVE PREPARATION METADATA
# ============================================================

preparation_metadata = {
    "random_state": RANDOM_STATE,
    "outer_test_size": TEST_SIZE,
    "inner_validation_size": VALIDATION_SIZE,
    "cleaned_dataset_rows": int(len(df)),
    "inner_training_rows_before_smote": int(len(inner_train_indices)),
    "training_rows_after_smote": int(len(X_train_balanced)),
    "validation_rows": int(len(validation_indices)),
    "test_rows": int(len(test_indices)),
    "feature_count": int(len(feature_columns)),
    "validation_imputer": "SimpleImputer(strategy=median)",
    "validation_scaler": "RobustScaler",
    "smote_random_state": RANDOM_STATE,
    "smote_applied_to": "Inner training data only",
    "test_export_max_absolute_difference": max_difference,
    "test_export_verified": True
}

with open(
    OUTPUT_PREPROCESSING_DIR / "preparation_metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(preparation_metadata, file, indent=4)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("DATA PREPARATION COMPLETED SUCCESSFULLY")
print("=" * 55)

print("Balanced training:", X_train_balanced.shape)
print("Validation:", X_validation_scaled.shape)
print("Test:", X_test_export.shape)

print("\nSaved files:")
for path in output_files:
    print(path.relative_to(BASE_DIR))

print("\nTest export verification:", max_difference)
print("Leakage control: SMOTE applied only to inner training data.")
print("\nAll preparation artifacts saved.")
