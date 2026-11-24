
from pathlib import Path
import json
import gc

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import average_precision_score, roc_auc_score

from imblearn.over_sampling import SMOTE


ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "creditcard.csv"
METADATA_PATH = (
    ROOT / "processed_data/preprocessing_artifacts/preprocessing_metadata.json"
)
OUTPUT_DIR = ROOT / "artifacts/cross_validation"

TARGET = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
SMOTE_RATIO = 0.03
K_NEIGHBORS = 5

BASE_PARAMS = {
    "num_leaves": 15,
    "max_depth": 5,
    "learning_rate": 0.05,
    "max_bin": 127,
    "random_state": 42,
    "n_jobs": 2,
    "verbosity": -1,
}

MODELS = {
    "baseline_min_child_50": {
        **BASE_PARAMS,
        "min_child_samples": 50,
        "n_estimators": 175,
    },
    "candidate_C_min_child_100": {
        **BASE_PARAMS,
        "min_child_samples": 100,
        "n_estimators": 84,
    },
}


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(RAW_PATH)

    if not METADATA_PATH.exists():
        raise FileNotFoundError(METADATA_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "final_fixed_iteration_cv.json"

    if output_path.exists():
        raise FileExistsError(
            f"Output already exists. Refusing to overwrite: {output_path}"
        )

    print("Loading original dataset...")

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_columns = metadata["feature_columns"]

    df = pd.read_csv(RAW_PATH)
    df = (
        df.dropna(subset=[TARGET])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    X_all = df[feature_columns]
    y_all = df[TARGET].to_numpy(dtype=np.int8)

    print("Cleaned dataset:", X_all.shape)
    print("Class distribution:", np.bincount(y_all))

    # Reconstruct the original outer 80% training partition.
    all_indices = np.arange(len(df))

    outer_train_indices, _ = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    X_outer = X_all.iloc[outer_train_indices].reset_index(drop=True)
    y_outer = y_all[outer_train_indices]

    print("\nOuter training shape:", X_outer.shape)
    print("Outer training classes:", np.bincount(y_outer))

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = {}

    for model_name, params in MODELS.items():

        print("\n" + "=" * 65)
        print("MODEL:", model_name)
        print("Fixed iterations:", params["n_estimators"])
        print("=" * 65)

        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X_outer, y_outer), start=1
        ):

            print(f"\nFOLD {fold}/{N_SPLITS}")

            X_train_raw = X_outer.iloc[train_idx]
            X_val_raw = X_outer.iloc[val_idx]

            y_train = y_outer[train_idx]
            y_val = y_outer[val_idx]

            # Fit preprocessing only on this fold's training data.
            imputer = SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            )

            scaler = RobustScaler()

            X_train = imputer.fit_transform(X_train_raw)
            X_val = imputer.transform(X_val_raw)

            X_train = X_train.astype(np.float32)
            X_val = X_val.astype(np.float32)

            X_train = scaler.fit_transform(X_train).astype(np.float32)
            X_val = scaler.transform(X_val).astype(np.float32)

            # SMOTE only on the fold's training data.
            smote = SMOTE(
                sampling_strategy=SMOTE_RATIO,
                random_state=RANDOM_STATE,
                k_neighbors=K_NEIGHBORS,
            )

            X_resampled, y_resampled = smote.fit_resample(
                X_train, y_train
            )

            print("Training rows:", len(train_idx))
            print("Validation rows:", len(val_idx))
            print("Validation fraud:", int(np.sum(y_val == 1)))
            print("After SMOTE:", X_resampled.shape)

            # No validation set and no early stopping.
            model = lgb.LGBMClassifier(**params)

            model.fit(X_resampled, y_resampled)

            y_prob = model.predict_proba(X_val)[:, 1]

            pr_auc = average_precision_score(y_val, y_prob)
            roc_auc = roc_auc_score(y_val, y_prob)

            fold_metrics = {
                "fold": fold,
                "pr_auc": float(pr_auc),
                "roc_auc": float(roc_auc),
                "iterations": int(params["n_estimators"]),
                "training_rows": int(len(train_idx)),
                "validation_rows": int(len(val_idx)),
                "validation_fraud_count": int(np.sum(y_val == 1)),
            }

            fold_results.append(fold_metrics)

            print(f"PR-AUC:  {pr_auc:.6f}")
            print(f"ROC-AUC: {roc_auc:.6f}")

            del model, X_resampled, y_resampled
            del X_train, X_val, y_prob
            del imputer, scaler, smote
            gc.collect()

        pr_values = [item["pr_auc"] for item in fold_results]
        roc_values = [item["roc_auc"] for item in fold_results]

        summary = {
            "mean_pr_auc": float(np.mean(pr_values)),
            "std_pr_auc": float(np.std(pr_values, ddof=1)),
            "mean_roc_auc": float(np.mean(roc_values)),
            "std_roc_auc": float(np.std(roc_values, ddof=1)),
        }

        results[model_name] = {
            "parameters": {
                key: value
                for key, value in params.items()
            },
            "fold_results": fold_results,
            "summary": summary,
        }

        print("\nMODEL SUMMARY:", model_name)
        print(
            f"Mean PR-AUC: {summary['mean_pr_auc']:.6f} "
            f"+/- {summary['std_pr_auc']:.6f}"
        )
        print(
            f"Mean ROC-AUC: {summary['mean_roc_auc']:.6f} "
            f"+/- {summary['std_roc_auc']:.6f}"
        )

    output = {
        "experiment": "final_fixed_iteration_5fold_cv",
        "smote_ratio": SMOTE_RATIO,
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "early_stopping": False,
        "models": results,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print("\n" + "=" * 65)
    print("FINAL FIXED-ITERATION EXPERIMENT COMPLETED")
    print("=" * 65)

    for name, result in results.items():
        summary = result["summary"]
        print(
            f"{name}: "
            f"PR-AUC={summary['mean_pr_auc']:.6f} "
            f"+/- {summary['std_pr_auc']:.6f}"
        )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
