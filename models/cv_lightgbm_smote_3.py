
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


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "creditcard.csv"
METADATA_PATH = (
    ROOT / "processed_data/preprocessing_artifacts/preprocessing_metadata.json"
)

OUTPUT_DIR = ROOT / "artifacts/cross_validation"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

SMOTE_RATIO = 0.03
K_NEIGHBORS = 5

TARGET = "Class"


# ============================================================
# 2. MODEL CONFIGURATIONS
# ============================================================

BASE_PARAMS = {
    "n_estimators": 200,
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
    },
    "regularized_min_child_100": {
        **BASE_PARAMS,
        "min_child_samples": 100,
    },
}


# ============================================================
# 3. MAIN
# ============================================================

def main():

    if not RAW_PATH.exists():
        raise FileNotFoundError(RAW_PATH)

    if not METADATA_PATH.exists():
        raise FileNotFoundError(METADATA_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "lightgbm_5fold_results.json"

    if output_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            "Refusing to overwrite."
        )

    # --------------------------------------------------------
    # Load original feature metadata and dataset
    # --------------------------------------------------------

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_columns = metadata["feature_columns"]

    if metadata["random_state"] != RANDOM_STATE:
        raise ValueError("Random-state mismatch.")

    if metadata["test_size"] != TEST_SIZE:
        raise ValueError("Test-size mismatch.")

    print("Loading original dataset...")

    df = pd.read_csv(RAW_PATH)

    df = df.dropna(subset=[TARGET])
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    X_all = df[feature_columns]
    y_all = df[TARGET].to_numpy(dtype=np.int8)

    print("Cleaned dataset:", X_all.shape)
    print("Class distribution:", np.bincount(y_all))

    # --------------------------------------------------------
    # Reconstruct original outer training portion
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Stratified five-fold cross-validation
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = {}

    for model_name, params in MODELS.items():

        print("\n" + "=" * 65)
        print("MODEL:", model_name)
        print("=" * 65)

        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X_outer, y_outer), start=1
        ):

            print("\n" + "-" * 55)
            print(f"FOLD {fold}/{N_SPLITS}")
            print("-" * 55)

            X_train_raw = X_outer.iloc[train_idx]
            X_val_raw = X_outer.iloc[val_idx]

            y_train = y_outer[train_idx]
            y_val = y_outer[val_idx]

            print("Training rows:", len(train_idx))
            print("Validation rows:", len(val_idx))
            print("Training classes:", np.bincount(y_train))
            print("Validation classes:", np.bincount(y_val))

            # ------------------------------------------------
            # Fit preprocessing ONLY on fold training data
            # ------------------------------------------------

            imputer = SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            )

            scaler = RobustScaler()

            X_train = imputer.fit_transform(X_train_raw)
            X_val = imputer.transform(X_val_raw)

            X_train = scaler.fit_transform(X_train).astype(np.float32)
            X_val = scaler.transform(X_val).astype(np.float32)

            # ------------------------------------------------
            # Apply SMOTE ONLY to fold training data
            # ------------------------------------------------

            smote = SMOTE(
                sampling_strategy=SMOTE_RATIO,
                random_state=RANDOM_STATE,
                k_neighbors=K_NEIGHBORS,
            )

            X_train_resampled, y_train_resampled = (
                smote.fit_resample(X_train, y_train)
            )

            print("After SMOTE:", X_train_resampled.shape)
            print("Resampled classes:", np.bincount(y_train_resampled))

            # ------------------------------------------------
            # Train LightGBM
            # ------------------------------------------------

            model = lgb.LGBMClassifier(**params)

            model.fit(
                X_train_resampled,
                y_train_resampled,
                eval_set=[(X_val, y_val)],
                eval_metric="average_precision",
                callbacks=[
                    lgb.early_stopping(25, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )

            # ------------------------------------------------
            # Evaluate fold
            # ------------------------------------------------

            y_prob = model.predict_proba(X_val)[:, 1]

            fold_metrics = {
                "fold": fold,
                "average_precision": float(
                    average_precision_score(y_val, y_prob)
                ),
                "roc_auc": float(
                    roc_auc_score(y_val, y_prob)
                ),
                "best_iteration": int(model.best_iteration_),
                "training_rows": int(len(train_idx)),
                "validation_rows": int(len(val_idx)),
                "validation_fraud_count": int(np.sum(y_val == 1)),
            }

            fold_results.append(fold_metrics)

            print("PR-AUC:", round(fold_metrics["average_precision"], 6))
            print("ROC-AUC:", round(fold_metrics["roc_auc"], 6))
            print("Best iteration:", fold_metrics["best_iteration"])

            del (
                X_train_raw,
                X_val_raw,
                X_train,
                X_val,
                X_train_resampled,
                y_train_resampled,
                model,
                imputer,
                scaler,
                smote,
                y_prob,
            )

            gc.collect()

        # ----------------------------------------------------
        # Calculate mean and standard deviation
        # ----------------------------------------------------

        pr_scores = [
            x["average_precision"] for x in fold_results
        ]

        roc_scores = [
            x["roc_auc"] for x in fold_results
        ]

        results[model_name] = {
            "fold_results": fold_results,
            "mean_pr_auc": float(np.mean(pr_scores)),
            "std_pr_auc": float(np.std(pr_scores, ddof=1)),
            "mean_roc_auc": float(np.mean(roc_scores)),
            "std_roc_auc": float(np.std(roc_scores, ddof=1)),
        }

        print("\nMODEL SUMMARY:", model_name)
        print(
            "Mean PR-AUC:",
            round(results[model_name]["mean_pr_auc"], 6),
        )
        print(
            "PR-AUC standard deviation:",
            round(results[model_name]["std_pr_auc"], 6),
        )
        print(
            "Mean ROC-AUC:",
            round(results[model_name]["mean_roc_auc"], 6),
        )
        print(
            "ROC-AUC standard deviation:",
            round(results[model_name]["std_roc_auc"], 6),
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output = {
        "experiment": "Stratified 5-Fold Cross-Validation",
        "training_data": "Original outer training portion only",
        "smote_ratio": SMOTE_RATIO,
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "preprocessing": "Imputer and scaler fitted independently per fold",
        "smote_applied_to": "Fold training data only",
        "models": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print("\n" + "=" * 65)
    print("CROSS-VALIDATION COMPLETED")
    print("=" * 65)

    for name, result in results.items():
        print(
            f"{name}: "
            f"PR-AUC={result['mean_pr_auc']:.6f} "
            f"+/- {result['std_pr_auc']:.6f}"
        )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
