
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "external_evaluation" / "external_config.json"


def prepare_external_data(csv_path):

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    def project_path(key):
        return BASE_DIR / config[key]

    with open(project_path("feature_file"), "r", encoding="utf-8") as f:
        features = json.load(f)

    data = pd.read_csv(csv_path)

    target = config["target_column"]

    # Separate optional ground-truth labels.
    y_true = None
    if target in data.columns:
        y_true = data[target].copy()
        data = data.drop(columns=[target])

        if not y_true.isin([0, 1]).all():
            raise ValueError("Target must contain only 0 and 1.")

    # Strict feature compatibility.
    missing = [col for col in features if col not in data.columns]

    if missing:
        raise ValueError(
            f"Incompatible dataset. Missing features: {missing}"
        )

    X = data[features].copy()

    # Ensure numerical input.
    for column in features:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    # Reject columns that contain no usable values.
    empty = X.columns[X.isna().all()].tolist()
    if empty:
        raise ValueError(f"Entirely empty features: {empty}")

    # Reuse original fitted transformations.
    imputer = joblib.load(project_path("imputer"))
    scaler = joblib.load(project_path("scaler"))

    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)

    if not np.isfinite(X_scaled).all():
        raise ValueError("Non-finite values after preprocessing.")

    X_scaled = X_scaled.astype(np.float32)

    if X_scaled.shape[1] != len(features):
        raise ValueError("Unexpected feature count.")

    print("\nExternal dataset prepared successfully.")
    print("Transactions:", len(X_scaled))
    print("Features:", X_scaled.shape[1])
    print("Ground truth available:", y_true is not None)

    return X_scaled, y_true


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python prepare_external_data.py path_to_csv"
        )

    prepare_external_data(sys.argv[1])
