
from pathlib import Path
import json
import joblib

BASE = Path(__file__).resolve().parents[1]

directories = [
    BASE / "processed_data" / "preprocessing_artifacts",
    BASE / "artifacts" / "preprocessing",
]

print("\n========== PREPROCESSING ARTIFACT AUDIT ==========")

for directory in directories:
    print(f"\nDirectory: {directory.relative_to(BASE)}")

    if not directory.exists():
        print("Directory does not exist.")
        continue

    for path in sorted(directory.glob("*")):
        if path.is_file():
            print(f"\nFile: {path.name}")

            if path.suffix == ".joblib":
                try:
                    obj = joblib.load(path)
                    print("Object type:", type(obj).__name__)

                    if hasattr(obj, "n_features_in_"):
                        print("Input features:", obj.n_features_in_)

                    if hasattr(obj, "n_components_"):
                        print("PCA components:", obj.n_components_)

                    if hasattr(obj, "components_"):
                        print("Components shape:", obj.components_.shape)

                    if hasattr(obj, "feature_names_in_"):
                        print("Feature names:", list(obj.feature_names_in_))

                    if hasattr(obj, "steps"):
                        print("Pipeline steps:", [
                            (name, type(step).__name__)
                            for name, step in obj.steps
                        ])

                except Exception as error:
                    print("Could not load:", error)

            elif path.suffix == ".json":
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    print(json.dumps(data, indent=2))
                except Exception as error:
                    print("Could not read:", error)

print("\n========== FEATURE CONFIGURATION ==========")

feature_file = BASE / "artifacts" / "datasets" / "feature_columns.json"

if feature_file.exists():
    with open(feature_file, "r", encoding="utf-8") as f:
        features = json.load(f)

    print("Feature count:", len(features))
    print("Features:", features)
else:
    print("feature_columns.json not found.")

print("\n========== MODEL CONFIGURATION ==========")

model_path = BASE / "artifacts" / "trained_models" / "lightgbm_smote_ratio_0_03.joblib"

if model_path.exists():
    model = joblib.load(model_path)
    print("Model:", type(model).__name__)
    print("Expected features:", model.n_features_in_)
    print("Best iteration:", model.best_iteration_)
else:
    print("Final baseline model not found.")

print("\n========== AUDIT COMPLETE ==========")
