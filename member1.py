from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Dataset collection and exploratory data analysis (EDA)
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "creditcard.csv"
REPORT_DIR = BASE_DIR / "reports"
GRAPH_DIR = BASE_DIR / "graphs"

REPORT_DIR.mkdir(exist_ok=True)
GRAPH_DIR.mkdir(exist_ok=True)

if not DATA_PATH.is_file():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place creditcard.csv in the same folder as member1.py."
    )

# 1. Load and validate the dataset
df = pd.read_csv(DATA_PATH)
required_columns = {"Amount", "Class"}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

print("Dataset loaded successfully.")
print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
print("\nFirst five rows:")
print(df.head())

# 2. Dataset structure and quality
overview = pd.DataFrame(
    {
        "rows": [df.shape[0]],
        "columns": [df.shape[1]],
        "duplicate_rows": [int(df.duplicated().sum())],
        "total_missing_values": [int(df.isna().sum().sum())],
    }
)
overview.to_csv(REPORT_DIR / "dataset_overview.csv", index=False)

missing_report = (
    df.isna()
    .sum()
    .rename("missing_values")
    .to_frame()
    .assign(missing_percent=lambda result: result["missing_values"] / len(df) * 100)
)
missing_report.to_csv(REPORT_DIR / "missing_values.csv")

numeric_summary = df.describe().T
numeric_summary.to_csv(REPORT_DIR / "numeric_summary.csv")

print("\nDataset information:")
df.info()
print(f"\nDuplicate rows: {df.duplicated().sum():,}")
print(f"Missing values: {df.isna().sum().sum():,}")
print("\nStatistical summary saved to reports/numeric_summary.csv")

# 3. Examine the target label and class imbalance
class_counts = df["Class"].value_counts().sort_index()
class_percentages = df["Class"].value_counts(normalize=True).sort_index() * 100
class_report = pd.DataFrame(
    {
        "transactions": class_counts,
        "percentage": class_percentages,
    }
)
class_report.index.name = "Class (0=legitimate, 1=fraud)"
class_report.to_csv(REPORT_DIR / "class_distribution.csv")

print("\nClass distribution (0=legitimate, 1=fraud):")
print(class_report)

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(x=class_counts.index.astype(str), y=class_counts.values, ax=ax)
ax.set(title="Transaction class counts", xlabel="Class (0=legitimate, 1=fraud)", ylabel="Transactions")
fig.tight_layout()
fig.savefig(GRAPH_DIR / "class_distribution.png", dpi=150)
plt.close(fig)

# 4. Compare transaction amounts for legitimate and fraudulent transactions
amount_plot_data = df[["Amount", "Class"]].copy()
amount_plot_data["Log amount (log1p)"] = np.log1p(
    amount_plot_data["Amount"].clip(lower=0)
)
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=amount_plot_data, x="Class", y="Log amount (log1p)", ax=ax)
ax.set(
    title="Transaction amount by class (log1p scale)",
    xlabel="Class (0=legitimate, 1=fraud)",
    ylabel="log(1 + Amount)",
)
fig.tight_layout()
fig.savefig(GRAPH_DIR / "amount_by_class.png", dpi=150)
plt.close(fig)

# 5. Explore numeric feature relationships
correlation = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(16, 13))
sns.heatmap(correlation, cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Feature correlation heatmap")
fig.tight_layout()
fig.savefig(GRAPH_DIR / "correlation_heatmap.png", dpi=150)
plt.close(fig)

print("\nEDA complete.")
print(f"Reports saved in: {REPORT_DIR}")
print(f"Graphs saved in: {GRAPH_DIR}")
