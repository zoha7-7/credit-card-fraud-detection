# Credit Card Fraud Analysis — Member 1

# Overview

'member1.py' performs an initial exploratory data analysis (EDA) of the
credit-card transaction dataset. It checks the dataset's structure and quality,
summarizes the `Class` labels, compares transaction amounts by class, and
visualizes correlations between numeric features.

# Requirements

- Python 3
- pandas
- numpy
- matplotlib
- seaborn

Install the required packages from PowerShell:

```powershell
python -m pip install pandas numpy matplotlib seaborn
```

# Dataset setup

Place `creditcard.csv` in the same folder as `member1.py`. The CSV must include
the `Amount` and `Class` columns. The script checks that these columns exist
before performing its analysis.

# Run the analysis

Open PowerShell in this project folder and run:

```powershell
python .\member1.py
```

If `python` is not recognized, try:

```powershell
py .\member1.py
```

# Outputs

The script creates these folders if they do not already exist:

- `reports/`
  - `dataset_overview.csv` — dataset dimensions, duplicate count, and total missing values.
  - `missing_values.csv` — missing-value counts and percentages by column.
  - `numeric_summary.csv` — descriptive statistics for numeric columns.
  - `class_distribution.csv` — transaction counts and percentages by class.
- `graphs/`
  - `class_distribution.png` — transaction counts for each class.
  - `amount_by_class.png` — transaction amounts by class, displayed on a `log(1 + Amount)` scale.
  - `correlation_heatmap.png` — correlations between numeric features.

The `Class` column uses `0` for legitimate transactions and `1` for fraudulent
transactions.
