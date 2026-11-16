from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "creditcard.csv"

df = pd.read_csv(DATA_PATH)

# Display first 5 rows
print("\nFIRST FIVE ROWS:")
print(df.head())

# Display last 5 rows
print("\nLAST FIVE ROWS:")
print(df.tail())

# Display all column names
print("\nCOLUMN NAMES:")
print(df.columns.tolist())

# Dataset dimensions
print("\nDATASET SHAPE:")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# Dataset information
print("\nDATASET INFORMATION:")
df.info()

# Statistical summary
print("\nSTATISTICAL SUMMARY:")
print(df.describe())