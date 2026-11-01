
from pathlib import Path
import numpy as np
import pandas as pd

# Project paths
project_dir = Path(r"F:\Projects\SentinelX")
data_dir = project_dir / "Datasets"

input_file = data_dir / "Wednesday-14-02-2018_cleaned.csv"
output_file = data_dir / "Wednesday-14-02-2018_prepared.csv"

print("Loading cleaned dataset...")
df = pd.read_csv(input_file, low_memory=False)

# Convert timestamps to datetime
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce"
)

# Remove records with invalid timestamps
df = df.dropna(subset=["Timestamp"]).copy()

# Sort records by time for temporal processing
df = df.sort_values("Timestamp", kind="stable").reset_index(drop=True)

# Identify numeric feature columns, excluding the target label
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

# Identify constant numeric columns
constant_columns = [
    col for col in numeric_columns
    if df[col].nunique(dropna=False) <= 1
]

print("\nConstant numeric features to exclude:")
print(constant_columns)

# Remove constant features from model inputs
df = df.drop(columns=constant_columns)

# Safety check: replace any remaining infinite numeric values
numeric_columns = df.select_dtypes(include=np.number).columns

df[numeric_columns] = df[numeric_columns].replace(
    [np.inf, -np.inf],
    np.nan
)

# Save prepared dataset without overwriting earlier files
df.to_csv(output_file, index=False)

print("\n" + "=" * 50)
print("FEATURE PREPARATION SUMMARY")
print("=" * 50)
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Features removed:", len(constant_columns))
print("Remaining numeric features:",
      len(df.select_dtypes(include=np.number).columns))
print("Missing cells:", int(df.isna().sum().sum()))
print("Timestamp type in memory:", df["Timestamp"].dtype)
print("Earliest timestamp:", df["Timestamp"].min())
print("Latest timestamp:", df["Timestamp"].max())
print("\nLabel distribution:")
print(df["Label"].value_counts())
print("\nSaved to:", output_file)
