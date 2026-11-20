
from pathlib import Path
import pandas as pd

# Project paths
project_dir = Path(r"F:\Projects\SentinelX")
data_dir = project_dir / "Datasets"

input_file = data_dir / "Wednesday-14-02-2018_prepared.csv"

train_file = data_dir / "train.csv"
validation_file = data_dir / "validation.csv"
test_file = data_dir / "test.csv"

print("Loading prepared dataset...")
df = pd.read_csv(input_file, low_memory=False)

# Restore timestamps and ensure chronological order
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="coerce"
)

df = df.dropna(subset=["Timestamp"])
df = df.sort_values("Timestamp", kind="stable").reset_index(drop=True)

# Chronological 70% / 15% / 15% split
total_rows = len(df)
train_end = int(total_rows * 0.70)
validation_end = int(total_rows * 0.85)

train_df = df.iloc[:train_end].copy()
validation_df = df.iloc[train_end:validation_end].copy()
test_df = df.iloc[validation_end:].copy()

# Save without overwriting the prepared dataset
train_df.to_csv(train_file, index=False)
validation_df.to_csv(validation_file, index=False)
test_df.to_csv(test_file, index=False)

# Report the split
print("\n" + "=" * 55)
print("CHRONOLOGICAL SPLIT SUMMARY")
print("=" * 55)

for name, part in [
    ("TRAINING", train_df),
    ("VALIDATION", validation_df),
    ("TEST", test_df)
]:
    print(f"\n{name}")
    print("Rows:", len(part))
    print("Start:", part["Timestamp"].min())
    print("End:", part["Timestamp"].max())
    print("\nLabel counts:")
    print(part["Label"].value_counts())

print("\nFiles saved:")
print(train_file)
print(validation_file)
print(test_file)
