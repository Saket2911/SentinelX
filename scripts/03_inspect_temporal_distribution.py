
from pathlib import Path
import pandas as pd

DATA_PATH = Path(
    r"F:\Projects\SentinelX\Datasets\Wednesday-14-02-2018_prepared.csv"
)

df = pd.read_csv(DATA_PATH, low_memory=False)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"], errors="raise"
)

df = df.sort_values("Timestamp", kind="stable")

print("\n========== DATASET SUMMARY ==========")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))
print("Start time:", df["Timestamp"].min())
print("End time:", df["Timestamp"].max())
print("Unique timestamps:", df["Timestamp"].nunique())

print("\n========== LABEL DISTRIBUTION ==========")
print(df["Label"].value_counts())

print("\n========== FLOWS PER SECOND ==========")
per_second = df.groupby(
    df["Timestamp"].dt.floor("s")
).size()

print(per_second.describe())

print("\n========== LABELS BY 1-MINUTE WINDOW ==========")
df["Minute"] = df["Timestamp"].dt.floor("min")

minute_labels = pd.crosstab(
    df["Minute"], df["Label"]
)

print(minute_labels.to_string())

print("\n========== CHRONOLOGICAL SPLIT CHECK ==========")

n = len(df)
train_end = int(n * 0.70)
validation_end = int(n * 0.85)

train_cutoff = df.iloc[train_end]["Timestamp"]
validation_cutoff = df.iloc[validation_end]["Timestamp"]

train = df[df["Timestamp"] < train_cutoff]
validation = df[
    (df["Timestamp"] >= train_cutoff)
    & (df["Timestamp"] < validation_cutoff)
]
test = df[df["Timestamp"] >= validation_cutoff]

for name, part in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test),
]:
    print(f"\n{name}")
    print("Rows:", len(part))
    print("Start:", part["Timestamp"].min())
    print("End:", part["Timestamp"].max())
    print(part["Label"].value_counts())

print("\n========== INSPECTION COMPLETE ==========")

