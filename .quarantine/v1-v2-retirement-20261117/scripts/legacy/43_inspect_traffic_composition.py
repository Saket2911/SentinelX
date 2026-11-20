import pandas as pd
from pathlib import Path

path = Path(
    r"F:\Projects\SentinelX\Datasets\diagnostics\mixed_traffic_analysis_v1\per_second_traffic_composition.csv"
)

if not path.exists():
    raise FileNotFoundError(path)

df = pd.read_csv(path)

print("=" * 65)
print("TRAFFIC COMPOSITION INSPECTION")
print("=" * 65)

print("\nShape:", df.shape)
print("\nColumns and data types:")
print(df.dtypes.to_string())

print("\nFirst 5 records:")
print(df.head().to_string(index=False))

time_columns = [
    c for c in df.columns
    if any(word in c.lower() for word in ["timestamp", "datetime", "time", "date"])
]

if not time_columns:
    print("\nNo obvious timestamp column found.")
else:
    col = time_columns[0]
    print("\nUsing timestamp column:", col)

    parsed = pd.to_datetime(df[col], errors="coerce")
    target = pd.Timestamp("2018-03-02 10:13:28")

    mask = (
        (parsed >= target - pd.Timedelta(seconds=10)) &
        (parsed <= target + pd.Timedelta(seconds=30))
    )

    print("\nRecords around the first rapid transition:")
    print(df.loc[mask].to_string(index=False))

print("\nInspection complete.")
