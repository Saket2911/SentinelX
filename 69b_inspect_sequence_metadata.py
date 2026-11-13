import pandas as pd
from pathlib import Path

root = Path(r"F:\Projects\SentinelX\Datasets")

for part in ["train", "validation", "test"]:
    path = root / f"sentinelx_{part}_sequence_metadata_v1.csv"
    df = pd.read_csv(path)

    print(f"\n--- {part.upper()} ---")
    print("First metadata rows:")
    print(df.head(3).to_string(index=False))

    print("\nLast metadata rows:")
    print(df.tail(2).to_string(index=False))
