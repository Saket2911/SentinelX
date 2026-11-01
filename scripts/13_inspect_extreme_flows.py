
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentinelx.data.loader import load_dataset
from sentinelx.data.splitter import split_dataset

DATASET_PATH = (
    PROJECT_ROOT / "Datasets"
    / "Wednesday-14-02-2018_prepared.csv"
)

TARGET_TIMES = pd.to_datetime([
    "2018-02-14 12:16:21",
    "2018-02-14 12:16:22",
])


def main():
    df = load_dataset(DATASET_PATH)
    train_df, _, test_df = split_dataset(df)

    feature = "TotLen Fwd Pkts"
    if feature not in df.columns:
        raise ValueError(f"Missing column: {feature}")

    df[feature] = pd.to_numeric(df[feature], errors="coerce")

    print("===== OVERALL FLOW-LEVEL DISTRIBUTION =====")
    print("Training flow count:", len(train_df))
    print("Test flow count:", len(test_df))

    for name, part in [
        ("Training", train_df),
        ("Test", test_df),
    ]:
        values = pd.to_numeric(part[feature], errors="coerce")
        print(f"\n{name} {feature} quantiles:")
        print(values.quantile(
            [0, 0.5, 0.9, 0.99, 0.999, 1]
        ).to_string())

    print("\n===== LARGEST ORIGINAL FLOWS IN TEST SET =====")
    available = [
        "Timestamp", "Label", "TotLen Fwd Pkts",
        "Subflow Fwd Byts", "Tot Fwd Pkts",
        "Flow Duration", "Fwd Pkt Len Max",
        "Fwd Pkt Len Mean", "Flow Byts/s", "Flow Pkts/s",
    ]
    available = [c for c in available if c in test_df.columns]

    largest = test_df.nlargest(20, feature)
    print(largest[available].to_string(index=False))

    print("\n===== RECORDS AT EXTREME TIMESTAMPS =====")
    times = pd.to_datetime(df["Timestamp"]).dt.floor("s")
    mask = times.isin(TARGET_TIMES)
    local = df.loc[mask].copy()

    print("Total flow records at these seconds:", len(local))
    print("\nPer-second summary:")
    summary = local.groupby(
        pd.to_datetime(local["Timestamp"]).dt.floor("s")
    ).agg(
        flow_count=(feature, "size"),
        total_forward_bytes=(feature, "sum"),
        mean_forward_bytes=(feature, "mean"),
        std_forward_bytes=(feature, "std"),
        max_forward_bytes=(feature, "max"),
        label_count=("Label", "nunique"),
    )
    print(summary.to_string())

    print("\nLargest flows at those timestamps:")
    local = local.sort_values(feature, ascending=False)
    print(local[available].to_string(index=False, max_rows=20))

    print("\nInspection complete. No data was modified.")


if __name__ == "__main__":
    main()
