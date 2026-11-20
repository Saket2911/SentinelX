
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"F:\Projects\SentinelX")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentinelx.data.loader import load_dataset
from sentinelx.data.splitter import split_dataset
from sentinelx.features.sequences import build_network_states


DATA_PATH = (
    PROJECT_ROOT
    / "Datasets"
    / "Wednesday-14-02-2018_prepared.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "state_feature_distribution_audit.csv"
)


def main():
    print("Loading prepared dataset...")
    df = load_dataset(DATA_PATH)

    print("Creating chronological data splits...")
    train_df, val_df, test_df = split_dataset(df)

    print("Building training network states...")
    train_states = build_network_states(train_df)

    metadata = {"Timestamp", "Label", "Segment ID"}
    feature_columns = [
        c for c in train_states.columns
        if c not in metadata
        and pd.api.types.is_numeric_dtype(train_states[c])
    ]

    features = train_states[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    rows = []

    for column in feature_columns:
        series = features[column].dropna()

        if series.empty:
            continue

        minimum = float(series.min())
        maximum = float(series.max())
        skewness = float(series.skew())

        rows.append({
            "Feature": column,
            "Minimum": minimum,
            "Median": float(series.median()),
            "P99": float(series.quantile(0.99)),
            "P999": float(series.quantile(0.999)),
            "Maximum": maximum,
            "Skewness": skewness,
            "Nonnegative": minimum >= 0,
            "Recommend_Log1p_Candidate": (
                minimum >= 0 and skewness > 2
            ),
        })

    report = pd.DataFrame(rows)

    report = report.sort_values(
        ["Recommend_Log1p_Candidate", "Skewness"],
        ascending=[False, False],
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)

    candidates = report[
        report["Recommend_Log1p_Candidate"]
    ]

    print("\n--- AUDIT SUMMARY ---")
    print(f"Training states: {len(train_states):,}")
    print(f"Numeric state features: {len(feature_columns)}")
    print(f"Log1p candidates: {len(candidates)}")

    print("\nTop 20 candidate features:")
    print(
        candidates[
            [
                "Feature",
                "Median",
                "P99",
                "P999",
                "Maximum",
                "Skewness",
            ]
        ].head(20).to_string(index=False)
    )

    target = "TotLen Fwd Pkts__std"
    matching = report[report["Feature"] == target]

    if not matching.empty:
        print("\nPreviously identified extreme feature:")
        print(matching.to_string(index=False))

    print(f"\nFull report saved to:\n{REPORT_PATH}")


if __name__ == "__main__":
    main()
