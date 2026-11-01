
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentinelx.data.loader import load_dataset
from sentinelx.data.splitter import split_dataset
from sentinelx.features.sequences import build_network_states


DATASET_PATH = (
    PROJECT_ROOT / "Datasets"
    / "Wednesday-14-02-2018_prepared.csv"
)
PREPROCESSOR_PATH = (
    PROJECT_ROOT / "artifacts" / "preprocessors"
    / "state_preprocessor.joblib"
)

FEATURE = "TotLen Fwd Pkts__std"
TARGET_TIMES = pd.to_datetime([
    "2018-02-14 12:16:21",
    "2018-02-14 12:16:22",
])


def main():
    print("Loading data...")
    df = load_dataset(DATASET_PATH)
    train_df, _, test_df = split_dataset(df)

    print("Building unscaled network states...")
    train_states = build_network_states(train_df)
    test_states = build_network_states(test_df)

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    if FEATURE not in train_states.columns:
        raise ValueError(f"Feature not found: {FEATURE}")

    # Check the scaler statistics fitted on training states.
    idx = preprocessor.feature_columns.index(FEATURE)
    scaler = preprocessor.scaler

    print("\n===== TRAINING SCALER STATISTICS =====")
    print("Feature:", FEATURE)
    print("Training mean:", scaler.mean_[idx])
    print("Training scale:", scaler.scale_[idx])
    print("Training variance:", scaler.var_[idx])

    # Compare unscaled state values before standardization.
    train_values = train_states[FEATURE].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    test_values = test_states[FEATURE].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    print("\n===== RAW STATE DISTRIBUTION =====")
    print("Training state count:", len(train_values))
    print("Test state count:", len(test_values))
    print("\nTraining quantiles:")
    print(train_values.quantile(
        [0, 0.5, 0.9, 0.99, 0.999, 1]
    ).to_string())
    print("\nTest quantiles:")
    print(test_values.quantile(
        [0, 0.5, 0.9, 0.99, 0.999, 1]
    ).to_string())

    # Show raw state values at the extreme timestamps.
    print("\n===== EXTREME TEST STATES =====")
    for timestamp in TARGET_TIMES:
        mask = pd.to_datetime(
            test_states["Timestamp"]
        ).eq(timestamp)
        rows = test_states.loc[mask]

        if rows.empty:
            print(f"\nNo state found at {timestamp}")
            continue

        for _, row in rows.iterrows():
            raw_value = float(row[FEATURE])
            scaled_value = (
                raw_value - scaler.mean_[idx]
            ) / scaler.scale_[idx]

            print(f"\nTimestamp: {timestamp}")
            print("Label:", row.get("Label", "unknown"))
            print("Raw state feature:", raw_value)
            print("Scaled state feature:", scaled_value)
            print("Flow count:", row.get("Flow Count", "unknown"))

    # Trace the state back to its original flow records.
    print("\n===== ORIGINAL FLOW RECORDS =====")
    flow_times = df["Timestamp"].dt.floor("s")
    mask = flow_times.isin(TARGET_TIMES)
    flow_rows = df.loc[mask].copy()

    print("Matching flow records:", len(flow_rows))

    columns = [
        "Timestamp",
        "Label",
        "TotLen Fwd Pkts",
        "Subflow Fwd Byts",
        "Flow Byts/s",
        "Flow Pkts/s",
    ]
    columns = [c for c in columns if c in flow_rows.columns]

    if not flow_rows.empty:
        print(flow_rows[columns].to_string(
            index=False, max_rows=30
        ))

    print("\nDiagnostic complete. No files were modified.")


if __name__ == "__main__":
    main()
