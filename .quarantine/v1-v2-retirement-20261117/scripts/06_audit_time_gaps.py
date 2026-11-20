
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentinelx.data.loader import load_dataset

DATASET_PATH = (
    PROJECT_ROOT
    / "Datasets"
    / "Wednesday-14-02-2018_prepared.csv"
)


def main():
    df = load_dataset(DATASET_PATH)

    # Identify the distinct one-second intervals containing flows.
    observed = pd.DatetimeIndex(
        df["Timestamp"].dt.floor("1s").unique()
    ).sort_values()

    expected = pd.date_range(
        start=observed.min(),
        end=observed.max(),
        freq="1s",
    )

    missing = expected.difference(observed)

    print("TIME-GAP AUDIT")
    print("First observed second:", observed.min())
    print("Last observed second:", observed.max())
    print("Expected one-second windows:", len(expected))
    print("Windows containing flows:", len(observed))
    print("Windows without recorded flows:", len(missing))

    if len(missing):
        print("\nFirst 20 missing seconds:")
        print(missing[:20].to_list())

    # Measure the gaps between consecutive observed seconds.
    gaps = observed.to_series().diff().dropna()
    gaps_seconds = gaps.dt.total_seconds()

    print("\nGAP BETWEEN CONSECUTIVE OBSERVED SECONDS")
    print(gaps_seconds.describe().to_string())

    print("\nLargest 10 gaps:")
    print(gaps_seconds.nlargest(10).to_string())


if __name__ == "__main__":
    main()
