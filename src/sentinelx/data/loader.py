
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"Timestamp", "Label"}


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """
    Load a prepared network-flow dataset and validate its structure.

    The function preserves labels and missing feature values.
    It does not perform imputation, scaling, or model training.
    """
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path, low_memory=False)

    # Check the columns needed by the pipeline.
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: "
            f"{sorted(missing_columns)}"
        )

    # Parse timestamps without silently discarding bad records.
    timestamps = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    invalid_count = int(timestamps.isna().sum())

    if invalid_count:
        raise ValueError(
            f"Found {invalid_count} invalid timestamps. "
            "Inspect the source data before proceeding."
        )

    if df["Label"].isna().any():
        raise ValueError(
            "Missing labels found. Resolve them before training."
        )

    df["Timestamp"] = timestamps

    # Preserve original row order when timestamps are equal.
    df = (
        df.sort_values("Timestamp", kind="stable")
        .reset_index(drop=True)
    )

    return df
