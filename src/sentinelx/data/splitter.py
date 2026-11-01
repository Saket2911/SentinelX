
import pandas as pd


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronologically split timestamped network states.

    Identical timestamps always remain in the same partition.
    Existing Segment IDs are preserved.

    Returns:
        train_df, validation_df, test_df
    """

    if "Timestamp" not in df.columns:
        raise ValueError("Dataset must contain a Timestamp column.")

    if df.empty:
        raise ValueError("Cannot split an empty dataset.")

    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1.")

    if not (0 < validation_ratio < 1):
        raise ValueError("validation_ratio must be between 0 and 1.")

    if train_ratio + validation_ratio >= 1:
        raise ValueError("Split ratios must sum to less than 1.")

    work = df.copy()

    work["Timestamp"] = pd.to_datetime(
        work["Timestamp"], errors="coerce"
    )

    if work["Timestamp"].isna().any():
        raise ValueError("Invalid timestamps found.")

    work = work.sort_values(
        "Timestamp", kind="stable"
    ).reset_index(drop=True)

    if len(work) < 3:
        raise ValueError("At least three states are required.")

    # Select boundaries using row-count targets.
    # All states sharing a timestamp stay together.
    train_target = int(len(work) * train_ratio)
    validation_target = int(
        len(work) * (train_ratio + validation_ratio)
    )

    train_cutoff = work.iloc[train_target]["Timestamp"]
    validation_cutoff = work.iloc[validation_target]["Timestamp"]

    if train_cutoff >= validation_cutoff:
        raise ValueError(
            "Unable to create distinct chronological boundaries."
        )

    train_df = work[
        work["Timestamp"] < train_cutoff
    ].copy()

    validation_df = work[
        (work["Timestamp"] >= train_cutoff)
        & (work["Timestamp"] < validation_cutoff)
    ].copy()

    test_df = work[
        work["Timestamp"] >= validation_cutoff
    ].copy()

    if min(len(train_df), len(validation_df), len(test_df)) == 0:
        raise ValueError("One or more partitions are empty.")

    # Verify chronological separation.
    if not (
        train_df["Timestamp"].max()
        < validation_df["Timestamp"].min()
        and validation_df["Timestamp"].max()
        < test_df["Timestamp"].min()
    ):
        raise RuntimeError("Chronological split verification failed.")

    return train_df, validation_df, test_df
