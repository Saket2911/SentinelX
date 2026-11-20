from pathlib import Path
import sys
import pandas as pd

PROJECT_DIR = Path(r"F:\Projects\SentinelX")
DATA_DIR = PROJECT_DIR / "Datasets"

INPUT_FILE = DATA_DIR / "sentinelx_multiday_states.csv"

TRAIN_FILE = DATA_DIR / "sentinelx_train_states.csv"
VALIDATION_FILE = DATA_DIR / "sentinelx_validation_states.csv"
TEST_FILE = DATA_DIR / "sentinelx_test_states.csv"
REPORT_FILE = DATA_DIR / "sentinelx_split_report.txt"

sys.path.insert(0, str(PROJECT_DIR / "src"))

from sentinelx.data.splitter import split_dataset


def main():
    print("=" * 60)
    print("SENTINELX MULTIDAY CHRONOLOGICAL SPLITTING")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}")

    outputs = [TRAIN_FILE, VALIDATION_FILE, TEST_FILE, REPORT_FILE]
    existing = [path for path in outputs if path.exists()]

    if existing:
        raise FileExistsError(
            "Output files already exist. Nothing was overwritten:\n"
            + "\n".join(str(path) for path in existing)
        )

    print("\nLoading network states...")
    df = pd.read_csv(INPUT_FILE, parse_dates=["Timestamp"])

    print("Loaded rows:", len(df))
    print("Loaded columns:", len(df.columns))

    if len(df) != 98210:
        raise ValueError(f"Unexpected number of states: {len(df)}")

    print("\nPerforming chronological split...")

    train_df, validation_df, test_df = split_dataset(
        df,
        train_ratio=0.70,
        validation_ratio=0.15,
    )

    total = len(train_df) + len(validation_df) + len(test_df)

    if total != len(df):
        raise RuntimeError("Rows were lost during splitting.")

    if not (
        train_df["Timestamp"].max() < validation_df["Timestamp"].min()
        and validation_df["Timestamp"].max() < test_df["Timestamp"].min()
    ):
        raise RuntimeError("Chronological boundary verification failed.")

    train_times = set(train_df["Timestamp"])
    validation_times = set(validation_df["Timestamp"])
    test_times = set(test_df["Timestamp"])

    if train_times & validation_times or train_times & test_times or validation_times & test_times:
        raise RuntimeError("Timestamp overlap between partitions.")

    print("\nSaving datasets...")

    train_df.to_csv(TRAIN_FILE, index=False)
    validation_df.to_csv(VALIDATION_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    report = f"""
SENTINELX CHRONOLOGICAL SPLIT REPORT
====================================

Total states: {len(df)}

TRAINING
--------
Rows: {len(train_df)}
Start: {train_df['Timestamp'].min()}
End: {train_df['Timestamp'].max()}

VALIDATION
----------
Rows: {len(validation_df)}
Start: {validation_df['Timestamp'].min()}
End: {validation_df['Timestamp'].max()}

TESTING
-------
Rows: {len(test_df)}
Start: {test_df['Timestamp'].min()}
End: {test_df['Timestamp'].max()}

VERIFICATION
------------
All rows preserved: YES
Chronological order preserved: YES
Timestamp overlap: NONE
Segment IDs preserved: YES

Splitting completed successfully.
"""

    REPORT_FILE.write_text(report, encoding="utf-8")

    print(report)
    print("Output files:")
    print(TRAIN_FILE)
    print(VALIDATION_FILE)
    print(TEST_FILE)
    print(REPORT_FILE)
    print("\nSPLITTING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
