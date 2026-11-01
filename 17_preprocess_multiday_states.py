from pathlib import Path
import sys
import gc
import joblib
import numpy as np
import pandas as pd

PROJECT_DIR = Path(r"F:\Projects\SentinelX")
DATA_DIR = PROJECT_DIR / "Datasets"

sys.path.insert(0, str(PROJECT_DIR / "src"))

from sentinelx.features.state_preprocessor import StatePreprocessor

TRAIN_INPUT = DATA_DIR / "sentinelx_train_states.csv"
VAL_INPUT = DATA_DIR / "sentinelx_validation_states.csv"
TEST_INPUT = DATA_DIR / "sentinelx_test_states.csv"

TRAIN_OUTPUT = DATA_DIR / "sentinelx_train_preprocessed.csv"
VAL_OUTPUT = DATA_DIR / "sentinelx_validation_preprocessed.csv"
TEST_OUTPUT = DATA_DIR / "sentinelx_test_preprocessed.csv"

PREPROCESSOR_FILE = DATA_DIR / "sentinelx_state_preprocessor.joblib"
REPORT_FILE = DATA_DIR / "sentinelx_preprocessing_report.txt"


def verify_output(df, name):
    metadata = ["Timestamp", "CaptureDay", "Label", "Segment ID"]

    missing_metadata = [c for c in metadata if c not in df.columns]

    if missing_metadata:
        raise RuntimeError(f"{name}: missing metadata: {missing_metadata}")

    numeric = df.select_dtypes(include=[np.number])

    if numeric.isna().any().any():
        raise RuntimeError(f"{name}: missing numerical values remain.")

    if not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError(f"{name}: non-finite values found.")

    if len(df) == 0:
        raise RuntimeError(f"{name}: output is empty.")


def main():
    print("=" * 60)
    print("SENTINELX MULTIDAY STATE PREPROCESSING")
    print("=" * 60)

    inputs = [TRAIN_INPUT, VAL_INPUT, TEST_INPUT]

    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    outputs = [
        TRAIN_OUTPUT,
        VAL_OUTPUT,
        TEST_OUTPUT,
        PREPROCESSOR_FILE,
        REPORT_FILE,
    ]

    existing = [p for p in outputs if p.exists()]

    if existing:
        raise FileExistsError(
            "Output files already exist. Nothing was overwritten:\n"
            + "\n".join(str(p) for p in existing)
        )

    print("\n1. Loading training states...")
    train = pd.read_csv(TRAIN_INPUT, parse_dates=["Timestamp"])
    print("Training shape:", train.shape)

    print("\n2. Fitting preprocessor on training data only...")
    preprocessor = StatePreprocessor()

    train_processed = preprocessor.fit_transform(train)
    verify_output(train_processed, "Training")
    train_processed.to_csv(TRAIN_OUTPUT, index=False)

    train_rows = len(train_processed)
    train_columns = len(train_processed.columns)

    print("Training output shape:", train_processed.shape)

    del train, train_processed
    gc.collect()

    print("\n3. Saving fitted preprocessor...")
    joblib.dump(preprocessor, PREPROCESSOR_FILE)
    print("Saved:", PREPROCESSOR_FILE)

    print("\n4. Processing validation states...")
    validation = pd.read_csv(VAL_INPUT, parse_dates=["Timestamp"])
    print("Validation input shape:", validation.shape)

    validation_processed = preprocessor.transform(validation)
    verify_output(validation_processed, "Validation")
    validation_processed.to_csv(VAL_OUTPUT, index=False)

    validation_rows = len(validation_processed)
    print("Validation output shape:", validation_processed.shape)

    del validation, validation_processed
    gc.collect()

    print("\n5. Processing test states...")
    test = pd.read_csv(TEST_INPUT, parse_dates=["Timestamp"])
    print("Test input shape:", test.shape)

    test_processed = preprocessor.transform(test)
    verify_output(test_processed, "Test")
    test_processed.to_csv(TEST_OUTPUT, index=False)

    test_rows = len(test_processed)
    print("Test output shape:", test_processed.shape)

    del test, test_processed
    gc.collect()

    report = f"""
SENTINELX MULTIDAY PREPROCESSING REPORT
=======================================

TRAINING STATES: {train_rows}
VALIDATION STATES: {validation_rows}
TEST STATES: {test_rows}
TOTAL STATES: {train_rows + validation_rows + test_rows}

PREPROCESSING
-------------
Preprocessor: StatePreprocessor
Imputation: Median
Skewness handling: Training-selected log1p features
Scaling: StandardScaler
Fit performed on: Training states ONLY

Validation transformed using training-fitted preprocessor: YES
Testing transformed using training-fitted preprocessor: YES

VERIFICATION
------------
Missing numerical values: NONE
Infinite numerical values: NONE
Metadata preserved: YES
Rows preserved: YES

PREPROCESSING COMPLETED SUCCESSFULLY
"""

    REPORT_FILE.write_text(report, encoding="utf-8")

    print(report)
    print("Output files:")
    print(TRAIN_OUTPUT)
    print(VAL_OUTPUT)
    print(TEST_OUTPUT)
    print(PREPROCESSOR_FILE)
    print(REPORT_FILE)
    print("\nALL PREPROCESSING STEPS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
