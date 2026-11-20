from pathlib import Path
import pandas as pd

DATA = Path(r"F:\Projects\SentinelX\Datasets")

files = {
    "TRAIN": "sentinelx_train_sequence_metadata_v1.csv",
    "VALIDATION": "sentinelx_validation_sequence_metadata_v1.csv",
    "TEST": "sentinelx_test_sequence_metadata_v1.csv",
}

for split, filename in files.items():
    path = DATA / filename
    df = pd.read_csv(path)

    print("\n" + "=" * 70)
    print(split, "LABEL DISTRIBUTION")
    print("=" * 70)

    print("Total sequences:", len(df))
    print("\nLabels:")
    print(df["Target Label"].value_counts(dropna=False).to_string())

    if "Target CaptureDay" in df.columns:
        print("\nCaptureDay distribution:")
        print(df["Target CaptureDay"].value_counts(dropna=False).to_string())

        print("\nLabels by CaptureDay:")
        print(pd.crosstab(
            df["Target CaptureDay"],
            df["Target Label"]
        ).to_string())

    elif "CaptureDay" in df.columns:
        print("\nCaptureDay distribution:")
        print(df["CaptureDay"].value_counts(dropna=False).to_string())

        print("\nLabels by CaptureDay:")
        print(pd.crosstab(
            df["CaptureDay"],
            df["Target Label"]
        ).to_string())
    else:
        print("\nNo CaptureDay column found.")
        print("Columns:", list(df.columns))
