
from pathlib import Path
import pandas as pd

root = Path(r"F:\Projects\SentinelX")
data_dir = root / "Datasets"

files = {
    "TRAIN": data_dir / "sentinelx_train_sequence_metadata_v1.csv",
    "VALIDATION": data_dir / "sentinelx_validation_sequence_metadata_v1.csv",
    "TEST": data_dir / "sentinelx_test_sequence_metadata_v1.csv",
}

for partition, path in files.items():

    print("\n" + "=" * 75)
    print(f"{partition} PARTITION")
    print("=" * 75)

    if not path.exists():
        print("Metadata file missing:", path)
        continue

    df = pd.read_csv(path)

    print("\nShape:", df.shape)

    print("\nColumns:")
    for column in df.columns:
        print(f"  {column}: {df[column].dtype}")

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nMissing values:")
    print(df.isna().sum().to_string())

    for column in df.columns:

        name = column.lower()

        if any(word in name for word in [
            "label", "segment", "captureday", "day"
        ]):
            print(f"\nDistribution of {column}:")
            print(
                df[column]
                .value_counts(dropna=False)
                .head(30)
                .to_string()
            )

print("\n" + "=" * 75)
print("PARTITION INSPECTION COMPLETED")
print("No model or dataset files were modified.")
