import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = r"F:\Projects\SentinelX\Datasets"

FILES = {
    "train": os.path.join(ROOT, "sentinelx_expanded_train_v1.csv"),
    "validation": os.path.join(ROOT, "sentinelx_expanded_validation_v1.csv"),
    "test": os.path.join(ROOT, "sentinelx_expanded_test_v1.csv"),
}

OUTPUTS = {
    "train": os.path.join(ROOT, "sentinelx_expanded_train_preprocessed_v1.csv"),
    "validation": os.path.join(ROOT, "sentinelx_expanded_validation_preprocessed_v1.csv"),
    "test": os.path.join(ROOT, "sentinelx_expanded_test_preprocessed_v1.csv"),
}

IMPUTER_PATH = os.path.join(ROOT, "sentinelx_train_medians_v1.npz")
SCALER_PATH = os.path.join(ROOT, "sentinelx_standard_scaler_v1.joblib")
FEATURES_PATH = os.path.join(ROOT, "sentinelx_model_features_v1.json")
REPORT_PATH = os.path.join(ROOT, "sentinelx_preprocessing_report_v1.txt")

CHUNK_SIZE = 10000
METADATA = ["Timestamp", "CaptureDay", "Label", "Segment ID"]

all_outputs = list(OUTPUTS.values()) + [
    IMPUTER_PATH, SCALER_PATH, FEATURES_PATH, REPORT_PATH
]

for path in all_outputs:
    if os.path.exists(path):
        raise FileExistsError(
            f"Output already exists: {path}\n"
            "Do not overwrite existing preprocessing artifacts."
        )

print("=" * 70)
print("SENTINELX TRAIN-ONLY PREPROCESSING")
print("=" * 70)

# ---------------------------------------------------------
# Step 1: Read training feature names
# ---------------------------------------------------------
header = pd.read_csv(FILES["train"], nrows=0)
feature_columns = [c for c in header.columns if c not in METADATA]

if len(feature_columns) != 212:
    raise RuntimeError(f"Expected 212 features, found {len(feature_columns)}")

print(f"Model input features: {len(feature_columns)}")

# ---------------------------------------------------------
# Step 2: Fit imputation medians on TRAIN only
# ---------------------------------------------------------
print("\nLoading training features to calculate medians...")

train_df = pd.read_csv(
    FILES["train"],
    usecols=feature_columns,
    dtype=np.float32
)

train_df.replace([np.inf, -np.inf], np.nan, inplace=True)

medians = train_df.median(axis=0)

if medians.isna().any():
    bad = medians[medians.isna()].index.tolist()
    raise RuntimeError(f"Entirely missing training features: {bad}")

print(f"Missing values in training before imputation: {train_df.isna().sum().sum():,}")

train_df = train_df.fillna(medians)

print(f"Missing values after imputation: {train_df.isna().sum().sum():,}")

# Save medians and feature names
np.savez(
    IMPUTER_PATH,
    medians=medians.to_numpy(dtype=np.float64),
    feature_names=np.array(feature_columns, dtype=str)
)

# ---------------------------------------------------------
# Step 3: Signed log transform on TRAIN
# ---------------------------------------------------------
print("\nApplying signed log transform...")

X_train = train_df.to_numpy(dtype=np.float32)

X_train = np.sign(X_train) * np.log1p(np.abs(X_train))

if not np.isfinite(X_train).all():
    raise RuntimeError("Non-finite values appeared after log transform.")

# ---------------------------------------------------------
# Step 4: Fit scaler on TRAIN only
# ---------------------------------------------------------
print("\nFitting StandardScaler on training data only...")

scaler = StandardScaler()
scaler.fit(X_train)

joblib.dump(scaler, SCALER_PATH)

# Release memory before chunk processing
del train_df
del X_train

# ---------------------------------------------------------
# Step 5: Transform each partition in chunks
# ---------------------------------------------------------
print("\nTransforming train, validation, and test...")

report = []

for name, input_path in FILES.items():
    output_path = OUTPUTS[name]

    rows_written = 0
    first_chunk = True

    for chunk in pd.read_csv(input_path, chunksize=CHUNK_SIZE):
        metadata_df = chunk[METADATA].copy()

        X = chunk[feature_columns].apply(
            pd.to_numeric, errors="coerce"
        )

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X = X.fillna(medians)

        arr = X.to_numpy(dtype=np.float32)

        arr = np.sign(arr) * np.log1p(np.abs(arr))
        arr = scaler.transform(arr).astype(np.float32)

        if not np.isfinite(arr).all():
            raise RuntimeError(
                f"Non-finite values after transformation in {name}"
            )

        processed = pd.DataFrame(arr, columns=feature_columns)
        result = pd.concat(
            [metadata_df.reset_index(drop=True), processed],
            axis=1
        )

        result.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            index=False,
            header=first_chunk
        )

        first_chunk = False
        rows_written += len(result)

        if rows_written % 50000 < CHUNK_SIZE:
            print(f"{name}: processed {rows_written:,} rows")

    print(f"{name.upper()} complete: {rows_written:,} rows")
    report.append(f"{name}: {rows_written:,} rows -> {output_path}")

# ---------------------------------------------------------
# Step 6: Save preprocessing report
# ---------------------------------------------------------
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("SENTINELX PREPROCESSING REPORT\n")
    f.write("=" * 60 + "\n")
    f.write("Imputation medians fitted on TRAIN only.\n")
    f.write("Signed log transform applied to all 212 numeric features.\n")
    f.write("StandardScaler fitted on TRAIN only.\n")
    f.write("Validation and test were transformed without fitting.\n\n")
    f.write(f"Feature count: {len(feature_columns)}\n")
    f.write(f"Chunk size: {CHUNK_SIZE}\n\n")
    f.write("\n".join(report))

    f.write("\n\nSaved artifacts:\n")
    f.write(IMPUTER_PATH + "\n")
    f.write(SCALER_PATH + "\n")
    f.write(FEATURES_PATH + "\n")

with open(FEATURES_PATH, "w", encoding="utf-8") as f:
    json.dump(feature_columns, f, indent=2)

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED")
print("=" * 70)
print(f"Imputer: {IMPUTER_PATH}")
print(f"Scaler: {SCALER_PATH}")
print(f"Feature list: {FEATURES_PATH}")
print(f"Report: {REPORT_PATH}")
