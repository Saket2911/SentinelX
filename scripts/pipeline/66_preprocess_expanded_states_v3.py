
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
    "train": os.path.join(
        ROOT, "sentinelx_expanded_train_preprocessed_v3.csv"
    ),
    "validation": os.path.join(
        ROOT, "sentinelx_expanded_validation_preprocessed_v3.csv"
    ),
    "test": os.path.join(
        ROOT, "sentinelx_expanded_test_preprocessed_v3.csv"
    ),
}

FEATURES_PATH = os.path.join(
    ROOT, "sentinelx_model_features_v3.json"
)
IMPUTER_PATH = os.path.join(
    ROOT, "sentinelx_train_medians_v3.npz"
)
SCALER_PATH = os.path.join(
    ROOT, "sentinelx_standard_scaler_v3.joblib"
)
REPORT_PATH = os.path.join(
    ROOT, "sentinelx_preprocessing_report_v3.txt"
)

CHUNK_SIZE = 10000
EXPECTED_FEATURES = 209
METADATA = ["Timestamp", "CaptureDay", "Label", "Segment ID"]


# ---------------------------------------------------------
# STEP 0: PREFLIGHT CHECKS
# ---------------------------------------------------------

print("=" * 70)
print("SENTINELX V3 TRAIN-ONLY PREPROCESSING")
print("=" * 70)

for path in FILES.values():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input missing: {path}")

for path in [
    FEATURES_PATH,
    *OUTPUTS.values(),
    IMPUTER_PATH,
    SCALER_PATH,
    REPORT_PATH,
]:
    if os.path.exists(path):
        raise FileExistsError(
            f"Output already exists: {path}\n"
            "No existing artifact will be overwritten."
        )

with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    feature_columns = json.load(f)

if len(feature_columns) != EXPECTED_FEATURES:
    raise RuntimeError(
        f"Expected {EXPECTED_FEATURES} features, "
        f"found {len(feature_columns)}"
    )

if len(feature_columns) != len(set(feature_columns)):
    raise RuntimeError("Duplicate feature names in V3 mapping.")

if not all(name.endswith(("__mean", "__std", "__max"))
           for name in feature_columns):
    raise RuntimeError("Unexpected feature naming detected.")

for name, input_path in FILES.items():
    header = pd.read_csv(input_path, nrows=0).columns.tolist()

    missing = [
        col for col in METADATA + feature_columns
        if col not in header
    ]

    if missing:
        raise RuntimeError(
            f"{name}: missing required columns: {missing[:10]}"
        )

    print(f"{name}: input schema verified")


# ---------------------------------------------------------
# STEP 1: LOAD TRAINING FEATURES
# ---------------------------------------------------------

print("\nLoading training features...")

train_df = pd.read_csv(
    FILES["train"],
    usecols=feature_columns,
    dtype=np.float32
)

# Enforce the exact V3 feature order.
train_df = train_df.loc[:, feature_columns]

if len(train_df) == 0:
    raise RuntimeError("Training partition is empty.")

train_df.replace([np.inf, -np.inf], np.nan, inplace=True)

print("Training rows:", f"{len(train_df):,}")
print(
    "Missing/infinite values before imputation:",
    f"{train_df.isna().sum().sum():,}"
)


# ---------------------------------------------------------
# STEP 2: FIT MEDIANS ON TRAIN ONLY
# ---------------------------------------------------------

medians = train_df.median(axis=0)

if medians.isna().any():
    bad = medians[medians.isna()].index.tolist()
    raise RuntimeError(
        f"Entirely missing training features: {bad}"
    )

train_df = train_df.fillna(medians)

if not np.isfinite(train_df.to_numpy(dtype=np.float32)).all():
    raise RuntimeError("Non-finite values after imputation.")

np.savez(
    IMPUTER_PATH,
    medians=medians.to_numpy(dtype=np.float64),
    feature_names=np.array(feature_columns, dtype=str)
)

print("Training-only median imputation completed.")


# ---------------------------------------------------------
# STEP 3: SIGNED LOG TRANSFORM
# ---------------------------------------------------------

print("\nApplying signed log transform...")

X_train = train_df.to_numpy(dtype=np.float32)

X_train = np.sign(X_train) * np.log1p(np.abs(X_train))

if not np.isfinite(X_train).all():
    raise RuntimeError("Non-finite values after log transform.")


# ---------------------------------------------------------
# STEP 4: FIT SCALER ON TRAIN ONLY
# ---------------------------------------------------------

print("\nFitting StandardScaler on training data only...")

scaler = StandardScaler()
scaler.fit(X_train)

if not np.isfinite(scaler.mean_).all():
    raise RuntimeError("Non-finite scaler means.")

if not np.isfinite(scaler.scale_).all():
    raise RuntimeError("Non-finite scaler scales.")

joblib.dump(scaler, SCALER_PATH)

del train_df, X_train


# ---------------------------------------------------------
# STEP 5: TRANSFORM ALL PARTITIONS IN CHUNKS
# ---------------------------------------------------------

print("\nTransforming train, validation, and test...")

report = []
row_counts = {}

for name, input_path in FILES.items():

    output_path = OUTPUTS[name]

    rows_written = 0
    first_chunk = True

    for chunk in pd.read_csv(
        input_path,
        chunksize=CHUNK_SIZE
    ):

        metadata_df = chunk[METADATA].copy()

        X = chunk[feature_columns].apply(
            pd.to_numeric,
            errors="coerce"
        )

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X = X.fillna(medians)

        arr = X.to_numpy(dtype=np.float32)

        arr = np.sign(arr) * np.log1p(np.abs(arr))
        arr = scaler.transform(arr).astype(np.float32)

        if not np.isfinite(arr).all():
            raise RuntimeError(
                f"Non-finite transformed values in {name}"
            )

        processed = pd.DataFrame(
            arr,
            columns=feature_columns
        )

        result = pd.concat(
            [
                metadata_df.reset_index(drop=True),
                processed
            ],
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
            print(
                f"{name}: processed {rows_written:,} rows"
            )

    if rows_written == 0:
        raise RuntimeError(f"No rows processed for {name}")

    row_counts[name] = rows_written

    print(
        f"{name.upper()} complete: {rows_written:,} rows"
    )

    report.append(
        f"{name}: {rows_written:,} rows -> {output_path}"
    )


# ---------------------------------------------------------
# STEP 6: SAVE V3 PREPROCESSING REPORT
# ---------------------------------------------------------

with open(REPORT_PATH, "w", encoding="utf-8") as f:

    f.write("SENTINELX V3 PREPROCESSING REPORT\n")
    f.write("=" * 65 + "\n")

    f.write("Feature mapping: sentinelx_model_features_v3.json\n")
    f.write(f"Feature count: {EXPECTED_FEATURES}\n")
    f.write(f"Chunk size: {CHUNK_SIZE}\n\n")

    f.write("Preprocessing:\n")
    f.write("- Median imputation fitted on training data only.\n")
    f.write("- Signed log transform applied to all features.\n")
    f.write("- StandardScaler fitted on training data only.\n")
    f.write("- Validation and test transformed without fitting.\n\n")

    f.write("Partition row counts:\n")
    f.write("\n".join(report))

    f.write("\n\nSaved artifacts:\n")
    f.write(IMPUTER_PATH + "\n")
    f.write(SCALER_PATH + "\n")
    f.write(FEATURES_PATH + "\n")


print("\n" + "=" * 70)
print("V3 PREPROCESSING COMPLETED")
print("=" * 70)

print("Feature count:", EXPECTED_FEATURES)

for name, count in row_counts.items():
    print(f"{name.capitalize()} rows: {count:,}")

print("Imputer:", IMPUTER_PATH)
print("Scaler:", SCALER_PATH)
print("Report:", REPORT_PATH)
print("\nV2 artifacts were not modified.")
