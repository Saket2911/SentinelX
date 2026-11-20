import os
import numpy as np
import pandas as pd

ROOT = r"F:\Projects\SentinelX\Datasets"
CHUNK_SIZE = 10000

PARTITIONS = {
    "train": (
        "sentinelx_expanded_train_v1.csv",
        "sentinelx_expanded_train_preprocessed_v1.csv",
        199120,
    ),
    "validation": (
        "sentinelx_expanded_validation_v1.csv",
        "sentinelx_expanded_validation_preprocessed_v1.csv",
        51303,
    ),
    "test": (
        "sentinelx_expanded_test_v1.csv",
        "sentinelx_expanded_test_preprocessed_v1.csv",
        31426,
    ),
}

METADATA = ["Timestamp", "CaptureDay", "Label", "Segment ID"]
REPORT = os.path.join(ROOT, "sentinelx_preprocessed_validation_v1.txt")

if os.path.exists(REPORT):
    raise FileExistsError(f"Report already exists: {REPORT}")

report = []
train_sum = None
train_sum_sq = None
train_n = 0

for name, (original_name, processed_name, expected_rows) in PARTITIONS.items():
    original_path = os.path.join(ROOT, original_name)
    processed_path = os.path.join(ROOT, processed_name)

    print("\n" + "=" * 70)
    print(f"VALIDATING {name.upper()}")
    print("=" * 70)

    orig_iter = pd.read_csv(original_path, chunksize=CHUNK_SIZE)
    proc_iter = pd.read_csv(processed_path, chunksize=CHUNK_SIZE)

    rows = 0
    missing = 0
    inf_count = 0
    metadata_errors = 0
    feature_columns = None

    for orig, proc in zip(orig_iter, proc_iter):
        rows += len(orig)

        if len(orig) != len(proc):
            raise RuntimeError(f"Chunk row mismatch in {name}")

        if feature_columns is None:
            feature_columns = [c for c in proc.columns if c not in METADATA]

        if len(proc.columns) != 216 or len(feature_columns) != 212:
            raise RuntimeError(
                f"Unexpected columns in {name}: "
                f"{len(proc.columns)} total, {len(feature_columns)} features"
            )

        # Metadata must remain unchanged
        for col in METADATA:
            if not orig[col].astype(str).equals(proc[col].astype(str)):
                metadata_errors += int(
                    (orig[col].astype(str) != proc[col].astype(str)).sum()
                )

        X = proc[feature_columns].to_numpy(dtype=np.float64)

        missing += int(np.isnan(X).sum())
        inf_count += int(np.isinf(X).sum())

        if name == "train":
            if train_sum is None:
                train_sum = np.zeros(X.shape[1], dtype=np.float64)
                train_sum_sq = np.zeros(X.shape[1], dtype=np.float64)

            train_sum += np.sum(X, axis=0)
            train_sum_sq += np.sum(X * X, axis=0)
            train_n += X.shape[0]

    print(f"Rows: {rows:,} / expected {expected_rows:,}")
    print(f"Missing model values: {missing:,}")
    print(f"Infinite model values: {inf_count:,}")
    print(f"Metadata mismatches: {metadata_errors:,}")

    report.extend([
        f"{name.upper()}",
        f"Rows: {rows:,}",
        f"Missing values: {missing:,}",
        f"Infinite values: {inf_count:,}",
        f"Metadata mismatches: {metadata_errors:,}",
        "",
    ])

    if rows != expected_rows:
        raise RuntimeError(f"Wrong row count in {name}")
    if missing != 0 or inf_count != 0:
        raise RuntimeError(f"Invalid numeric values in {name}")
    if metadata_errors != 0:
        raise RuntimeError(f"Metadata mismatch in {name}")

# Training mean and population standard deviation
means = train_sum / train_n
variances = train_sum_sq / train_n - means * means
variances = np.maximum(variances, 0)
stds = np.sqrt(variances)

print("\n" + "=" * 70)
print("TRAIN SCALING CHECK")
print("=" * 70)

print(f"Mean of feature means: {means.mean():.8f}")
print(f"Maximum absolute feature mean: {np.max(np.abs(means)):.8f}")
print(f"Mean feature standard deviation: {stds.mean():.8f}")
print(f"Minimum feature standard deviation: {stds.min():.8f}")
print(f"Maximum feature standard deviation: {stds.max():.8f}")

report.extend([
    "TRAIN SCALING",
    f"Mean of feature means: {means.mean():.8f}",
    f"Maximum absolute feature mean: {np.max(np.abs(means)):.8f}",
    f"Mean feature std: {stds.mean():.8f}",
    f"Min feature std: {stds.min():.8f}",
    f"Max feature std: {stds.max():.8f}",
])

with open(REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("\nVALIDATION PASSED")
print(f"Report saved: {REPORT}")
