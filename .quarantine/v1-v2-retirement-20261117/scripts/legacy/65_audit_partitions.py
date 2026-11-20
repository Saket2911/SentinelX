import os
import pandas as pd
import numpy as np

ROOT = r"F:\Projects\SentinelX\Datasets"

FILES = {
    "TRAIN": os.path.join(ROOT, "sentinelx_expanded_train_v1.csv"),
    "VALIDATION": os.path.join(ROOT, "sentinelx_expanded_validation_v1.csv"),
    "TEST": os.path.join(ROOT, "sentinelx_expanded_test_v1.csv"),
}

REPORT = os.path.join(ROOT, "sentinelx_partition_audit_v1.txt")
CHUNK_SIZE = 20000

if os.path.exists(REPORT):
    raise FileExistsError(f"Report already exists: {REPORT}")

metadata = ["Timestamp", "CaptureDay", "Label", "Segment ID"]
feature_columns = None

report = []

for name, path in FILES.items():
    print("\n" + "=" * 70)
    print(f"AUDITING {name}")
    print("=" * 70)

    rows = 0
    missing = {}
    inf_counts = {}
    label_counts = {}
    segment_ids = set()
    feature_min = {}
    feature_max = {}
    feature_cols = None

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
        rows += len(chunk)

        if feature_cols is None:
            feature_cols = [c for c in chunk.columns if c not in metadata]
            if len(feature_cols) != 212:
                raise RuntimeError(
                    f"Expected 212 numeric features, found {len(feature_cols)}"
                )

        label_counts_chunk = chunk["Label"].value_counts()
        for label, count in label_counts_chunk.items():
            label_counts[label] = label_counts.get(label, 0) + int(count)

        segment_ids.update(chunk["Segment ID"].unique())

        for col in feature_cols:
            values = pd.to_numeric(chunk[col], errors="coerce")

            missing[col] = missing.get(col, 0) + int(values.isna().sum())

            arr = values.to_numpy(dtype=np.float64)
            inf_counts[col] = inf_counts.get(col, 0) + int(np.isinf(arr).sum())

            finite = arr[np.isfinite(arr)]
            if len(finite):
                mn = float(finite.min())
                mx = float(finite.max())

                feature_min[col] = min(feature_min.get(col, mn), mn)
                feature_max[col] = max(feature_max.get(col, mx), mx)

    print(f"Rows: {rows:,}")
    print(f"Unique segments: {len(segment_ids)}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Total missing feature cells: {sum(missing.values()):,}")
    print(f"Total infinite feature cells: {sum(inf_counts.values()):,}")

    report.append(f"\n{name}")
    report.append("-" * 60)
    report.append(f"Rows: {rows:,}")
    report.append(f"Segments: {len(segment_ids)}")
    report.append(f"Feature count: {len(feature_cols)}")
    report.append(f"Missing cells: {sum(missing.values()):,}")
    report.append(f"Infinite cells: {sum(inf_counts.values()):,}")

    report.append("\nLabel counts:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count:,}")
        report.append(f"  {label}: {count:,}")

    report.append("\nFeatures with missing values:")
    for col, count in missing.items():
        if count:
            report.append(f"  {col}: {count:,}")

    report.append("\nFeatures with infinite values:")
    for col, count in inf_counts.items():
        if count:
            report.append(f"  {col}: {count:,}")

    report.append("\nFeature min/max:")
    for col in feature_cols:
        report.append(
            f"  {col}: min={feature_min.get(col)}, "
            f"max={feature_max.get(col)}"
        )

with open(REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("\n" + "=" * 70)
print("PARTITION AUDIT COMPLETED")
print(f"Report: {REPORT}")
print("=" * 70)
