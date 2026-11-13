

import os
import json
import numpy as np
import pandas as pd

ROOT = r"F:\Projects\SentinelX\Datasets"

SEQ_LEN = 10
FEATURES = 212
CHUNK_SIZE = 10000

PARTITIONS = {
    "train": 199120,
    "validation": 51303,
    "test": 31426,
}

METADATA_COLUMNS = [
    "Timestamp",
    "CaptureDay",
    "Label",
    "Segment ID",
]

REPORT_PATH = os.path.join(
    ROOT, "sentinelx_step69c_validation_v1.txt"
)

if os.path.exists(REPORT_PATH):
    raise FileExistsError(
        f"Report already exists: {REPORT_PATH}"
    )

with open(
    os.path.join(ROOT, "sentinelx_model_features_v1.json"),
    "r",
    encoding="utf-8",
) as f:
    feature_columns = json.load(f)

if len(feature_columns) != FEATURES:
    raise RuntimeError(
        f"Expected {FEATURES} features, got {len(feature_columns)}"
    )

report = []
total_samples = 0


def record(message):
    print(message)
    report.append(str(message))


record("=" * 75)
record("SENTINELX STEP 69C - SEQUENCE ALIGNMENT VALIDATION")
record("=" * 75)


for name, expected_rows in PARTITIONS.items():

    record("\n" + "=" * 75)
    record(f"PARTITION: {name.upper()}")
    record("=" * 75)

    source_path = os.path.join(
        ROOT,
        f"sentinelx_expanded_{name}_preprocessed_v1.csv",
    )

    X_path = os.path.join(
        ROOT, f"sentinelx_{name}_X_seq10_v1.npy"
    )

    y_path = os.path.join(
        ROOT, f"sentinelx_{name}_y_next_v1.npy"
    )

    metadata_path = os.path.join(
        ROOT,
        f"sentinelx_{name}_sequence_metadata_v1.csv",
    )

    # --------------------------------------------------
    # 1. Validate source schema
    # --------------------------------------------------

    header = pd.read_csv(source_path, nrows=0)

    source_feature_columns = [
        c for c in header.columns
        if c not in METADATA_COLUMNS
    ]

    if source_feature_columns != feature_columns:
        raise RuntimeError(
            f"{name}: Feature names/order mismatch"
        )

    if len(header.columns) != FEATURES + len(METADATA_COLUMNS):
        raise RuntimeError(
            f"{name}: Unexpected source column count"
        )

    # --------------------------------------------------
    # 2. Load arrays as memory maps
    # --------------------------------------------------

    X = np.load(X_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")

    if X.ndim != 3:
        raise RuntimeError(f"{name}: X must be 3D")

    if y.ndim != 2:
        raise RuntimeError(f"{name}: y must be 2D")

    if X.shape[1:] != (SEQ_LEN, FEATURES):
        raise RuntimeError(
            f"{name}: Invalid X shape {X.shape}"
        )

    if y.shape[1] != FEATURES:
        raise RuntimeError(
            f"{name}: Invalid y shape {y.shape}"
        )

    if X.shape[0] != y.shape[0]:
        raise RuntimeError(
            f"{name}: X/y sample count mismatch"
        )

    # --------------------------------------------------
    # 3. Check complete arrays for NaN and infinity
    # --------------------------------------------------

    nonfinite_X = 0
    nonfinite_y = 0

    for start in range(0, len(X), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(X))

        nonfinite_X += int(
            (~np.isfinite(X[start:end])).sum()
        )

        nonfinite_y += int(
            (~np.isfinite(y[start:end])).sum()
        )

    if nonfinite_X or nonfinite_y:
        raise RuntimeError(
            f"{name}: Non-finite values found "
            f"in X={nonfinite_X}, y={nonfinite_y}"
        )

    # --------------------------------------------------
    # 4. Load source metadata and sequence metadata
    # --------------------------------------------------

    source_meta = pd.read_csv(
        source_path,
        usecols=METADATA_COLUMNS,
        dtype=str,
        keep_default_na=False,
    )

    meta = pd.read_csv(
        metadata_path,
        dtype=str,
        keep_default_na=False,
    )

    if len(source_meta) != expected_rows:
        raise RuntimeError(
            f"{name}: Source rows {len(source_meta)} "
            f"!= {expected_rows}"
        )

    if len(meta) != len(X):
        raise RuntimeError(
            f"{name}: Metadata/X sample count mismatch"
        )

    if len(meta) != len(y):
        raise RuntimeError(
            f"{name}: Metadata/y sample count mismatch"
        )

    required_meta = [
        "Sample Index",
        "Target Timestamp",
        "Target CaptureDay",
        "Target Label",
        "Segment ID",
        "Target State Row",
    ]

    if list(meta.columns) != required_meta:
        raise RuntimeError(
            f"{name}: Unexpected metadata columns: "
            f"{list(meta.columns)}"
        )

    # --------------------------------------------------
    # 5. Reconstruct expected target rows
    # --------------------------------------------------

    segment_ids = source_meta["Segment ID"].to_numpy()

    boundaries = (
        np.flatnonzero(
            segment_ids[1:] != segment_ids[:-1]
        ) + 1
    )

    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [expected_rows]))

    expected_targets = np.concatenate([
        np.arange(
            start + SEQ_LEN,
            end,
            dtype=np.int64,
        )
        for start, end in zip(starts, ends)
        if end - start > SEQ_LEN
    ])

    if len(expected_targets) != len(X):
        raise RuntimeError(
            f"{name}: Reconstructed sequence count "
            f"{len(expected_targets)} != {len(X)}"
        )

    actual_targets = pd.to_numeric(
        meta["Target State Row"],
        errors="raise",
    ).to_numpy(dtype=np.int64)

    sample_indices = pd.to_numeric(
        meta["Sample Index"],
        errors="raise",
    ).to_numpy(dtype=np.int64)

    if not np.array_equal(
        sample_indices,
        np.arange(len(meta)),
    ):
        raise RuntimeError(
            f"{name}: Sample Index mismatch"
        )

    if not np.array_equal(
        actual_targets,
        expected_targets,
    ):
        raise RuntimeError(
            f"{name}: Target State Row mismatch"
        )

    record(
        f"Sequence count and target-row mapping: PASS "
        f"({len(X):,})"
    )

    # --------------------------------------------------
    # 6. Verify target metadata against source rows
    # --------------------------------------------------

    target_rows = expected_targets

    checks = {
        "Target Timestamp": "Timestamp",
        "Target CaptureDay": "CaptureDay",
        "Target Label": "Label",
        "Segment ID": "Segment ID",
    }

    for metadata_col, source_col in checks.items():

        actual = meta[metadata_col].to_numpy()
        expected = source_meta[source_col].to_numpy()[
            target_rows
        ]

        if not np.array_equal(actual, expected):
            mismatch_count = int(
                np.count_nonzero(actual != expected)
            )

            raise RuntimeError(
                f"{name}: {metadata_col} mismatch: "
                f"{mismatch_count}"
            )

    record("Target metadata alignment: PASS")

    # --------------------------------------------------
    # 7. Verify every sequence stays within its segment
    # --------------------------------------------------

    segment_start_for_row = np.empty(
        expected_rows, dtype=np.int64
    )

    segment_end_for_row = np.empty(
        expected_rows, dtype=np.int64
    )

    for start, end in zip(starts, ends):
        segment_start_for_row[start:end] = start
        segment_end_for_row[start:end] = end

    if not np.all(
        target_rows - SEQ_LEN
        >= segment_start_for_row[target_rows]
    ):
        raise RuntimeError(
            f"{name}: Sequence crosses segment start"
        )

    if not np.all(
        target_rows < segment_end_for_row[target_rows]
    ):
        raise RuntimeError(
            f"{name}: Target crosses segment end"
        )

    record(
        f"Segment boundary integrity: PASS "
        f"({len(starts)} segments)"
    )

    # --------------------------------------------------
    # 8. Select samples from every segment
    # --------------------------------------------------

    selected = set(
        np.linspace(
            0,
            len(X) - 1,
            min(25, len(X)),
            dtype=int,
        ).tolist()
    )

    sample_position = 0

    for start, end in zip(starts, ends):

        count = max(0, end - start - SEQ_LEN)

        if count > 0:
            selected.add(sample_position)
            selected.add(sample_position + count - 1)

        sample_position += count

    selected = sorted(selected)

    required_rows = set()

    for i in selected:
        target = int(expected_targets[i])

        for row in range(target - SEQ_LEN, target + 1):
            required_rows.add(row)

    # --------------------------------------------------
    # 9. Read only selected source feature rows
    # --------------------------------------------------

    source_values = {}

    row_offset = 0

    for chunk in pd.read_csv(
        source_path,
        usecols=feature_columns,
        chunksize=CHUNK_SIZE,
    ):

        chunk_end = row_offset + len(chunk)

        local_rows = [
            r - row_offset
            for r in required_rows
            if row_offset <= r < chunk_end
        ]

        if local_rows:
            values = chunk.iloc[
                local_rows
            ].to_numpy(dtype=np.float32)

            global_rows = [
                row_offset + r for r in local_rows
            ]

            for global_row, value in zip(
                global_rows, values
            ):
                source_values[global_row] = value

        row_offset = chunk_end

    if len(source_values) != len(required_rows):
        raise RuntimeError(
            f"{name}: Could not retrieve all sampled source rows"
        )

    # --------------------------------------------------
    # 10. Verify actual X/y numerical alignment
    # --------------------------------------------------

    max_error = 0.0

    for i in selected:

        target = int(expected_targets[i])

        expected_x = np.stack([
            source_values[r]
            for r in range(target - SEQ_LEN, target)
        ])

        expected_y = source_values[target]

        actual_x = np.asarray(X[i])
        actual_y = np.asarray(y[i])

        np.testing.assert_allclose(
            actual_x,
            expected_x,
            rtol=1e-5,
            atol=1e-5,
        )

        np.testing.assert_allclose(
            actual_y,
            expected_y,
            rtol=1e-5,
            atol=1e-5,
        )

        max_error = max(
            max_error,
            float(np.max(np.abs(actual_x - expected_x))),
            float(np.max(np.abs(actual_y - expected_y))),
        )

    record(
        f"Numerical X/y alignment: PASS "
        f"({len(selected):,} samples across all segments)"
    )

    record(f"Maximum sampled absolute error: {max_error:.10g}")
    record(f"Non-finite X values: {nonfinite_X}")
    record(f"Non-finite y values: {nonfinite_y}")
    record(f"Partition result: PASS")

    total_samples += len(X)

    del X, y, source_meta, meta, source_values


record("\n" + "=" * 75)
record("FINAL STEP 69C VALIDATION RESULT")
record("=" * 75)
record(f"Total verified sequences: {total_samples:,}")
record("All three partitions passed all implemented checks.")
record("Numerical alignment was sampled across every segment.")
record("No source dataset or model files were modified.")

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print(f"\nReport saved: {REPORT_PATH}")

