import os
import numpy as np
import pandas as pd

ROOT = r"F:\Projects\SentinelX\Datasets"
SEQ_LEN = 10
FEATURES = 212

PARTITIONS = {
    "train": 199120,
    "validation": 51303,
    "test": 31426,
}

print("=" * 70)
print("SENTINELX STEP 69 — SEQUENCE INTEGRITY VALIDATION")
print("=" * 70)

for name, expected_rows in PARTITIONS.items():
    print(f"\nPARTITION: {name.upper()}")

    x_path = os.path.join(ROOT, f"sentinelx_{name}_X_seq10_v1.npy")
    y_path = os.path.join(ROOT, f"sentinelx_{name}_y_next_v1.npy")
    source_path = os.path.join(
        ROOT, f"sentinelx_expanded_{name}_preprocessed_v1.csv"
    )
    metadata_path = os.path.join(
        ROOT, f"sentinelx_{name}_sequence_metadata_v1.csv"
    )

    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    assert X.ndim == 3, "X must be 3-dimensional"
    assert y.ndim == 2, "y must be 2-dimensional"
    assert X.shape[1:] == (SEQ_LEN, FEATURES)
    assert y.shape[1] == FEATURES
    assert X.shape[0] == y.shape[0]

    print("Shape checks: PASS")

    # Verify sampled sequence values against original preprocessed rows.
    source = pd.read_csv(source_path)
    feature_cols = [
        c for c in source.columns
        if c not in ["Timestamp", "CaptureDay", "Label", "Segment ID"]
    ]

    assert len(feature_cols) == FEATURES

    # Metadata is used to locate the source start row.
    meta = pd.read_csv(metadata_path)

    print("Metadata columns:", list(meta.columns))
    print("Metadata rows:", len(meta))

    # Find likely source-index column.
    candidates = [
        c for c in meta.columns
        if c.lower() in ["start_index", "source_start_index", "start_row"]
    ]

    if not candidates:
        print(
            "Could not automatically identify source start-index column. "
            "Please share metadata column names for manual verification."
        )
        continue

    start_col = candidates[0]

    sample_indices = np.linspace(
        0, len(X) - 1, min(25, len(X)), dtype=int
    )

    for i in sample_indices:
        start = int(meta.iloc[i][start_col])

        expected_x = source.iloc[
            start:start + SEQ_LEN
        ][feature_cols].to_numpy(dtype=np.float32)

        expected_y = source.iloc[
            start + SEQ_LEN
        ][feature_cols].to_numpy(dtype=np.float32)

        np.testing.assert_allclose(
            X[i], expected_x, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            y[i], expected_y, rtol=1e-5, atol=1e-5
        )

    print("Sampled X/y alignment: PASS")
    print("Segment boundary verification: requires segment metadata.")
    print("Partition validation: PASS")

print("\nStep 69 preliminary checks completed.")
