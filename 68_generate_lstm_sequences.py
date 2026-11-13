import os
import gc
import json
import numpy as np
import pandas as pd

ROOT = r"F:\Projects\SentinelX\Datasets"
SEQ_LEN = 10
CHUNK_SIZE = 10000

PARTITIONS = {
    "train": {
        "input": "sentinelx_expanded_train_preprocessed_v1.csv",
        "rows": 199120,
    },
    "validation": {
        "input": "sentinelx_expanded_validation_preprocessed_v1.csv",
        "rows": 51303,
    },
    "test": {
        "input": "sentinelx_expanded_test_preprocessed_v1.csv",
        "rows": 31426,
    },
}

METADATA = ["Timestamp", "CaptureDay", "Label", "Segment ID"]

print("=" * 70)
print("SENTINELX LSTM SEQUENCE GENERATION")
print("=" * 70)

# Check that all outputs are new
for name in PARTITIONS:
    outputs = [
        f"sentinelx_{name}_states_features_v1.npy",
        f"sentinelx_{name}_X_seq10_v1.npy",
        f"sentinelx_{name}_y_next_v1.npy",
        f"sentinelx_{name}_sequence_metadata_v1.csv",
        f"sentinelx_{name}_sequence_report_v1.txt",
    ]

    for filename in outputs:
        path = os.path.join(ROOT, filename)
        if os.path.exists(path):
            raise FileExistsError(
                f"Output exists: {path}\n"
                "Rename/remove intentionally before rerunning."
            )

feature_list_path = os.path.join(
    ROOT, "sentinelx_model_features_v1.json"
)

with open(feature_list_path, "r", encoding="utf-8") as f:
    feature_columns = json.load(f)

N_FEATURES = len(feature_columns)

if N_FEATURES != 212:
    raise RuntimeError(f"Expected 212 features, got {N_FEATURES}")

overall_report = []

for name, config in PARTITIONS.items():
    print("\n" + "=" * 70)
    print(f"PARTITION: {name.upper()}")
    print("=" * 70)

    input_path = os.path.join(ROOT, config["input"])
    n_rows = config["rows"]

    feature_path = os.path.join(
        ROOT, f"sentinelx_{name}_states_features_v1.npy"
    )
    X_path = os.path.join(
        ROOT, f"sentinelx_{name}_X_seq10_v1.npy"
    )
    y_path = os.path.join(
        ROOT, f"sentinelx_{name}_y_next_v1.npy"
    )
    metadata_path = os.path.join(
        ROOT, f"sentinelx_{name}_sequence_metadata_v1.csv"
    )
    report_path = os.path.join(
        ROOT, f"sentinelx_{name}_sequence_report_v1.txt"
    )

    # -----------------------------------------------------
    # Pass 1: store numeric states in memory-mapped file
    # -----------------------------------------------------
    print("Pass 1: Writing numeric feature states...")

    states = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_rows, N_FEATURES),
    )

    segment_ids = np.empty(n_rows, dtype=np.int64)
    labels = np.empty(n_rows, dtype=object)
    timestamps = np.empty(n_rows, dtype=object)
    capture_days = np.empty(n_rows, dtype=object)

    row = 0

    for chunk in pd.read_csv(input_path, chunksize=CHUNK_SIZE):
        end = row + len(chunk)

        states[row:end] = chunk[feature_columns].to_numpy(
            dtype=np.float32
        )

        segment_ids[row:end] = chunk["Segment ID"].to_numpy(
            dtype=np.int64
        )
        labels[row:end] = chunk["Label"].astype(str).to_numpy()
        timestamps[row:end] = chunk["Timestamp"].astype(str).to_numpy()
        capture_days[row:end] = chunk["CaptureDay"].astype(str).to_numpy()

        row = end

        if row % 50000 < CHUNK_SIZE:
            print(f"  Loaded {row:,}/{n_rows:,} states")

    states.flush()

    if row != n_rows:
        raise RuntimeError(f"Row count mismatch: {row} vs {n_rows}")

    print(f"Feature matrix stored: {states.shape}")

    # -----------------------------------------------------
    # Pass 2: identify segment ranges and count sequences
    # -----------------------------------------------------
    print("Pass 2: Counting valid sequences...")

    boundaries = np.flatnonzero(
        segment_ids[1:] != segment_ids[:-1]
    ) + 1

    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n_rows]))

    segment_lengths = ends - starts

    sample_count = int(
        np.maximum(segment_lengths - SEQ_LEN, 0).sum()
    )

    print(f"Segments: {len(starts):,}")
    print(f"Sequence length: {SEQ_LEN}")
    print(f"Valid sequence samples: {sample_count:,}")

    if sample_count == 0:
        raise RuntimeError(f"No valid sequences in {name}")

    # -----------------------------------------------------
    # Pass 3: create memory-mapped X and y
    # -----------------------------------------------------
    print("Pass 3: Generating X/y sequences...")

    X = np.lib.format.open_memmap(
        X_path,
        mode="w+",
        dtype=np.float32,
        shape=(sample_count, SEQ_LEN, N_FEATURES),
    )

    y = np.lib.format.open_memmap(
        y_path,
        mode="w+",
        dtype=np.float32,
        shape=(sample_count, N_FEATURES),
    )

    metadata_rows = []
    sample_idx = 0

    for seg_num, (start, end) in enumerate(zip(starts, ends)):
        length = end - start
        n_seg_samples = max(0, length - SEQ_LEN)

        if n_seg_samples == 0:
            continue

        for offset in range(n_seg_samples):
            target_idx = start + offset + SEQ_LEN

            X[sample_idx] = states[
                target_idx - SEQ_LEN:target_idx
            ]
            y[sample_idx] = states[target_idx]

            metadata_rows.append({
                "Sample Index": sample_idx,
                "Target Timestamp": timestamps[target_idx],
                "Target CaptureDay": capture_days[target_idx],
                "Target Label": labels[target_idx],
                "Segment ID": segment_ids[target_idx],
                "Target State Row": target_idx,
            })

            sample_idx += 1

        if seg_num % 100 == 0:
            print(
                f"  Segments processed: {seg_num:,}/{len(starts):,} "
                f"| samples: {sample_idx:,}"
            )

    X.flush()
    y.flush()

    if sample_idx != sample_count:
        raise RuntimeError(
            f"Sequence count mismatch: {sample_idx} vs {sample_count}"
        )

    del X, y, states
    gc.collect()

    # Save sequence target metadata for analysis
    meta_df = pd.DataFrame(metadata_rows)
    meta_df.to_csv(metadata_path, index=False)

    # Report
    report_text = [
        f"SentinelX {name.upper()} sequence report",
        "=" * 60,
        f"Partition rows: {n_rows:,}",
        f"Segment count: {len(starts):,}",
        f"Sequence length: {SEQ_LEN}",
        f"Feature count: {N_FEATURES}",
        f"Sequence samples: {sample_count:,}",
        f"X shape: ({sample_count}, {SEQ_LEN}, {N_FEATURES})",
        f"y shape: ({sample_count}, {N_FEATURES})",
        f"X file: {X_path}",
        f"y file: {y_path}",
        f"Metadata file: {metadata_path}",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_text))

    print(f"\n{name.upper()} sequence generation complete.")
    print(f"X shape: ({sample_count}, {SEQ_LEN}, {N_FEATURES})")
    print(f"y shape: ({sample_count}, {N_FEATURES})")

    overall_report.extend(report_text + [""])

    del segment_ids, labels, timestamps, capture_days
    del meta_df, metadata_rows
    gc.collect()

# Overall report
overall_path = os.path.join(
    ROOT, "sentinelx_sequence_generation_report_v1.txt"
)

if os.path.exists(overall_path):
    raise FileExistsError(overall_path)

with open(overall_path, "w", encoding="utf-8") as f:
    f.write("\n".join(overall_report))

print("\n" + "=" * 70)
print("ALL SEQUENCE PARTITIONS GENERATED")
print("=" * 70)
print(f"Overall report: {overall_path}")
