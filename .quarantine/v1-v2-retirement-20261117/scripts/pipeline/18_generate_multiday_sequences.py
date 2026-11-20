from pathlib import Path
import gc
import numpy as np
import pandas as pd

PROJECT_DIR = Path(r"F:\Projects\SentinelX")
DATA_DIR = PROJECT_DIR / "Datasets"

SEQUENCE_LENGTH = 10

PARTITIONS = {
    "train": DATA_DIR / "sentinelx_train_preprocessed.csv",
    "validation": DATA_DIR / "sentinelx_validation_preprocessed.csv",
    "test": DATA_DIR / "sentinelx_test_preprocessed.csv",
}

METADATA_COLUMNS = {
    "Timestamp",
    "Label",
    "CaptureDay",
    "Segment ID",
}


def generate_partition(name, input_file, feature_columns):

    print(f"\n{'=' * 60}")
    print(f"GENERATING {name.upper()} SEQUENCES")
    print("=" * 60)

    print("Loading:", input_file)

    df = pd.read_csv(
        input_file,
        parse_dates=["Timestamp"]
    )

    if not {"Timestamp", "Label", "Segment ID"}.issubset(df.columns):
        raise ValueError(f"{name}: required metadata columns missing.")

    current_features = [
        col for col in df.columns
        if col not in METADATA_COLUMNS
    ]

    if feature_columns is not None and current_features != feature_columns:
        raise ValueError("Feature columns differ between partitions.")

    if feature_columns is None:
        feature_columns = current_features

    if not feature_columns:
        raise ValueError("No numerical model features found.")

    if not all(
        pd.api.types.is_numeric_dtype(df[col])
        for col in feature_columns
    ):
        raise ValueError("Non-numerical model features found.")

    if df["Timestamp"].isna().any():
        raise ValueError("Invalid timestamps found.")

    if df["Segment ID"].isna().any():
        raise ValueError("Missing Segment IDs found.")

    if df["Label"].isna().any():
        raise ValueError("Missing labels found.")

    # Count sequences before allocating disk-backed arrays.
    segment_sizes = df.groupby("Segment ID", sort=False).size()

    total_sequences = sum(
        max(0, int(size) - SEQUENCE_LENGTH)
        for size in segment_sizes
    )

    if total_sequences == 0:
        raise ValueError(f"{name}: no sequences can be generated.")

    number_of_features = len(feature_columns)

    print("States:", len(df))
    print("Segments:", len(segment_sizes))
    print("Features:", number_of_features)
    print("Sequence length:", SEQUENCE_LENGTH)
    print("Sequences:", total_sequences)

    prefix = f"sentinelx_{name}"

    x_path = DATA_DIR / f"{prefix}_X.npy"
    y_path = DATA_DIR / f"{prefix}_y.npy"
    timestamp_path = DATA_DIR / f"{prefix}_target_timestamps.npy"
    label_path = DATA_DIR / f"{prefix}_target_labels.npy"
    segment_path = DATA_DIR / f"{prefix}_target_segment_ids.npy"

    paths = [
        x_path,
        y_path,
        timestamp_path,
        label_path,
        segment_path,
    ]

    existing = [p for p in paths if p.exists()]

    if existing:
        raise FileExistsError(
            "Output files already exist. Nothing was overwritten:\n"
            + "\n".join(str(p) for p in existing)
        )

    # Create disk-backed arrays.
    X = np.lib.format.open_memmap(
        x_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            total_sequences,
            SEQUENCE_LENGTH,
            number_of_features,
        ),
    )

    y = np.lib.format.open_memmap(
        y_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_sequences, number_of_features),
    )

    target_timestamps = np.lib.format.open_memmap(
        timestamp_path,
        mode="w+",
        dtype="datetime64[ns]",
        shape=(total_sequences,),
    )

    target_labels = np.lib.format.open_memmap(
        label_path,
        mode="w+",
        dtype="U64",
        shape=(total_sequences,),
    )

    target_segments = np.lib.format.open_memmap(
        segment_path,
        mode="w+",
        dtype=np.int32,
        shape=(total_sequences,),
    )

    position = 0

    # Process one independent segment at a time.
    for segment_id, segment in df.groupby(
        "Segment ID",
        sort=False
    ):

        segment = segment.sort_values(
            "Timestamp",
            kind="stable"
        )

        if len(segment) <= SEQUENCE_LENGTH:
            continue

        values = segment[
            feature_columns
        ].to_numpy(dtype=np.float32)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite features in segment {segment_id}"
            )

        windows = np.lib.stride_tricks.sliding_window_view(
            values,
            window_shape=SEQUENCE_LENGTH + 1,
            axis=0,
        ).transpose(0, 2, 1)

        count = len(segment) - SEQUENCE_LENGTH
        end = position + count

        X[position:end] = windows[:, :-1, :]
        y[position:end] = windows[:, -1, :]

        target_timestamps[position:end] = (
            segment["Timestamp"]
            .iloc[SEQUENCE_LENGTH:]
            .to_numpy(dtype="datetime64[ns]")
        )

        target_labels[position:end] = (
            segment["Label"]
            .iloc[SEQUENCE_LENGTH:]
            .astype(str)
            .to_numpy()
        )

        target_segments[position:end] = int(segment_id)

        position = end

        del values, windows
        gc.collect()

    if position != total_sequences:
        raise RuntimeError(
            f"Expected {total_sequences} sequences, generated {position}."
        )

    # Flush disk-backed arrays.
    for array in [
        X, y, target_timestamps, target_labels, target_segments
    ]:
        array.flush()

    print("\nSequence generation completed.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Target timestamps:", len(target_timestamps))
    print("Target labels:", len(target_labels))
    print("Target Segment IDs:", len(target_segments))

    del X, y, target_timestamps, target_labels, target_segments
    del df, segment_sizes
    gc.collect()

    return feature_columns, total_sequences


def main():

    print("=" * 65)
    print("SENTINELX MEMORY-SAFE SEQUENCE GENERATION")
    print("=" * 65)

    for name, path in PARTITIONS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")

    feature_columns = None
    results = {}

    for name, input_file in PARTITIONS.items():

        feature_columns, count = generate_partition(
            name,
            input_file,
            feature_columns,
        )

        results[name] = count

    feature_file = DATA_DIR / "sentinelx_sequence_feature_names.txt"
    report_file = DATA_DIR / "sentinelx_sequence_generation_report.txt"

    if feature_file.exists() or report_file.exists():
        raise FileExistsError(
            "Feature list or report already exists. "
            "Inspect existing outputs before rerunning."
        )

    feature_file.write_text(
        "\n".join(feature_columns),
        encoding="utf-8",
    )

    report = f"""
SENTINELX SEQUENCE GENERATION REPORT
====================================

Sequence length: {SEQUENCE_LENGTH}
Numerical features: {len(feature_columns)}

TRAINING
--------
Sequences: {results['train']}

VALIDATION
----------
Sequences: {results['validation']}

TESTING
-------
Sequences: {results['test']}

TOTAL SEQUENCES
---------------
{sum(results.values())}

INPUT:
Previous {SEQUENCE_LENGTH} network states.

TARGET:
The immediately following network state.

SEGMENT HANDLING:
Sequences never cross Segment ID boundaries.

METADATA:
Target timestamps, labels and Segment IDs saved separately.

STORAGE:
Disk-backed NumPy arrays (.npy).

SEQUENCE GENERATION COMPLETED SUCCESSFULLY.
"""

    report_file.write_text(report, encoding="utf-8")

    print(report)
    print("Feature list:", feature_file)
    print("Report:", report_file)
    print("\nALL PARTITIONS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
