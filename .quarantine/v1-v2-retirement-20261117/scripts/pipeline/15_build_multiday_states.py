
from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_DIR = Path(r"F:\Projects\SentinelX")
DATA_DIR = PROJECT_DIR / "Datasets"

INPUT_FILE = DATA_DIR / "sentinelx_multiday_chronological_v2.csv"

OBSERVED_FILE = DATA_DIR / "sentinelx_multiday_observed_aggregates.csv"
OUTPUT_FILE = DATA_DIR / "sentinelx_multiday_states.csv"
REPORT_FILE = DATA_DIR / "sentinelx_multiday_states_report.txt"

CHUNK_SIZE = 25_000
MAX_GAP_SECONDS = 8

CONSTANT_FEATURES = {
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "CWE Flag Count",
    "Fwd Byts/b Avg",
    "Fwd Pkts/b Avg",
    "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg",
    "Bwd Pkts/b Avg",
    "Bwd Blk Rate Avg",
}

METADATA_COLUMNS = {
    "Timestamp",
    "Label",
    "CaptureDay",
}


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

def validate_paths():

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(INPUT_FILE)

    existing = [
        path for path in [
            OBSERVED_FILE,
            OUTPUT_FILE,
            REPORT_FILE,
        ]
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "Output files already exist. "
            "Do not overwrite them automatically:\n"
            + "\n".join(map(str, existing))
        )


def get_feature_columns():

    header = pd.read_csv(INPUT_FILE, nrows=0)

    excluded = METADATA_COLUMNS | CONSTANT_FEATURES

    features = [
        column
        for column in header.columns
        if column not in excluded
    ]

    if len(features) != 68:
        raise ValueError(
            f"Expected 68 flow features, found {len(features)}."
        )

    return features


# ---------------------------------------------------------
# LABEL AGGREGATION
# ---------------------------------------------------------

def summarize_label(labels):

    attacks = sorted({
        str(label)
        for label in labels
        if str(label).strip().lower() != "benign"
    })

    if not attacks:
        return "Benign"

    if len(attacks) == 1:
        return attacks[0]

    return "MIXED_ATTACK"


# ---------------------------------------------------------
# AGGREGATE COMPLETE TIMESTAMP GROUPS
# ---------------------------------------------------------

def aggregate_block(frame, feature_columns):

    if frame.empty:
        return

    # Convert infinite values into missing measurements.
    frame[feature_columns] = frame[
        feature_columns
    ].replace([np.inf, -np.inf], np.nan)

    grouped = frame.groupby(
        "Timestamp",
        sort=True,
    )

    aggregated = grouped[feature_columns].agg(
        ["mean", "std", "max"]
    )

    aggregated.columns = [
        f"{feature}__{stat}"
        for feature, stat in aggregated.columns
    ]

    # Preserve capture-day metadata.
    capture_days = grouped["CaptureDay"].first()

    expected_days = (
        aggregated.index.strftime("%Y-%m-%d")
    )

    if not capture_days.astype(str).eq(
        pd.Series(expected_days, index=aggregated.index)
    ).all():
        raise ValueError(
            "CaptureDay/Timestamp mismatch during aggregation."
        )

    flow_counts = grouped.size()

    labels = grouped["Label"].agg(summarize_label)

    aggregated["CaptureDay"] = capture_days.to_numpy()
    aggregated["Flow Count"] = flow_counts.to_numpy()
    aggregated["Label"] = labels.to_numpy()

    aggregated = aggregated.reset_index()

    aggregated.to_csv(
        OBSERVED_FILE,
        mode="a",
        header=not OBSERVED_FILE.exists(),
        index=False,
    )


# ---------------------------------------------------------
# FIRST PASS: STREAM RAW CSV
# ---------------------------------------------------------

def build_observed_aggregates(feature_columns):

    print("\nPASS 1: Aggregating timestamp groups")

    carry = None
    previous_timestamp = None

    total_rows = 0
    processed_blocks = 0

    reader = pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        dtype={
            "Timestamp": "string",
            "CaptureDay": "string",
            "Label": "string",
        },
    )

    for chunk_number, chunk in enumerate(reader, start=1):

        total_rows += len(chunk)

        chunk["Timestamp"] = pd.to_datetime(
            chunk["Timestamp"],
            errors="coerce",
        )

        if chunk["Timestamp"].isna().any():
            raise ValueError(
                f"Invalid timestamp in chunk {chunk_number}."
            )

        if chunk["CaptureDay"].isna().any():
            raise ValueError(
                f"Missing CaptureDay in chunk {chunk_number}."
            )

        if chunk["Label"].isna().any():
            raise ValueError(
                f"Missing Label in chunk {chunk_number}."
            )

        # Validate timestamp order between and within chunks.
        if not chunk["Timestamp"].is_monotonic_increasing:
            raise ValueError(
                f"Timestamp order violation in chunk {chunk_number}."
            )

        if previous_timestamp is not None:

            if chunk["Timestamp"].iloc[0] < previous_timestamp:
                raise ValueError(
                    "Timestamp order violation across CSV chunks."
                )

        previous_timestamp = chunk["Timestamp"].iloc[-1]

        expected_days = (
            chunk["Timestamp"].dt.strftime("%Y-%m-%d")
        )

        if not chunk["CaptureDay"].eq(expected_days).all():
            raise ValueError(
                f"CaptureDay/Timestamp mismatch in chunk {chunk_number}."
            )

        # Check that numerical features were read correctly.
        for column in feature_columns:

            if not pd.api.types.is_numeric_dtype(chunk[column]):
                raise ValueError(
                    f"Non-numeric flow feature: {column}"
                )

        # Reattach the unfinished timestamp group from
        # the previous chunk.
        if carry is not None:

            chunk = pd.concat(
                [carry, chunk],
                ignore_index=True,
            )

        # Keep the final timestamp group for the next chunk.
        final_timestamp = chunk["Timestamp"].iloc[-1]

        complete = chunk[
            chunk["Timestamp"] < final_timestamp
        ]

        carry = chunk[
            chunk["Timestamp"] == final_timestamp
        ].copy()

        if not complete.empty:

            aggregate_block(
                complete,
                feature_columns,
            )

            processed_blocks += 1

        if chunk_number % 10 == 0:

            print(
                f"Chunks read: {chunk_number:,} | "
                f"Raw rows read: {total_rows:,}"
            )

    # Process the final carried timestamp group.
    if carry is not None and not carry.empty:

        aggregate_block(
            carry,
            feature_columns,
        )

        processed_blocks += 1

    print("\nRaw rows processed:", f"{total_rows:,}")
    print("Aggregation blocks:", f"{processed_blocks:,}")


# ---------------------------------------------------------
# SECOND PASS: BUILD CONTINUOUS NETWORK STATES
# ---------------------------------------------------------

def build_final_states(feature_columns):

    print("\nPASS 2: Creating continuous network states")

    aggregate_columns = [
        f"{feature}__{stat}"
        for feature in feature_columns
        for stat in ["mean", "std", "max"]
    ]

    output_columns = [
        "Timestamp",
        *aggregate_columns,
        "CaptureDay",
        "Flow Count",
        "Has Flows",
        "Label",
        "Segment ID",
    ]

    segment_id = 0
    previous_timestamp = None

    observed_count = 0
    generated_count = 0
    empty_count = 0

    buffer = []
    first_write = True

    def flush_buffer():

        nonlocal first_write

        if not buffer:
            return

        output = pd.DataFrame(
            buffer,
            columns=output_columns,
        )

        output.to_csv(
            OUTPUT_FILE,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )

        first_write = False
        buffer.clear()

    def make_empty_state(timestamp):

        state = {
            "Timestamp": timestamp,
        }

        for column in aggregate_columns:
            state[column] = 0.0

        state.update({
            "CaptureDay": timestamp.strftime("%Y-%m-%d"),
            "Flow Count": 0,
            "Has Flows": 0.0,
            "Label": "NO_FLOWS",
            "Segment ID": segment_id,
        })

        return state

    reader = pd.read_csv(
        OBSERVED_FILE,
        chunksize=2_000,
        low_memory=False,
    )

    for chunk in reader:

        for values in chunk.itertuples(
            index=False,
            name=None,
        ):

            record = dict(
                zip(chunk.columns, values)
            )

            timestamp = pd.Timestamp(
                record["Timestamp"]
            )

            if previous_timestamp is not None:

                gap = int(
                    (timestamp - previous_timestamp).total_seconds()
                )

                if gap <= 0:
                    raise ValueError(
                        "Duplicate or unordered observed timestamps."
                    )

                if gap > MAX_GAP_SECONDS:

                    # Start a new independent temporal segment.
                    segment_id += 1

                else:

                    # Fill only short internal gaps.
                    for seconds in range(1, gap):

                        missing_timestamp = (
                            previous_timestamp
                            + pd.Timedelta(seconds=seconds)
                        )

                        buffer.append(
                            make_empty_state(missing_timestamp)
                        )

                        generated_count += 1
                        empty_count += 1

            state = {
                "Timestamp": timestamp,
            }

            for column in aggregate_columns:
                state[column] = record[column]

            state.update({
                "CaptureDay": record["CaptureDay"],
                "Flow Count": record["Flow Count"],
                "Has Flows": 1.0,
                "Label": record["Label"],
                "Segment ID": segment_id,
            })

            buffer.append(state)

            observed_count += 1
            generated_count += 1

            previous_timestamp = timestamp

            if len(buffer) >= 1_000:
                flush_buffer()

    flush_buffer()

    return (
        observed_count,
        generated_count,
        empty_count,
        segment_id + 1,
        len(aggregate_columns),
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    validate_paths()

    print("=" * 60)
    print("SENTINELX MULTIDAY NETWORK STATE BUILDER")
    print("=" * 60)

    feature_columns = get_feature_columns()

    print("\nInput:", INPUT_FILE)
    print("Raw flow features:", len(feature_columns))
    print("Aggregate features:", len(feature_columns) * 3)
    print("Chunk size:", f"{CHUNK_SIZE:,}")
    print("Maximum gap:", MAX_GAP_SECONDS, "seconds")

    build_observed_aggregates(feature_columns)

    (
        observed,
        generated,
        empty,
        segments,
        aggregate_features,
    ) = build_final_states(feature_columns)

    report = [
        "SENTINELX MULTIDAY STATE BUILD REPORT",
        "=" * 45,
        f"Input file: {INPUT_FILE}",
        f"Output file: {OUTPUT_FILE}",
        f"Raw flow features: {len(feature_columns)}",
        f"Aggregate features: {aggregate_features}",
        f"Observed network states: {observed}",
        f"Generated network states: {generated}",
        f"Inserted empty states: {empty}",
        f"Independent temporal segments: {segments}",
        f"Maximum gap threshold: {MAX_GAP_SECONDS} seconds",
        "",
        "State generation completed.",
    ]

    REPORT_FILE.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("STATE BUILD SUMMARY")
    print("=" * 60)

    for line in report:
        print(line)

    print("\nIntermediate aggregates:", OBSERVED_FILE)
    print("Final states:", OUTPUT_FILE)
    print("Report:", REPORT_FILE)


if __name__ == "__main__":
    main()
