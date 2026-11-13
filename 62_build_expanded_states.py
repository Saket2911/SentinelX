from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"F:\Projects\SentinelX")
DATA_DIR = PROJECT_DIR / "Datasets"

INPUT_FILE = DATA_DIR / "sentinelx_expanded_chronological_v1.csv"

OBSERVED_FILE = DATA_DIR / "sentinelx_expanded_observed_aggregates_v1.csv"
OUTPUT_FILE = DATA_DIR / "sentinelx_expanded_states_v1.csv"
REPORT_FILE = DATA_DIR / "sentinelx_expanded_states_report_v1.txt"

CHUNK_SIZE = 25_000
MAX_GAP_SECONDS = 8

CONSTANT_FEATURES = {
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Byts/b Avg",
    "Fwd Pkts/b Avg",
    "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg",
    "Bwd Pkts/b Avg",
    "Bwd Blk Rate Avg",
}

METADATA_COLUMNS = {"Timestamp", "CaptureDay", "Label"}


# ============================================================
# VALIDATION
# ============================================================

def validate_paths():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(INPUT_FILE)

    outputs = [OBSERVED_FILE, OUTPUT_FILE, REPORT_FILE]
    existing = [p for p in outputs if p.exists()]

    if existing:
        raise FileExistsError(
            "Output files already exist. Refusing to overwrite:\n"
            + "\n".join(map(str, existing))
        )


def get_feature_columns():
    header = pd.read_csv(INPUT_FILE, nrows=0).columns.tolist()

    features = [
        c for c in header
        if c not in METADATA_COLUMNS
        and c not in CONSTANT_FEATURES
    ]

    if len(features) != 70:
        raise ValueError(
            f"Expected 70 variable features, found {len(features)}."
        )

    return features


# ============================================================
# LABEL SUMMARY
# ============================================================

def summarize_label(labels):
    attacks = sorted({
        str(label).strip()
        for label in labels
        if str(label).strip().lower() != "benign"
    })

    if not attacks:
        return "Benign"

    if len(attacks) == 1:
        return attacks[0]

    return "MIXED_ATTACK"


# ============================================================
# PASS 1: AGGREGATE FLOW RECORDS BY TIMESTAMP
# ============================================================

def aggregate_block(frame, feature_columns):
    if frame.empty:
        return

    frame = frame.copy()

    frame[feature_columns] = frame[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    grouped = frame.groupby("Timestamp", sort=True)

    aggregated = grouped[feature_columns].agg(
        ["mean", "std", "max"]
    )

    aggregated.columns = [
        f"{feature}__{stat}"
        for feature, stat in aggregated.columns
    ]

    capture_days = grouped["CaptureDay"].first()
    expected_days = aggregated.index.strftime("%Y-%m-%d")

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


def build_observed_aggregates(feature_columns):
    print("\nPASS 1: Aggregating raw flow records")
    print("Chunk size:", f"{CHUNK_SIZE:,}")

    carry = None
    previous_timestamp = None
    total_rows = 0
    chunk_count = 0

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
        chunk_count = chunk_number
        total_rows += len(chunk)

        chunk["Timestamp"] = pd.to_datetime(
            chunk["Timestamp"],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )

        if chunk["Timestamp"].isna().any():
            raise ValueError(f"Invalid timestamp in chunk {chunk_number}")

        if chunk["CaptureDay"].isna().any():
            raise ValueError(f"Missing CaptureDay in chunk {chunk_number}")

        if chunk["Label"].isna().any():
            raise ValueError(f"Missing Label in chunk {chunk_number}")

        if not chunk["Timestamp"].is_monotonic_increasing:
            raise ValueError(f"Timestamp order violation in chunk {chunk_number}")

        if previous_timestamp is not None:
            if chunk["Timestamp"].iloc[0] < previous_timestamp:
                raise ValueError("Timestamp order violation across chunks")

        previous_timestamp = chunk["Timestamp"].iloc[-1]

        expected_days = chunk["Timestamp"].dt.strftime("%Y-%m-%d")
        if not chunk["CaptureDay"].eq(expected_days).all():
            raise ValueError(
                f"CaptureDay/Timestamp mismatch in chunk {chunk_number}"
            )

        for col in feature_columns:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)

        final_timestamp = chunk["Timestamp"].iloc[-1]

        complete = chunk[
            chunk["Timestamp"] < final_timestamp
        ]

        carry = chunk[
            chunk["Timestamp"] == final_timestamp
        ].copy()

        if not complete.empty:
            aggregate_block(complete, feature_columns)

        if chunk_number % 100 == 0:
            print(
                f"Chunks: {chunk_number:,} | "
                f"Raw rows read: {total_rows:,}"
            )

    if carry is not None and not carry.empty:
        aggregate_block(carry, feature_columns)

    print("Raw rows processed:", f"{total_rows:,}")
    print("Chunks processed:", f"{chunk_count:,}")


# ============================================================
# PASS 2: BUILD NETWORK STATES AND TEMPORAL SEGMENTS
# ============================================================

def build_final_states(feature_columns):
    print("\nPASS 2: Building timestamped network states")

    aggregate_columns = [
        f"{feature}__{stat}"
        for feature in feature_columns
        for stat in ("mean", "std", "max")
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
    previous_capture_day = None

    observed_count = 0
    generated_count = 0
    empty_count = 0
    segment_count = 0

    buffer = []
    first_write = True

    def flush_buffer():
        nonlocal first_write

        if not buffer:
            return

        output = pd.DataFrame(buffer, columns=output_columns)

        output.to_csv(
            OUTPUT_FILE,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )

        first_write = False
        buffer.clear()

    def make_empty_state(timestamp, capture_day, current_segment):
        state = {"Timestamp": timestamp}

        for col in aggregate_columns:
            state[col] = 0.0

        state.update({
            "CaptureDay": capture_day,
            "Flow Count": 0,
            "Has Flows": 0.0,
            "Label": "NO_FLOWS",
            "Segment ID": current_segment,
        })

        return state

    reader = pd.read_csv(
        OBSERVED_FILE,
        chunksize=2_000,
        low_memory=False,
    )

    for chunk in reader:
        for values in chunk.itertuples(index=False, name=None):
            record = dict(zip(chunk.columns, values))
            timestamp = pd.Timestamp(record["Timestamp"])
            capture_day = str(record["CaptureDay"])

            if previous_timestamp is None:
                segment_count = 1

            else:
                gap = int(
                    (timestamp - previous_timestamp).total_seconds()
                )

                if gap <= 0:
                    raise ValueError(
                        "Duplicate or unordered observed timestamps."
                    )

                same_capture_day = (
                    capture_day == previous_capture_day
                )

                if gap > MAX_GAP_SECONDS or not same_capture_day:
                    segment_id += 1
                    segment_count += 1

                else:
                    # Fill only short gaps within the same capture day.
                    for seconds in range(1, gap):
                        missing_timestamp = (
                            previous_timestamp
                            + pd.Timedelta(seconds=seconds)
                        )

                        buffer.append(
                            make_empty_state(
                                missing_timestamp,
                                capture_day,
                                segment_id,
                            )
                        )

                        generated_count += 1
                        empty_count += 1

            state = {"Timestamp": timestamp}

            for col in aggregate_columns:
                value = record[col]
                state[col] = (
                    float(value) if pd.notna(value) else np.nan
                )

            state.update({
                "CaptureDay": capture_day,
                "Flow Count": int(record["Flow Count"]),
                "Has Flows": 1.0,
                "Label": record["Label"],
                "Segment ID": segment_id,
            })

            buffer.append(state)

            observed_count += 1
            generated_count += 1

            previous_timestamp = timestamp
            previous_capture_day = capture_day

            if len(buffer) >= 1_000:
                flush_buffer()

    flush_buffer()

    return (
        observed_count,
        generated_count,
        empty_count,
        segment_count,
        len(aggregate_columns),
    )


# ============================================================
# MAIN
# ============================================================

def main():
    validate_paths()

    print("=" * 70)
    print("SENTINELX EXPANDED NETWORK STATE BUILDER")
    print("=" * 70)

    features = get_feature_columns()

    print("Input:", INPUT_FILE)
    print("Variable raw flow features:", len(features))
    print("Aggregate features:", len(features) * 3)
    print("Maximum short-gap fill:", MAX_GAP_SECONDS, "seconds")

    build_observed_aggregates(features)

    (
        observed,
        generated,
        empty,
        segments,
        aggregate_features,
    ) = build_final_states(features)

    report = [
        "SENTINELX EXPANDED STATE BUILD REPORT",
        "=" * 50,
        f"Input: {INPUT_FILE}",
        f"Observed aggregates: {OBSERVED_FILE}",
        f"Final states: {OUTPUT_FILE}",
        f"Raw variable flow features: {len(features)}",
        f"Aggregate feature count: {aggregate_features}",
        f"Observed network states: {observed:,}",
        f"Generated network states: {generated:,}",
        f"Inserted NO_FLOWS states: {empty:,}",
        f"Temporal segments: {segments:,}",
        f"Maximum gap threshold: {MAX_GAP_SECONDS} seconds",
        "",
        "State generation completed.",
    ]

    REPORT_FILE.write_text("\n".join(report), encoding="utf-8")

    print("\n" + "=" * 70)
    print("STATE BUILD SUMMARY")
    print("=" * 70)

    for line in report:
        print(line)

    print("\nReport saved:", REPORT_FILE)


if __name__ == "__main__":
    main()
