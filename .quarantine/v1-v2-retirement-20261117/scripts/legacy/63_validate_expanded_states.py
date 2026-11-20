from pathlib import Path
import pandas as pd

DATA = Path(r"F:\Projects\SentinelX\Datasets")
SOURCE = DATA / "sentinelx_expanded_states_v1.csv"
REPORT = DATA / "sentinelx_expanded_states_validation_v1.txt"

EXPECTED_ROWS = 281_849
EXPECTED_AGG_FEATURES = 210

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

header = pd.read_csv(SOURCE, nrows=0).columns.tolist()

metadata = {
    "Timestamp", "CaptureDay", "Label", "Segment ID"
}
aggregate_features = [
    c for c in header
    if c not in metadata
    and c not in {"Flow Count", "Has Flows"}
]

print("=" * 70)
print("SENTINELX EXPANDED STATE VALIDATION")
print("=" * 70)
print("Columns:", len(header))
print("Aggregate features:", len(aggregate_features))

if len(aggregate_features) != EXPECTED_AGG_FEATURES:
    raise ValueError(
        f"Expected {EXPECTED_AGG_FEATURES} aggregate features, "
        f"found {len(aggregate_features)}"
    )

total_rows = 0
empty_states = 0
observed_states = 0
ordering_errors = 0
capture_day_errors = 0
segment_errors = 0
empty_state_errors = 0
observed_state_errors = 0

label_counts = {}
segment_ids = set()
day_counts = {}

first_timestamp = None
last_timestamp = None

prev_timestamp = None
prev_day = None
prev_segment = None

reader = pd.read_csv(
    SOURCE,
    chunksize=25_000,
    low_memory=False,
    dtype={
        "Timestamp": "string",
        "CaptureDay": "string",
        "Label": "string",
    },
)

for chunk_no, chunk in enumerate(reader, start=1):
    total_rows += len(chunk)

    timestamps = pd.to_datetime(
        chunk["Timestamp"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )

    if timestamps.isna().any():
        raise ValueError(f"Invalid timestamps in chunk {chunk_no}")

    if first_timestamp is None:
        first_timestamp = timestamps.iloc[0]

    last_timestamp = timestamps.iloc[-1]

    expected_days = timestamps.dt.strftime("%Y-%m-%d")
    capture_mismatch = chunk["CaptureDay"].ne(expected_days)
    capture_day_errors += int(capture_mismatch.sum())

    counts = chunk["Label"].value_counts()
    for label, count in counts.items():
        label_counts[label] = label_counts.get(label, 0) + int(count)

    day_counts_chunk = chunk["CaptureDay"].value_counts()
    for day, count in day_counts_chunk.items():
        day_counts[day] = day_counts.get(day, 0) + int(count)

    segment_ids.update(chunk["Segment ID"].unique().tolist())

    has_flows = pd.to_numeric(chunk["Has Flows"], errors="coerce")
    flow_count = pd.to_numeric(chunk["Flow Count"], errors="coerce")

    empty_mask = has_flows.eq(0)
    observed_mask = has_flows.eq(1)

    empty_states += int(empty_mask.sum())
    observed_states += int(observed_mask.sum())

    empty_state_errors += int(
        (
            empty_mask
            & (
                chunk["Label"].ne("NO_FLOWS")
                | flow_count.ne(0)
            )
        ).sum()
    )

    observed_state_errors += int(
        (
            observed_mask
            & (
                chunk["Label"].eq("NO_FLOWS")
                | flow_count.le(0)
            )
        ).sum()
    )

    ts_list = timestamps.tolist()
    days = chunk["CaptureDay"].tolist()
    segs = chunk["Segment ID"].tolist()

    for i in range(len(chunk)):
        ts = ts_list[i]
        day = days[i]
        seg = segs[i]

        if prev_timestamp is not None:
            gap = (ts - prev_timestamp).total_seconds()

            if gap <= 0:
                ordering_errors += 1

            # Day changes must always start a new segment.
            if day != prev_day and seg == prev_segment:
                segment_errors += 1

            # Long gaps must always start a new segment.
            if gap > 8 and seg == prev_segment:
                segment_errors += 1

            # Short filled gaps should remain within same segment.
            if (
                0 < gap <= 8
                and day == prev_day
                and seg != prev_segment
            ):
                segment_errors += 1

        prev_timestamp = ts
        prev_day = day
        prev_segment = seg

    if chunk_no % 5 == 0:
        print(f"Chunks checked: {chunk_no:,} | Rows: {total_rows:,}")

print("\n--- Validation summary ---")
print("Rows:", f"{total_rows:,}")
print("Expected rows:", f"{EXPECTED_ROWS:,}")
print("First timestamp:", first_timestamp)
print("Last timestamp:", last_timestamp)
print("Unique segments:", len(segment_ids))
print("Observed states:", f"{observed_states:,}")
print("NO_FLOWS states:", f"{empty_states:,}")
print("Chronology errors:", ordering_errors)
print("CaptureDay mismatches:", capture_day_errors)
print("Segment boundary errors:", segment_errors)
print("NO_FLOWS consistency errors:", empty_state_errors)
print("Observed-state consistency errors:", observed_state_errors)

print("\n--- States per capture day ---")
for day in sorted(day_counts):
    print(f"{day}: {day_counts[day]:,}")

print("\n--- State labels ---")
for label, count in sorted(
    label_counts.items(),
    key=lambda x: x[1],
    reverse=True,
):
    print(f"{label}: {count:,}")

checks = {
    "row count": total_rows == EXPECTED_ROWS,
    "chronological order": ordering_errors == 0,
    "capture-day integrity": capture_day_errors == 0,
    "segment boundaries": segment_errors == 0,
    "NO_FLOWS consistency": empty_state_errors == 0,
    "observed state consistency": observed_state_errors == 0,
}

print("\n--- Overall result ---")
for name, passed in checks.items():
    print(f"{name}: {'PASS' if passed else 'FAIL'}")

report = "\n".join([
    "SentinelX Expanded State Validation",
    f"Rows: {total_rows}",
    f"Segments: {len(segment_ids)}",
    f"Observed states: {observed_states}",
    f"NO_FLOWS states: {empty_states}",
    f"Chronology errors: {ordering_errors}",
    f"CaptureDay mismatches: {capture_day_errors}",
    f"Segment boundary errors: {segment_errors}",
    f"NO_FLOWS errors: {empty_state_errors}",
    f"Observed-state errors: {observed_state_errors}",
    "",
    *[
        f"{name}: {'PASS' if passed else 'FAIL'}"
        for name, passed in checks.items()
    ],
])

REPORT.write_text(report, encoding="utf-8")
print("\nReport saved:", REPORT)

if not all(checks.values()):
    raise SystemExit("VALIDATION FAILED — inspect the errors above.")

print("\nVALIDATION PASSED")
