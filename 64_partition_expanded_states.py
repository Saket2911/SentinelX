import os
import pandas as pd
import numpy as np

ROOT = r"F:\Projects\SentinelX\Datasets"
INPUT = os.path.join(ROOT, "sentinelx_expanded_states_v1.csv")

OUTPUTS = {
    "train": os.path.join(ROOT, "sentinelx_expanded_train_v1.csv"),
    "validation": os.path.join(ROOT, "sentinelx_expanded_validation_v1.csv"),
    "test": os.path.join(ROOT, "sentinelx_expanded_test_v1.csv"),
}

REPORT = os.path.join(ROOT, "sentinelx_expanded_partition_report_v1.txt")
CHUNK_SIZE = 25000

# Prevent accidental overwrite
for path in list(OUTPUTS.values()) + [REPORT]:
    if os.path.exists(path):
        raise FileExistsError(
            f"Output already exists: {path}\n"
            "Rename/remove it intentionally before rerunning."
        )

print("=" * 70)
print("SENTINELX CHRONOLOGICAL PARTITIONING")
print("=" * 70)

# ---------------------------------------------------------
# PASS 1: Find segment start positions without loading all data
# ---------------------------------------------------------
print("\nPass 1: Scanning segment boundaries...")

segment_starts = []
total_rows = 0
previous_segment = None

for chunk in pd.read_csv(
    INPUT,
    usecols=["Segment ID"],
    chunksize=CHUNK_SIZE
):
    for value in chunk["Segment ID"].to_numpy():
        if previous_segment is None or value != previous_segment:
            segment_starts.append(total_rows)
            previous_segment = value
        total_rows += 1

print(f"Total rows: {total_rows:,}")
print(f"Segments: {len(segment_starts):,}")

target_train = int(total_rows * 0.70)
target_val = int(total_rows * 0.85)

# Choose segment start at or after target row
train_end = next(
    (x for x in segment_starts if x >= target_train),
    None
)
val_end = next(
    (x for x in segment_starts if x >= target_val),
    None
)

if train_end is None or val_end is None or train_end >= val_end:
    raise RuntimeError("Could not find valid segment-aligned split boundaries.")

print(f"\nTarget train boundary: {target_train:,}")
print(f"Actual train boundary: {train_end:,}")
print(f"Target validation boundary: {target_val:,}")
print(f"Actual validation boundary: {val_end:,}")

# ---------------------------------------------------------
# PASS 2: Stream full state data into three partition files
# ---------------------------------------------------------
print("\nPass 2: Writing partition files...")

row_start = 0
written = {"train": 0, "validation": 0, "test": 0}
first_write = {"train": True, "validation": True, "test": True}

for chunk in pd.read_csv(INPUT, chunksize=CHUNK_SIZE):
    row_end = row_start + len(chunk)

    ranges = {
        "train": (0, train_end),
        "validation": (train_end, val_end),
        "test": (val_end, total_rows),
    }

    for name, (start, end) in ranges.items():
        left = max(row_start, start)
        right = min(row_end, end)

        if left >= right:
            continue

        part = chunk.iloc[left - row_start:right - row_start]

        part.to_csv(
            OUTPUTS[name],
            mode="w" if first_write[name] else "a",
            index=False,
            header=first_write[name]
        )

        first_write[name] = False
        written[name] += len(part)

    row_start = row_end

    if row_start % 100000 < CHUNK_SIZE:
        print(f"Processed {row_start:,}/{total_rows:,} rows")

# ---------------------------------------------------------
# PASS 3: Verify row counts and chronology
# ---------------------------------------------------------
print("\nPass 3: Checking partition outputs...")

report_lines = [
    "SENTINELX CHRONOLOGICAL PARTITION REPORT",
    "=" * 60,
    f"Input rows: {total_rows:,}",
    f"Segment count: {len(segment_starts):,}",
    f"Train boundary row: {train_end:,}",
    f"Validation boundary row: {val_end:,}",
    "",
]

for name, path in OUTPUTS.items():
    count = sum(1 for _ in open(path, "r", encoding="utf-8")) - 1
    print(f"{name.capitalize()} rows: {count:,}")

    if count != written[name]:
        raise RuntimeError(
            f"Row-count mismatch in {name}: "
            f"expected {written[name]}, got {count}"
        )

    # Check first and last timestamps in each partition
    first = pd.read_csv(path, nrows=1, usecols=["Timestamp"])
    last = None
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, usecols=["Timestamp"]):
        last = chunk["Timestamp"].iloc[-1]

    first_time = first["Timestamp"].iloc[0]
    print(f"  First timestamp: {first_time}")
    print(f"  Last timestamp:  {last}")

    report_lines.extend([
        f"{name.upper()} rows: {count:,}",
        f"First timestamp: {first_time}",
        f"Last timestamp: {last}",
        f"File: {path}",
        "",
    ])

if sum(written.values()) != total_rows:
    raise RuntimeError("Total partition row count does not equal input row count.")

report_lines.extend([
    "Partition row total matches input: PASS",
    "Segment-aligned boundaries: PASS",
])

with open(REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("\n" + "=" * 70)
print("PARTITIONING COMPLETED")
print("=" * 70)
print(f"Report: {REPORT}")
print("Train / validation / test files created.")
