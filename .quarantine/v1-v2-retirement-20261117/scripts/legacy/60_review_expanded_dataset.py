from pathlib import Path
import pandas as pd

BASE = Path(r"F:\Projects\SentinelX\Datasets")
SOURCE = BASE / "sentinelx_expanded_chronological_v1.csv"
REPORT = BASE / "sentinelx_expanded_sort_report_v1.txt"

print("=" * 75)
print("EXPANDED DATASET QUALITY REVIEW")
print("=" * 75)

print("\n--- Sorting report ---")
print(REPORT.read_text(encoding="utf-8"))

print("\n--- File size ---")
print(f"{SOURCE.stat().st_size / (1024**3):.3f} GB")

daily_rows = {}
daily_labels = {}
label_counts = {}
first_timestamp = None
last_timestamp = None
previous_timestamp = None
ordering_violations = 0
date_mismatches = 0
total_rows = 0

reader = pd.read_csv(
    SOURCE,
    chunksize=250_000,
    dtype={
        "Timestamp": "string",
        "CaptureDay": "string",
        "Label": "string",
    },
    low_memory=False,
)

for chunk in reader:
    total_rows += len(chunk)

    if first_timestamp is None:
        first_timestamp = chunk["Timestamp"].iloc[0]

    last_timestamp = chunk["Timestamp"].iloc[-1]

    # Date integrity
    mismatches = (
        chunk["Timestamp"].str.slice(0, 10) != chunk["CaptureDay"]
    )
    date_mismatches += int(mismatches.sum())

    # Daily rows
    daily_counts = chunk["CaptureDay"].value_counts()
    for day, count in daily_counts.items():
        daily_rows[day] = daily_rows.get(day, 0) + int(count)

    # Label counts
    counts = chunk["Label"].value_counts()
    for label, count in counts.items():
        label_counts[label] = label_counts.get(label, 0) + int(count)

    # Check ordering across chunk boundaries and within chunks
    timestamps = chunk["Timestamp"].tolist()

    if previous_timestamp is not None and timestamps[0] < previous_timestamp:
        ordering_violations += 1

    for a, b in zip(timestamps[:-1], timestamps[1:]):
        if b < a:
            ordering_violations += 1

    previous_timestamp = timestamps[-1]

print("\n--- Global integrity ---")
print(f"Total rows: {total_rows:,}")
print(f"First timestamp: {first_timestamp}")
print(f"Last timestamp: {last_timestamp}")
print(f"Chronological violations: {ordering_violations}")
print(f"CaptureDay mismatches: {date_mismatches}")

print("\n--- Rows per capture day ---")
for day in sorted(daily_rows):
    print(f"{day}: {daily_rows[day]:,}")

print("\n--- Raw flow labels ---")
for label, count in sorted(
    label_counts.items(), key=lambda x: x[1], reverse=True
):
    print(f"{label}: {count:,}")

print("\n--- Label count total ---")
print(sum(label_counts.values()))

print("\nQUALITY REVIEW COMPLETE")
