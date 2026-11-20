from pathlib import Path
import pandas as pd

ROOT = Path(r"F:\Projects\SentinelX")

RAW_FILE = (
    ROOT / "Datasets"
    / "sentinelx_multiday_chronological_v2.csv"
)

if not RAW_FILE.exists():
    raise FileNotFoundError(RAW_FILE)

# Inspect selected intervals around the label transitions.
intervals = [
    (
        pd.Timestamp("2018-03-02 10:13:20"),
        pd.Timestamp("2018-03-02 10:13:45")
    ),
    (
        pd.Timestamp("2018-03-02 12:13:05"),
        pd.Timestamp("2018-03-02 12:13:42")
    )
]

counts = {}

for start, end in intervals:
    for timestamp in pd.date_range(start, end, freq="1s"):
        counts[timestamp] = {}

print("=" * 65)
print("SENTINELX - ORIGINAL FLOW LABEL INSPECTION")
print("=" * 65)

print("Reading original flow records in chunks...")

chunksize = 200_000
rows_scanned = 0

for chunk in pd.read_csv(
    RAW_FILE,
    usecols=["Timestamp", "Label"],
    chunksize=chunksize
):
    rows_scanned += len(chunk)

    chunk["Timestamp"] = pd.to_datetime(
        chunk["Timestamp"], errors="coerce"
    )

    chunk = chunk.dropna(subset=["Timestamp"])

    for start, end in intervals:
        selected = chunk[
            (chunk["Timestamp"] >= start)
            & (chunk["Timestamp"] < end + pd.Timedelta(seconds=1))
        ]

        if selected.empty:
            continue

        grouped = (
            selected.groupby(
                [
                    selected["Timestamp"].dt.floor("1s"),
                    "Label"
                ]
            )
            .size()
        )

        for (timestamp, label), count in grouped.items():
            counts[timestamp][str(label)] = (
                counts[timestamp].get(str(label), 0) + int(count)
            )

    if rows_scanned % 1_000_000 < chunksize:
        print(f"Scanned {rows_scanned:,} rows...")

print("\n" + "=" * 65)
print("FLOW COUNTS PER SECOND")
print("=" * 65)

for start, end in intervals:
    print(f"\nInterval: {start} to {end}")

    for timestamp in pd.date_range(start, end, freq="1s"):
        label_counts = counts[timestamp]
        total = sum(label_counts.values())

        print(
            timestamp,
            "| Total flows:", total,
            "|", label_counts
        )

print("\nTotal raw rows scanned:", f"{rows_scanned:,}")
print("Inspection completed.")
