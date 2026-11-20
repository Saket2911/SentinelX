from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import pandas as pd

ROOT = Path(r"F:\Projects\SentinelX")
ARCHIVE = ROOT / "Datasets" / "archive.zip"
OUT = ROOT / "Datasets" / "diagnostics" / "additional_capture_scan_v1"

TARGET_FILES = [
    "02-16-2018.csv",
    "02-20-2018.csv",
    "02-21-2018.csv",
    "02-22-2018.csv",
    "02-23-2018.csv",
    "02-28-2018.csv",
    "03-01-2018.csv"
]

CHUNK_SIZE = 200_000

if not ARCHIVE.exists():
    raise FileNotFoundError(ARCHIVE)

if OUT.exists():
    raise FileExistsError(
        f"Output folder already exists: {OUT}. Inspect it before rerunning."
    )

OUT.mkdir(parents=True)

summary_rows = []
label_rows = []

with ZipFile(ARCHIVE, "r") as archive:
    members = {
        Path(item.filename).name: item
        for item in archive.infolist()
        if not item.is_dir()
    }

    for filename in TARGET_FILES:
        print("\n" + "=" * 75)
        print("SCANNING:", filename)
        print("=" * 75)

        if filename not in members:
            print("NOT FOUND — skipping")
            continue

        item = members[filename]
        print(f"Uncompressed size: {item.file_size / (1024**3):.3f} GB")

        with archive.open(item, "r") as raw:
            header = pd.read_csv(raw, nrows=0)

        columns = list(header.columns)

        if "Timestamp" not in columns or "Label" not in columns:
            print("REQUIRED COLUMNS NOT FOUND")
            print("Available columns:", columns)
            summary_rows.append({
                "Filename": filename,
                "Status": "MISSING_REQUIRED_COLUMNS",
                "Rows": 0
            })
            continue

        label_counts = Counter()
        total_rows = 0
        missing_labels = 0
        invalid_timestamps = 0
        min_timestamp = None
        max_timestamp = None

        with archive.open(item, "r") as raw:
            reader = pd.read_csv(
                raw,
                usecols=["Timestamp", "Label"],
                dtype={"Timestamp": "string", "Label": "string"},
                chunksize=CHUNK_SIZE,
                on_bad_lines="error"
            )

            for chunk_number, chunk in enumerate(reader, start=1):
                total_rows += len(chunk)

                missing_labels += int(chunk["Label"].isna().sum())

                labels = chunk["Label"].fillna("<MISSING>")
                label_counts.update(labels.value_counts().to_dict())

                timestamps = pd.to_datetime(
                    chunk["Timestamp"],
                    errors="coerce",
                    format="mixed"
                )

                invalid_timestamps += int(timestamps.isna().sum())
                valid = timestamps.dropna()

                if not valid.empty:
                    chunk_min = valid.min()
                    chunk_max = valid.max()

                    if min_timestamp is None or chunk_min < min_timestamp:
                        min_timestamp = chunk_min

                    if max_timestamp is None or chunk_max > max_timestamp:
                        max_timestamp = chunk_max

                print(
                    f"Processed chunk {chunk_number:,} | "
                    f"Rows scanned: {total_rows:,}"
                )

        for label, count in sorted(label_counts.items()):
            label_rows.append({
                "Filename": filename,
                "Label": label,
                "Flow_Count": int(count),
                "Percentage": (
                    100 * count / total_rows if total_rows else 0
                )
            })

        summary_rows.append({
            "Filename": filename,
            "Status": "SCANNED",
            "Rows": total_rows,
            "Missing_Labels": missing_labels,
            "Invalid_Timestamps": invalid_timestamps,
            "First_Timestamp": str(min_timestamp),
            "Last_Timestamp": str(max_timestamp),
            "Distinct_Labels": len(label_counts)
        })

        print("\nRows:", f"{total_rows:,}")
        print("Distinct labels:", len(label_counts))
        print("Missing labels:", missing_labels)
        print("Invalid timestamps:", invalid_timestamps)
        print("First timestamp:", min_timestamp)
        print("Last timestamp:", max_timestamp)
        print("Label counts:")

        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count:,}")

summary_df = pd.DataFrame(summary_rows)
labels_df = pd.DataFrame(label_rows)

summary_df.to_csv(OUT / "additional_capture_summary.csv", index=False)
labels_df.to_csv(OUT / "additional_capture_label_counts.csv", index=False)

report = [
    "SENTINELX — ADDITIONAL CIC-IDS-2018 CAPTURE SCAN",
    "=" * 75,
    "",
    "CAPTURE SUMMARY",
    "-" * 75,
    summary_df.to_string(index=False),
    "",
    "LABEL DISTRIBUTION",
    "-" * 75,
    labels_df.to_string(index=False),
    "",
    "NOTES",
    "-" * 75,
    "Files were streamed directly from archive.zip.",
    "No CSV files were extracted or modified.",
    "Only Timestamp and Label columns were read.",
    "This scan does not modify existing SentinelX datasets or models."
]

report_path = OUT / "additional_capture_scan_report.txt"
report_path.write_text("\n".join(report), encoding="utf-8")

print("\n" + "=" * 75)
print("ADDITIONAL CAPTURE SCAN COMPLETE")
print("=" * 75)
print("\nCapture summary:")
print(summary_df.to_string(index=False))
print("\nLabel counts:")
print(labels_df.to_string(index=False))
print("\nOutput folder:", OUT)
print("Report:", report_path)
