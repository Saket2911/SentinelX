from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import pandas as pd

ROOT = Path(r"F:\Projects\SentinelX")
ARCHIVE = ROOT / "Datasets" / "archive.zip"
OUT = ROOT / "Datasets" / "diagnostics" / "timestamp_integrity_v1"

FILES = [
    "02-16-2018.csv",
    "02-22-2018.csv",
    "02-28-2018.csv",
    "03-01-2018.csv"
]

CHUNK_SIZE = 200_000

if OUT.exists():
    raise FileExistsError(f"Output already exists: {OUT}")

OUT.mkdir(parents=True)

def expected_date(filename):
    return pd.to_datetime(
        filename.replace(".csv", ""),
        format="%m-%d-%Y"
    ).date()

report_lines = [
    "SENTINELX — TIMESTAMP INTEGRITY INVESTIGATION",
    "=" * 75
]

with ZipFile(ARCHIVE, "r") as archive:
    members = {
        Path(item.filename).name: item
        for item in archive.infolist()
        if not item.is_dir()
    }

    for filename in FILES:
        print("\n" + "=" * 75)
        print("INSPECTING:", filename)
        print("=" * 75)

        if filename not in members:
            print("File not found.")
            continue

        expected = expected_date(filename)
        header_rows = 0
        invalid_timestamps = 0
        wrong_date_rows = 0
        total_rows = 0

        wrong_date_examples = []
        header_examples = []
        invalid_examples = []
        first_examples = []
        last_examples = []

        with archive.open(members[filename], "r") as raw:
            reader = pd.read_csv(
                raw,
                usecols=["Timestamp", "Label"],
                dtype={"Timestamp": "string", "Label": "string"},
                chunksize=CHUNK_SIZE,
                on_bad_lines="error"
            )

            for chunk in reader:
                total_rows += len(chunk)

                timestamps_raw = chunk["Timestamp"].fillna("")
                labels = chunk["Label"].fillna("")

                header_mask = (
                    labels.str.strip().str.lower() == "label"
                )
                header_rows += int(header_mask.sum())

                for idx in chunk.index[header_mask][:5]:
                    header_examples.append({
                        "Timestamp": str(chunk.at[idx, "Timestamp"]),
                        "Label": str(chunk.at[idx, "Label"])
                    })

                parsed = pd.to_datetime(
                    timestamps_raw,
                    errors="coerce",
                    format="mixed"
                )

                invalid_mask = parsed.isna() & ~header_mask
                invalid_timestamps += int(invalid_mask.sum())

                for idx in chunk.index[invalid_mask][:5]:
                    invalid_examples.append(
                        str(chunk.at[idx, "Timestamp"])
                    )

                date_mask = parsed.notna()
                parsed_dates = parsed.dt.date

                wrong_mask = (
                    date_mask
                    & (parsed_dates != expected)
                    & ~header_mask
                )

                wrong_date_rows += int(wrong_mask.sum())

                for idx in chunk.index[wrong_mask][:10]:
                    if len(wrong_date_examples) < 20:
                        wrong_date_examples.append({
                            "Raw_Timestamp": str(
                                chunk.at[idx, "Timestamp"]
                            ),
                            "Parsed_Timestamp": str(parsed.loc[idx]),
                            "Label": str(chunk.at[idx, "Label"])
                        })

                valid_indices = list(chunk.index[date_mask])

                for idx in valid_indices[:3]:
                    if len(first_examples) < 10:
                        first_examples.append(
                            str(chunk.at[idx, "Timestamp"])
                        )

                for idx in valid_indices[-3:]:
                    last_examples.append(
                        str(chunk.at[idx, "Timestamp"])
                    )

        last_examples = last_examples[-10:]

        print("Expected capture date:", expected)
        print("Total rows:", f"{total_rows:,}")
        print("Embedded header rows:", f"{header_rows:,}")
        print("Unparseable timestamps:", f"{invalid_timestamps:,}")
        print("Parseable timestamps on wrong date:", f"{wrong_date_rows:,}")

        print("\nEmbedded header examples:")
        for item in header_examples:
            print(" ", item)

        print("\nWrong-date timestamp examples:")
        for item in wrong_date_examples:
            print(" ", item)

        print("\nUnparseable timestamp examples:")
        for item in invalid_examples:
            print(" ", item)

        print("\nFirst raw timestamp examples:")
        for item in first_examples:
            print(" ", item)

        print("\nLast raw timestamp examples:")
        for item in last_examples:
            print(" ", item)

        report_lines.extend([
            "",
            "=" * 75,
            f"FILE: {filename}",
            "=" * 75,
            f"Expected date: {expected}",
            f"Total rows: {total_rows}",
            f"Embedded header rows: {header_rows}",
            f"Unparseable timestamps: {invalid_timestamps}",
            f"Parseable timestamps on wrong date: {wrong_date_rows}",
            "",
            "Embedded header examples:",
            str(header_examples),
            "",
            "Wrong-date timestamp examples:",
            str(wrong_date_examples),
            "",
            "Unparseable timestamp examples:",
            str(invalid_examples),
            "",
            "First timestamp examples:",
            str(first_examples),
            "",
            "Last timestamp examples:",
            str(last_examples)
        ])

report_path = OUT / "timestamp_integrity_report.txt"
report_path.write_text("\n".join(report_lines), encoding="utf-8")

print("\n" + "=" * 75)
print("TIMESTAMP INVESTIGATION COMPLETE")
print("=" * 75)
print("Report:", report_path)
print("Original archive was not modified.")
