from pathlib import Path
from zipfile import ZipFile
import pandas as pd

ROOT = Path(r"F:\Projects\SentinelX")
ARCHIVE = ROOT / "Datasets" / "archive.zip"
OUT = ROOT / "Datasets" / "diagnostics" / "corrected_capture_validation_v1"

FILES = [
    "02-16-2018.csv",
    "02-20-2018.csv",
    "02-21-2018.csv",
    "02-22-2018.csv",
    "02-23-2018.csv",
    "02-28-2018.csv",
    "03-01-2018.csv"
]

if OUT.exists():
    raise FileExistsError(f"Output already exists: {OUT}")

OUT.mkdir(parents=True)

summary = []
anomalies = []

with ZipFile(ARCHIVE, "r") as archive:
    members = {
        Path(item.filename).name: item
        for item in archive.infolist()
        if not item.is_dir()
    }

    for filename in FILES:
        print("\nInspecting:", filename)

        if filename not in members:
            print("NOT FOUND")
            continue

        expected_date = pd.to_datetime(
            filename.replace(".csv", ""),
            format="%m-%d-%Y"
        ).date()

        total = 0
        headers = 0
        invalid = 0
        wrong_date = 0
        min_time = None
        max_time = None

        with archive.open(members[filename], "r") as raw:
            reader = pd.read_csv(
                raw,
                usecols=["Timestamp", "Label"],
                dtype={"Timestamp": "string", "Label": "string"},
                chunksize=200_000
            )

            for chunk in reader:
                total += len(chunk)

                header_mask = (
                    chunk["Timestamp"].str.strip().eq("Timestamp")
                    | chunk["Label"].str.strip().eq("Label")
                )
                headers += int(header_mask.sum())

                timestamps = pd.to_datetime(
                    chunk["Timestamp"],
                    format="%d/%m/%Y %H:%M:%S",
                    errors="coerce"
                )

                invalid_mask = timestamps.isna() & ~header_mask
                invalid += int(invalid_mask.sum())

                valid = timestamps.notna() & ~header_mask
                valid_times = timestamps[valid]

                wrong_mask = (
                    valid
                    & (timestamps.dt.date != expected_date)
                )
                wrong_date += int(wrong_mask.sum())

                for idx in chunk.index[invalid_mask][:5]:
                    anomalies.append({
                        "Filename": filename,
                        "Problem": "Invalid timestamp",
                        "Timestamp": str(chunk.at[idx, "Timestamp"]),
                        "Label": str(chunk.at[idx, "Label"])
                    })

                for idx in chunk.index[wrong_mask][:10]:
                    anomalies.append({
                        "Filename": filename,
                        "Problem": "Wrong capture date",
                        "Timestamp": str(chunk.at[idx, "Timestamp"]),
                        "Label": str(chunk.at[idx, "Label"])
                    })

                if not valid_times.empty:
                    chunk_min = valid_times.min()
                    chunk_max = valid_times.max()

                    if min_time is None or chunk_min < min_time:
                        min_time = chunk_min

                    if max_time is None or chunk_max > max_time:
                        max_time = chunk_max

        result = {
            "Filename": filename,
            "Expected_Date": str(expected_date),
            "Rows": total,
            "Embedded_Headers": headers,
            "Invalid_Timestamps": invalid,
            "Wrong_Date_Rows": wrong_date,
            "First_Valid_Timestamp": str(min_time),
            "Last_Valid_Timestamp": str(max_time),
            "Valid_Data_Rows": total - headers - invalid
        }

        summary.append(result)

        print("Rows:", f"{total:,}")
        print("Embedded headers:", headers)
        print("Invalid timestamps:", invalid)
        print("Wrong-date rows:", wrong_date)
        print("First:", min_time)
        print("Last:", max_time)

summary_df = pd.DataFrame(summary)
anomaly_df = pd.DataFrame(anomalies)

summary_df.to_csv(OUT / "corrected_capture_summary.csv", index=False)
anomaly_df.to_csv(OUT / "capture_timestamp_anomalies.csv", index=False)

report_path = OUT / "corrected_capture_validation_report.txt"

report_path.write_text(
    "CORRECTED CIC-IDS-2018 CAPTURE VALIDATION\n"
    + "=" * 70 + "\n\n"
    + "Timestamp parsing format: %d/%m/%Y %H:%M:%S\n\n"
    + summary_df.to_string(index=False)
    + "\n\nANOMALY EXAMPLES\n"
    + "=" * 70 + "\n"
    + anomaly_df.to_string(index=False),
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("CORRECTED CAPTURE VALIDATION COMPLETE")
print("=" * 70)
print(summary_df.to_string(index=False))
print("\nAnomaly examples:")
print(anomaly_df.to_string(index=False))
print("\nOutput:", OUT)
print("Report:", report_path)
