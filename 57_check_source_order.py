import zipfile
import pandas as pd
from pathlib import Path

ROOT = Path(r"F:\Projects\SentinelX")
ZIP_PATH = ROOT / "Datasets" / "archive.zip"

CAPTURES = [
    ("02-14-2018.csv", "2018-02-14"),
    ("02-15-2018.csv", "2018-02-15"),
    ("02-16-2018.csv", "2018-02-16"),
    ("02-20-2018.csv", "2018-02-20"),
    ("02-21-2018.csv", "2018-02-21"),
    ("02-22-2018.csv", "2018-02-22"),
    ("02-23-2018.csv", "2018-02-23"),
    ("02-28-2018.csv", "2018-02-28"),
    ("03-01-2018.csv", "2018-03-01"),
    ("03-02-2018.csv", "2018-03-02"),
]

CHUNK_SIZE = 200_000

with zipfile.ZipFile(ZIP_PATH) as archive:
    entries = {Path(x.filename).name: x for x in archive.infolist()}

    for filename, expected_date in CAPTURES:
        print(f"\nChecking {filename} ...")

        total = 0
        headers = 0
        invalid = 0
        wrong_date = 0
        violations = 0
        max_backward_seconds = 0
        previous = None
        first = None
        last = None

        with archive.open(entries[filename]) as raw:
            reader = pd.read_csv(
                raw,
                chunksize=CHUNK_SIZE,
                low_memory=False,
                dtype={"Timestamp": str},
            )

            for chunk in reader:
                total += len(chunk)

                header_mask = (
                    chunk["Timestamp"].astype(str).str.strip().eq("Timestamp")
                    | chunk["Label"].astype(str).str.strip().eq("Label")
                )
                headers += int(header_mask.sum())

                timestamps = pd.to_datetime(
                    chunk["Timestamp"],
                    format="%d/%m/%Y %H:%M:%S",
                    errors="coerce",
                )

                invalid_mask = timestamps.isna() & ~header_mask
                invalid += int(invalid_mask.sum())

                valid_mask = timestamps.notna() & ~header_mask
                valid_times = timestamps[valid_mask]

                expected = pd.Timestamp(expected_date).date()
                wrong_mask = valid_mask & (timestamps.dt.date != expected)
                wrong_date += int(wrong_mask.sum())

                valid_times = timestamps[
                    valid_mask & (timestamps.dt.date == expected)
                ]

                if len(valid_times) == 0:
                    continue

                if first is None:
                    first = valid_times.iloc[0]

                values = valid_times.tolist()

                if previous is not None and values[0] < previous:
                    violations += 1
                    delta = (previous - values[0]).total_seconds()
                    max_backward_seconds = max(
                        max_backward_seconds, int(delta)
                    )

                for a, b in zip(values[:-1], values[1:]):
                    if b < a:
                        violations += 1
                        delta = (a - b).total_seconds()
                        max_backward_seconds = max(
                            max_backward_seconds, int(delta)
                        )

                previous = values[-1]
                last = previous

        print(f"Rows scanned: {total:,}")
        print(f"Embedded headers: {headers:,}")
        print(f"Invalid timestamps: {invalid:,}")
        print(f"Wrong-date rows: {wrong_date:,}")
        print(f"Chronological violations: {violations:,}")
        print(f"Maximum backward jump (seconds): {max_backward_seconds:,}")
        print(f"First valid timestamp: {first}")
        print(f"Last valid timestamp: {last}")

        if violations == 0:
            print("RESULT: Source is chronologically ordered.")
        else:
            print("RESULT: Source requires sorting.")

print("\nTimestamp ordering scan finished.")
