
from pathlib import Path
import csv
import heapq
import shutil
import sys

import pandas as pd

BASE = Path(r"F:\Projects\SentinelX\Datasets")

SOURCE = BASE / "sentinelx_multiday_raw.csv"
OUTPUT = BASE / "sentinelx_multiday_chronological_v2.csv"
REPORT = BASE / "sentinelx_multiday_sort_report_v2.txt"
WORK = BASE / "sentinelx_sort_work_v2"

TEMP_OUTPUT = BASE / "sentinelx_multiday_chronological_v2.partial"
TEMP_REPORT = BASE / "sentinelx_multiday_sort_report_v2.partial"

CHUNK_ROWS = 100_000
EXPECTED_ROWS = 3_145_720
EXPECTED_COLUMNS = 81


def main():
    # Check every path before creating anything.
    if not SOURCE.is_file():
        raise RuntimeError(f"Source file missing: {SOURCE}")

    for path in (OUTPUT, REPORT, WORK, TEMP_OUTPUT, TEMP_REPORT):
        if path.exists():
            raise RuntimeError(
                f"Path already exists. Refusing to overwrite: {path}"
            )

    WORK.mkdir()

    chunk_paths = []
    total_input = 0
    expected_header = None

    try:
        print("STEP 1: Reading and sorting chunks")

        reader = pd.read_csv(
            SOURCE,
            chunksize=CHUNK_ROWS,
            dtype={
                "Timestamp": "string",
                "CaptureDay": "string",
                "Label": "string",
            },
            low_memory=False,
        )

        for number, chunk in enumerate(reader):

            if expected_header is None:
                expected_header = list(chunk.columns)

                required = {"Timestamp", "CaptureDay", "Label"}

                if not required.issubset(expected_header):
                    raise RuntimeError("Required columns are missing")

                if len(expected_header) != EXPECTED_COLUMNS:
                    raise RuntimeError(
                        f"Expected {EXPECTED_COLUMNS} columns, "
                        f"found {len(expected_header)}"
                    )

            if list(chunk.columns) != expected_header:
                raise RuntimeError(
                    f"Column mismatch in chunk {number}"
                )

            # Validate timestamps.
            parsed = pd.to_datetime(
                chunk["Timestamp"],
                format="%Y-%m-%d %H:%M:%S",
                errors="coerce",
            )

            if parsed.isna().any():
                raise RuntimeError(
                    f"Invalid timestamp in chunk {number}"
                )

            # Compare ordinary Python string values.
            # This avoids the pandas 3.0 string-dtype .equals() issue.
            timestamp_days = parsed.dt.strftime(
                "%Y-%m-%d"
            ).tolist()

            capture_days = chunk["CaptureDay"].astype(str).tolist()

            mismatch = next(
                (
                    i for i, (a, b) in enumerate(
                        zip(timestamp_days, capture_days)
                    ) if a != b
                ),
                None,
            )

            if mismatch is not None:
                raise RuntimeError(
                    f"CaptureDay/Timestamp mismatch in chunk "
                    f"{number}, row {mismatch}"
                )

            # ISO timestamps sort correctly as strings.
            chunk.sort_values(
                ["CaptureDay", "Timestamp"],
                kind="mergesort",
                inplace=True,
            )

            chunk_path = WORK / f"chunk_{number:05d}.csv"

            chunk.to_csv(
                chunk_path,
                index=False,
                encoding="utf-8",
                lineterminator="\n",
            )

            chunk_paths.append(chunk_path)
            total_input += len(chunk)

            print(
                f"  Chunk {number:05d}: "
                f"{len(chunk):,} rows sorted"
            )

        if total_input != EXPECTED_ROWS:
            raise RuntimeError(
                f"Expected {EXPECTED_ROWS:,} input rows, "
                f"found {total_input:,}"
            )

        if not chunk_paths:
            raise RuntimeError("No chunks were created")

        print("\nSTEP 2: Merging sorted chunks")

        files = [
            path.open("r", newline="", encoding="utf-8")
            for path in chunk_paths
        ]

        output_rows = 0
        previous_key = None
        first_by_day = {}
        last_by_day = {}

        try:
            readers = [csv.DictReader(f) for f in files]
            header = readers[0].fieldnames

            if header != expected_header:
                raise RuntimeError("Chunk header mismatch")

            for r in readers:
                if r.fieldnames != header:
                    raise RuntimeError("Different chunk headers")

            def row_key(row):
                return (row["CaptureDay"], row["Timestamp"])

            heap = []

            for index, r in enumerate(readers):
                row = next(r, None)

                if row is not None:
                    heapq.heappush(
                        heap,
                        (row_key(row), index, row),
                    )

            with TEMP_OUTPUT.open(
                "x", newline="", encoding="utf-8"
            ) as out:

                writer = csv.DictWriter(
                    out,
                    fieldnames=header,
                    lineterminator="\n",
                )

                writer.writeheader()

                while heap:
                    key, index, row = heapq.heappop(heap)

                    if previous_key is not None and key < previous_key:
                        raise RuntimeError(
                            "Chronological ordering violation"
                        )

                    writer.writerow(row)

                    previous_key = key
                    output_rows += 1

                    day = row["CaptureDay"]
                    timestamp = row["Timestamp"]

                    first_by_day.setdefault(day, timestamp)
                    last_by_day[day] = timestamp

                    next_row = next(readers[index], None)

                    if next_row is not None:
                        heapq.heappush(
                            heap,
                            (row_key(next_row), index, next_row),
                        )

                    if output_rows % 500_000 == 0:
                        print(
                            f"  Merged {output_rows:,} rows"
                        )

        finally:
            for f in files:
                f.close()

        if output_rows != EXPECTED_ROWS:
            raise RuntimeError(
                f"Expected {EXPECTED_ROWS:,} output rows, "
                f"found {output_rows:,}"
            )

        print("\nSTEP 3: Validating merged output")

        checked_rows = 0
        last_key = None

        with TEMP_OUTPUT.open(
            "r", newline="", encoding="utf-8"
        ) as f:

            r = csv.DictReader(f)

            if r.fieldnames != expected_header:
                raise RuntimeError("Final output header mismatch")

            for row in r:
                key = (row["CaptureDay"], row["Timestamp"])

                if last_key is not None and key < last_key:
                    raise RuntimeError(
                        f"Output order error near row {checked_rows}"
                    )

                if row["Timestamp"][:10] != row["CaptureDay"]:
                    raise RuntimeError(
                        f"Date mismatch near row {checked_rows}"
                    )

                last_key = key
                checked_rows += 1

        if checked_rows != EXPECTED_ROWS:
            raise RuntimeError(
                f"Final validation expected {EXPECTED_ROWS:,}, "
                f"found {checked_rows:,}"
            )

        report_lines = [
            "SentinelX chronological sorting report",
            f"Source: {SOURCE}",
            f"Output: {OUTPUT}",
            f"Rows verified: {checked_rows:,}",
            f"Columns: {EXPECTED_COLUMNS}",
            "Sort keys: CaptureDay, Timestamp",
            "Invalid timestamps: 0",
            "CaptureDay/Timestamp mismatches: 0",
            "Chronological ordering violations: 0",
            "",
            "Per-day timestamp ranges:",
        ]

        for day in sorted(first_by_day):
            report_lines.append(
                f"{day}: {first_by_day[day]} "
                f"to {last_by_day[day]}"
            )

        with TEMP_REPORT.open("x", encoding="utf-8") as report:
            report.write("\n".join(report_lines) + "\n")

        # Publish final files only after validation succeeds.
        TEMP_OUTPUT.replace(OUTPUT)
        TEMP_REPORT.replace(REPORT)

        print("\nSORTING SUCCESSFUL")
        print(f"Rows verified: {checked_rows:,}")
        print(f"Output: {OUTPUT}")
        print(f"Report: {REPORT}")

        # Remove only this script's new temporary work folder.
        shutil.rmtree(WORK)

        print("New temporary chunk folder removed.")
        print("Original raw CSV and previous work folders preserved.")

    except Exception as exc:
        print("\nSORTING FAILED")
        print("Reason:", exc)
        print("Do not delete or rerun over partial files.")
        print("New work folder:", WORK)
        print("Temporary output:", TEMP_OUTPUT)
        print("Inspect the error before retrying.")
        sys.exit(1)


if __name__ == "__main__":
    main()
