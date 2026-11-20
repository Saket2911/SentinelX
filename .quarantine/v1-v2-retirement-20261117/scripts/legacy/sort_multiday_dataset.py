
from pathlib import Path
import csv
import heapq
import shutil
import sys

import pandas as pd

BASE = Path(r"F:\Projects\SentinelX\Datasets")
SOURCE = BASE / "sentinelx_multiday_raw.csv"
OUTPUT = BASE / "sentinelx_multiday_chronological.csv"
REPORT = BASE / "sentinelx_multiday_sort_report.txt"
WORK = BASE / "sentinelx_sort_work_new"

CHUNK_ROWS = 100_000
EXPECTED_ROWS = 3_145_720
KEYS = ["CaptureDay", "Timestamp"]


def fail(message):
    raise RuntimeError(message)


def main():
    # Refuse to overwrite existing files or folders.
    if not SOURCE.is_file():
        fail(f"Source CSV not found: {SOURCE}")

    for path in (OUTPUT, REPORT, WORK):
        if path.exists():
            fail(f"Already exists; refusing to overwrite: {path}")

    WORK.mkdir()

    chunk_paths = []
    total_input = 0
    expected_columns = None

    try:
        print("Step 1/3: Reading and sorting CSV chunks...")

        reader = pd.read_csv(
            SOURCE,
            chunksize=CHUNK_ROWS,
            dtype={"Timestamp": "string", "CaptureDay": "string",
                   "Label": "string"},
            low_memory=False,
        )

        for number, chunk in enumerate(reader):
            if expected_columns is None:
                expected_columns = list(chunk.columns)
                required = {"Timestamp", "CaptureDay", "Label"}
                if not required.issubset(expected_columns):
                    fail(f"Required columns missing: {required - set(expected_columns)}")

                if len(expected_columns) != 81:
                    fail(f"Expected 81 columns, found {len(expected_columns)}")

            if list(chunk.columns) != expected_columns:
                fail(f"Column mismatch in chunk {number}")

            # Fail rather than silently dropping or repairing bad timestamps.
            parsed = pd.to_datetime(
                chunk["Timestamp"], format="%Y-%m-%d %H:%M:%S",
                errors="coerce",
            )
            if parsed.isna().any():
                fail(f"Invalid timestamps in input chunk {number}")

            parsed_day = parsed.dt.strftime("%Y-%m-%d")
            if not parsed_day.equals(chunk["CaptureDay"]):
                fail(f"CaptureDay/Timestamp mismatch in chunk {number}")

            chunk["_sort_timestamp"] = parsed
            chunk.sort_values(
                ["CaptureDay", "_sort_timestamp"],
                kind="mergesort",
                inplace=True,
            )
            chunk.drop(columns=["_sort_timestamp"], inplace=True)

            path = WORK / f"chunk_{number:05d}.csv"
            chunk.to_csv(path, index=False, encoding="utf-8",
                         lineterminator="\n")
            chunk_paths.append(path)
            total_input += len(chunk)

            print(f"  Chunk {number:05d}: {len(chunk):,} rows sorted")

        if total_input != EXPECTED_ROWS:
            fail(f"Expected {EXPECTED_ROWS:,} input rows, got {total_input:,}")

        if not chunk_paths:
            fail("No data chunks were produced")

        print("Step 2/3: Merging sorted chunks...")

        # Each chunk is sorted. Merge using (CaptureDay, Timestamp).
        files = [path.open("r", newline="", encoding="utf-8")
                 for path in chunk_paths]
        writers = None
        output_rows = 0
        previous_key = None

        try:
            readers = [csv.DictReader(f) for f in files]
            header = readers[0].fieldnames

            if header != expected_columns:
                fail("Chunk header does not match original header")

            if any(r.fieldnames != header for r in readers):
                fail("Chunk headers differ")

            def row_key(row):
                return (row["CaptureDay"], row["Timestamp"])

            heap = []
            for index, reader in enumerate(readers):
                row = next(reader, None)
                if row is not None:
                    heapq.heappush(heap, (row_key(row), index, row))

            with OUTPUT.open("x", newline="", encoding="utf-8") as out:
                writer = csv.DictWriter(
                    out, fieldnames=header, lineterminator="\n"
                )
                writer.writeheader()

                while heap:
                    key, index, row = heapq.heappop(heap)

                    if previous_key is not None and key < previous_key:
                        fail("Chronological order violation during merge")

                    writer.writerow(row)
                    previous_key = key
                    output_rows += 1

                    next_row = next(readers[index], None)
                    if next_row is not None:
                        heapq.heappush(
                            heap, (row_key(next_row), index, next_row)
                        )

                    if output_rows % 500_000 == 0:
                        print(f"  Merged {output_rows:,} rows")

        finally:
            for f in files:
                f.close()

        if output_rows != EXPECTED_ROWS:
            fail(f"Expected {EXPECTED_ROWS:,} output rows, got {output_rows:,}")

        print("Step 3/3: Validating final output...")

        # Verify final row count and ordering by streaming the output.
        checked_rows = 0
        last_key = None
        first_by_day = {}
        last_by_day = {}
        output_columns = None

        for frame in pd.read_csv(
            OUTPUT,
            chunksize=CHUNK_ROWS,
            dtype={"Timestamp": "string", "CaptureDay": "string",
                   "Label": "string"},
            low_memory=False,
        ):
            if output_columns is None:
                output_columns = list(frame.columns)
            elif list(frame.columns) != output_columns:
                fail("Output columns changed between chunks")

            timestamps = pd.to_datetime(
                frame["Timestamp"], format="%Y-%m-%d %H:%M:%S",
                errors="coerce",
            )
            if timestamps.isna().any():
                fail("Output contains invalid timestamps")

            if not timestamps.dt.strftime("%Y-%m-%d").equals(frame["CaptureDay"]):
                fail("Output contains CaptureDay/Timestamp mismatches")

            for day, timestamp in zip(frame["CaptureDay"], frame["Timestamp"]):
                key = (day, timestamp)
                if last_key is not None and key < last_key:
                    fail(f"Output is not chronologically sorted near row {checked_rows}")
                last_key = key
                first_by_day.setdefault(day, timestamp)
                last_by_day[day] = timestamp
                checked_rows += 1

        if checked_rows != EXPECTED_ROWS:
            fail(f"Validation expected {EXPECTED_ROWS:,}, got {checked_rows:,}")

        report_text = [
            "SentinelX chronological dataset sort report",
            f"Source: {SOURCE}",
            f"Output: {OUTPUT}",
            f"Rows verified: {checked_rows:,}",
            f"Columns verified: {len(output_columns or [])}",
            "Sort keys: CaptureDay, Timestamp",
            "Timestamp format: YYYY-MM-DD HH:MM:SS",
            "CaptureDay/Timestamp mismatches: 0",
            "Chronological order violations: 0",
            "",
            "Per-day timestamp ranges:",
        ]
        for day in sorted(first_by_day):
            report_text.append(
                f"{day}: {first_by_day[day]} -> {last_by_day[day]}"
            )

        # Exclusive create; never overwrite an existing report.
        with REPORT.open("x", encoding="utf-8") as report:
            report.write("\n".join(report_text) + "\n")

        print("\nSUCCESS")
        print(f"Verified rows: {checked_rows:,}")
        print(f"Output: {OUTPUT}")
        print(f"Report: {REPORT}")
        print("The original raw CSV has not been changed.")

        # Remove only the newly created work folder, after full validation.
        shutil.rmtree(WORK)
        print("Temporary new sorting folder removed after validation.")

    except Exception as exc:
        print("\nSORTING DID NOT COMPLETE SUCCESSFULLY.")
        print(f"Reason: {exc}")
        print(f"Any partial output, if created, has been preserved: {OUTPUT}")
        print(f"New work files have been preserved: {WORK}")
        print("Do not delete these files; inspect the error before retrying.")
        sys.exit(1)


if __name__ == "__main__":
    main()
