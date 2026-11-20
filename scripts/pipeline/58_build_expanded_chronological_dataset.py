from pathlib import Path
import zipfile
import csv
import heapq
import shutil
import sys

import pandas as pd

BASE = Path(r"F:\Projects\SentinelX\Datasets")
ZIP_PATH = BASE / "archive.zip"

OUTPUT = BASE / "sentinelx_expanded_chronological_v1.csv"
REPORT = BASE / "sentinelx_expanded_sort_report_v1.txt"
WORK = BASE / "sentinelx_expanded_sort_work_v1"

TEMP_OUTPUT = BASE / "sentinelx_expanded_chronological_v1.partial"
TEMP_REPORT = BASE / "sentinelx_expanded_sort_report_v1.partial"

CHUNK_ROWS = 100_000
EXPECTED_TOTAL_ROWS = 16_232_929

CAPTURES = [
    ("02-14-2018.csv", "2018-02-14", 1_048_570),
    ("02-15-2018.csv", "2018-02-15", 1_048_575),
    ("02-16-2018.csv", "2018-02-16", 1_048_574),
    ("02-20-2018.csv", "2018-02-20", 7_948_748),
    ("02-21-2018.csv", "2018-02-21", 1_048_575),
    ("02-22-2018.csv", "2018-02-22", 1_048_566),
    ("02-23-2018.csv", "2018-02-23", 1_048_575),
    ("02-28-2018.csv", "2018-02-28", 613_071),
    ("03-01-2018.csv", "2018-03-01", 331_100),
    ("03-02-2018.csv", "2018-03-02", 1_048_575),
]


def main():
    for path in (OUTPUT, REPORT, WORK, TEMP_OUTPUT, TEMP_REPORT):
        if path.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing path: {path}"
            )

    if not ZIP_PATH.is_file():
        raise RuntimeError(f"Archive missing: {ZIP_PATH}")

    WORK.mkdir()

    chunk_paths = []
    source_stats = {}
    expected_columns = None
    chunk_number = 0

    try:
        print("=" * 75)
        print("SENTINELX EXPANDED DATASET BUILDER")
        print("=" * 75)

        with zipfile.ZipFile(ZIP_PATH) as archive:
            archive_entries = {
                Path(info.filename).name: info.filename
                for info in archive.infolist()
                if info.filename.lower().endswith(".csv")
            }

            missing = [
                filename for filename, _, _ in CAPTURES
                if filename not in archive_entries
            ]

            if missing:
                raise RuntimeError(
                    f"Archive is missing capture files: {missing}"
                )

            # Use Feb 14 as the canonical common 80-column schema.
            with archive.open(archive_entries["02-14-2018.csv"]) as raw:
                reference = pd.read_csv(
                    raw,
                    nrows=0,
                    encoding="utf-8-sig",
                )

            expected_columns = list(reference.columns)

            if len(expected_columns) != 80:
                raise RuntimeError(
                    f"Expected 80 common columns, "
                    f"found {len(expected_columns)}"
                )

            if not {"Timestamp", "Label"}.issubset(expected_columns):
                raise RuntimeError(
                    "Canonical schema missing Timestamp or Label"
                )

            print(f"Canonical source columns: {len(expected_columns)}")
            print("February 20 extra columns will be excluded.")

            print("\nSTEP 1 — Read captures, clean, normalize and sort chunks")

            for filename, expected_day, expected_rows in CAPTURES:
                print(f"\nProcessing {filename} ({expected_day})")

                raw_rows = 0
                header_rows = 0
                invalid_timestamps = 0
                wrong_date_rows = 0
                kept_rows = 0

                with archive.open(archive_entries[filename]) as raw:
                    reader = pd.read_csv(
                        raw,
                        chunksize=CHUNK_ROWS,
                        encoding="utf-8-sig",
                        dtype={
                            "Timestamp": "string",
                            "Label": "string",
                        },
                        low_memory=False,
                    )

                    for chunk in reader:
                        raw_rows += len(chunk)

                        missing_columns = [
                            c for c in expected_columns
                            if c not in chunk.columns
                        ]

                        if missing_columns:
                            raise RuntimeError(
                                f"{filename}: missing columns "
                                f"{missing_columns}"
                            )

                        # Normalize all captures to the canonical schema.
                        chunk = chunk[expected_columns].copy()

                        # Remove embedded repeated header records.
                        header_mask = (
                            chunk["Timestamp"].astype(str).str.strip()
                            .eq("Timestamp")
                            |
                            chunk["Label"].astype(str).str.strip()
                            .eq("Label")
                        )

                        header_rows += int(header_mask.sum())

                        chunk = chunk.loc[~header_mask].copy()

                        timestamps = pd.to_datetime(
                            chunk["Timestamp"],
                            format="%d/%m/%Y %H:%M:%S",
                            errors="coerce",
                        )

                        invalid_mask = timestamps.isna()
                        invalid_timestamps += int(invalid_mask.sum())

                        valid_mask = timestamps.notna()
                        expected_date = pd.Timestamp(expected_day).date()

                        wrong_date_mask = (
                            valid_mask
                            & (timestamps.dt.date != expected_date)
                        )

                        wrong_date_rows += int(wrong_date_mask.sum())

                        keep_mask = valid_mask & ~wrong_date_mask

                        chunk = chunk.loc[keep_mask].copy()
                        timestamps = timestamps.loc[keep_mask]

                        if chunk.empty:
                            continue

                        # Normalize timestamps into ISO format.
                        chunk["Timestamp"] = timestamps.dt.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )

                        # CaptureDay is appended as metadata.
                        chunk["CaptureDay"] = expected_day

                        chunk.sort_values(
                            ["Timestamp", "CaptureDay"],
                            kind="mergesort",
                            inplace=True,
                        )

                        chunk_path = WORK / (
                            f"chunk_{chunk_number:06d}.csv"
                        )

                        chunk.to_csv(
                            chunk_path,
                            index=False,
                            encoding="utf-8",
                            lineterminator="\n",
                        )

                        chunk_paths.append(chunk_path)

                        kept = len(chunk)
                        kept_rows += kept
                        chunk_number += 1

                        print(
                            f"  Chunk {chunk_number:06d}: "
                            f"{kept:,} valid rows"
                        )

                source_stats[filename] = {
                    "expected_day": expected_day,
                    "raw_rows": raw_rows,
                    "headers": header_rows,
                    "invalid_timestamps": invalid_timestamps,
                    "wrong_date_rows": wrong_date_rows,
                    "kept_rows": kept_rows,
                }

                print(f"  Raw rows: {raw_rows:,}")
                print(f"  Embedded headers: {header_rows:,}")
                print(f"  Invalid timestamps: {invalid_timestamps:,}")
                print(f"  Wrong-date rows: {wrong_date_rows:,}")
                print(f"  Kept rows: {kept_rows:,}")

                if kept_rows != expected_rows:
                    raise RuntimeError(
                        f"{filename}: expected {expected_rows:,} "
                        f"valid rows, found {kept_rows:,}"
                    )

        total_kept = sum(
            stats["kept_rows"]
            for stats in source_stats.values()
        )

        if total_kept != EXPECTED_TOTAL_ROWS:
            raise RuntimeError(
                f"Expected total {EXPECTED_TOTAL_ROWS:,}, "
                f"found {total_kept:,}"
            )

        if not chunk_paths:
            raise RuntimeError("No sorted chunks were produced")

        print("\nSTEP 2 — Merge sorted chunks")

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

            if len(header) != 81:
                raise RuntimeError(
                    f"Expected 81 output columns, got {len(header)}"
                )

            if header[-1] != "CaptureDay":
                raise RuntimeError(
                    "CaptureDay should be the final output column"
                )

            for reader in readers:
                if reader.fieldnames != header:
                    raise RuntimeError("Chunk headers differ")

            def row_key(row):
                return (row["Timestamp"], row["CaptureDay"])

            heap = []

            for index, reader in enumerate(readers):
                row = next(reader, None)

                if row is not None:
                    heapq.heappush(
                        heap,
                        (row_key(row), index, row),
                    )

            with TEMP_OUTPUT.open(
                "x", newline="", encoding="utf-8"
            ) as output_file:

                writer = csv.DictWriter(
                    output_file,
                    fieldnames=header,
                    lineterminator="\n",
                )

                writer.writeheader()

                while heap:
                    key, index, row = heapq.heappop(heap)

                    if previous_key is not None and key < previous_key:
                        raise RuntimeError(
                            "Chronological violation during merge"
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

                    if output_rows % 1_000_000 == 0:
                        print(f"  Merged {output_rows:,} rows")

        finally:
            for file in files:
                file.close()

        if output_rows != EXPECTED_TOTAL_ROWS:
            raise RuntimeError(
                f"Expected {EXPECTED_TOTAL_ROWS:,} merged rows, "
                f"found {output_rows:,}"
            )

        print("\nSTEP 3 — Verify final output")

        checked_rows = 0
        previous_key = None

        with TEMP_OUTPUT.open(
            "r", newline="", encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            if reader.fieldnames != header:
                raise RuntimeError("Final header mismatch")

            for row in reader:
                key = (row["Timestamp"], row["CaptureDay"])

                if previous_key is not None and key < previous_key:
                    raise RuntimeError(
                        f"Final ordering error at row {checked_rows}"
                    )

                if row["Timestamp"][:10] != row["CaptureDay"]:
                    raise RuntimeError(
                        f"CaptureDay mismatch at row {checked_rows}"
                    )

                previous_key = key
                checked_rows += 1

        if checked_rows != EXPECTED_TOTAL_ROWS:
            raise RuntimeError(
                f"Final row count mismatch: {checked_rows:,}"
            )

        report_lines = [
            "SentinelX Expanded Chronological Dataset Report",
            "=" * 65,
            f"Archive: {ZIP_PATH}",
            f"Output: {OUTPUT}",
            f"Total rows: {checked_rows:,}",
            "Columns: 81 (80 common source columns + CaptureDay)",
            "Sort keys: Timestamp, CaptureDay",
            "Timestamp format: YYYY-MM-DD HH:MM:SS",
            "Chronological violations: 0",
            "CaptureDay mismatches: 0",
            "",
            "Per-capture source quality:",
        ]

        for filename, _, _ in CAPTURES:
            stats = source_stats[filename]

            report_lines.append(
                f"{filename}: "
                f"raw={stats['raw_rows']:,}, "
                f"headers={stats['headers']:,}, "
                f"invalid_ts={stats['invalid_timestamps']:,}, "
                f"wrong_date={stats['wrong_date_rows']:,}, "
                f"kept={stats['kept_rows']:,}"
            )

        report_lines.extend([
            "",
            "Per-day chronological timestamp ranges:",
        ])

        for day in sorted(first_by_day):
            report_lines.append(
                f"{day}: {first_by_day[day]} "
                f"to {last_by_day[day]}"
            )

        with TEMP_REPORT.open("x", encoding="utf-8") as report:
            report.write("\n".join(report_lines) + "\n")

        # Publish only after all validations pass.
        TEMP_OUTPUT.replace(OUTPUT)
        TEMP_REPORT.replace(REPORT)

        print("\n" + "=" * 75)
        print("EXPANDED DATASET BUILD SUCCESSFUL")
        print("=" * 75)
        print(f"Rows: {checked_rows:,}")
        print(f"Output: {OUTPUT}")
        print(f"Report: {REPORT}")

        shutil.rmtree(WORK)
        print("Temporary chunk folder removed.")

    except Exception as exc:
        print("\nBUILD FAILED")
        print("Reason:", exc)
        print("Do not delete partial files or work folders yet.")
        print("Work folder:", WORK)
        print("Partial output:", TEMP_OUTPUT)
        sys.exit(1)


if __name__ == "__main__":
    main()
