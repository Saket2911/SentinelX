
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Datasets"

FILES = [
    DATA / "sentinelx_multiday_raw.csv",
    DATA / "sentinelx_multiday_chronological_v2.csv",
]

COLUMN_A = "CWE Flag Count"
COLUMN_B = "Fwd URG Flags"

CHUNK_SIZE = 250_000


def normalize(name):
    return "".join(
        char.lower()
        for char in str(name)
        if char.isalnum()
    )


def audit_file(path):
    print("\n" + "=" * 75)
    print(f"FILE: {path.name}")
    print("=" * 75)

    if not path.exists():
        print(f"NOT FOUND: {path}")
        return

    header = pd.read_csv(path, nrows=0).columns.tolist()

    normalized = {
        normalize(column): column
        for column in header
    }

    key_a = normalize(COLUMN_A)
    key_b = normalize(COLUMN_B)

    if key_a not in normalized or key_b not in normalized:
        print("Required original feature columns not found.")
        print("Available matching columns:")

        for column in header:
            if "cwe" in normalize(column) or "urg" in normalize(column):
                print(" ", repr(column))

        return

    actual_a = normalized[key_a]
    actual_b = normalized[key_b]

    print(f"Column A: {actual_a}")
    print(f"Column B: {actual_b}")

    total_rows = 0
    matching_rows = 0
    mismatching_rows = 0
    both_null_rows = 0
    one_null_rows = 0

    unique_a = set()
    unique_b = set()

    mismatch_examples = []

    for chunk in pd.read_csv(
        path,
        usecols=[actual_a, actual_b],
        chunksize=CHUNK_SIZE,
    ):
        a = pd.to_numeric(chunk[actual_a], errors="coerce")
        b = pd.to_numeric(chunk[actual_b], errors="coerce")

        a_null = a.isna()
        b_null = b.isna()

        both_null = a_null & b_null
        one_null = a_null ^ b_null

        equal = a.eq(b) | both_null
        different = ~equal

        total_rows += len(chunk)
        matching_rows += int(equal.sum())
        mismatching_rows += int(different.sum())
        both_null_rows += int(both_null.sum())
        one_null_rows += int(one_null.sum())

        unique_a.update(a.dropna().unique().tolist())
        unique_b.update(b.dropna().unique().tolist())

        if len(mismatch_examples) < 10 and different.any():
            examples = pd.DataFrame({
                COLUMN_A: a[different],
                COLUMN_B: b[different],
            }).head(10 - len(mismatch_examples))

            mismatch_examples.extend(
                examples.to_dict("records")
            )

    print("\nRESULTS")
    print(f"Rows inspected: {total_rows:,}")
    print(f"Matching rows: {matching_rows:,}")
    print(f"Mismatching rows: {mismatching_rows:,}")
    print(f"Both columns null: {both_null_rows:,}")
    print(f"Only one column null: {one_null_rows:,}")
    print(f"Unique values in {COLUMN_A}: {len(unique_a)}")
    print(f"Unique values in {COLUMN_B}: {len(unique_b)}")

    print(
        "\nExact equality across all rows:",
        mismatching_rows == 0
    )

    if mismatch_examples:
        print("\nFirst mismatching examples:")
        print(pd.DataFrame(mismatch_examples).to_string(index=False))

    if unique_a != unique_b:
        print("\nThe two columns have different value sets.")
    else:
        print("\nThe two columns have identical value sets.")


def main():
    print("=" * 75)
    print("SENTINELX STEP 36")
    print("ORIGINAL FLOW FEATURE DUPLICATION AUDIT")
    print("=" * 75)

    for path in FILES:
        audit_file(path)

    print("\n" + "=" * 75)
    print("STEP 36 AUDIT COMPLETED")
    print("=" * 75)

    print("Original datasets were not modified.")
    print("No features were removed.")
    print("No model was retrained.")


if __name__ == "__main__":
    main()
