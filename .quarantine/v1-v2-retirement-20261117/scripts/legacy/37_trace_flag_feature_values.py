
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Datasets"

FILES = [
    ("ORIGINAL RAW", DATA / "sentinelx_multiday_raw.csv"),
    ("TIMESTAMPED STATES", DATA / "sentinelx_multiday_states.csv"),
    ("EXPANDED TRAIN", DATA / "sentinelx_expanded_train_v1.csv"),
    ("PREPROCESSED TRAIN",
     DATA / "sentinelx_expanded_train_preprocessed_v1.csv"),
]

TERMS = ["cwe", "urg"]
CHUNK_SIZE = 250_000


def relevant_columns(columns):
    result = []

    for column in columns:
        normalized = "".join(
            char.lower()
            for char in str(column)
            if char.isalnum()
        )

        if any(term in normalized for term in TERMS):
            result.append(column)

    return result


def inspect_file(description, path):
    print("\n" + "=" * 80)
    print(description)
    print("=" * 80)

    if not path.exists():
        print("FILE NOT FOUND:", path)
        return

    print("File:", path.name)

    header = pd.read_csv(path, nrows=0).columns.tolist()
    columns = relevant_columns(header)

    if not columns:
        print("No CWE/URG columns found.")
        return

    print("\nRelevant columns:")
    for column in columns:
        print(" ", column)

    stats = {
        column: {
            "rows": 0,
            "nulls": 0,
            "minimum": np.inf,
            "maximum": -np.inf,
            "unique": set(),
            "unique_overflow": False,
        }
        for column in columns
    }

    for chunk in pd.read_csv(
        path,
        usecols=columns,
        chunksize=CHUNK_SIZE,
    ):
        for column in columns:
            values = pd.to_numeric(
                chunk[column],
                errors="coerce",
            )

            current = stats[column]

            current["rows"] += len(values)
            current["nulls"] += int(values.isna().sum())

            if values.notna().any():
                current["minimum"] = min(
                    current["minimum"],
                    float(values.min()),
                )

                current["maximum"] = max(
                    current["maximum"],
                    float(values.max()),
                )

                new_values = set(values.dropna().unique().tolist())

                current["unique"].update(new_values)

                if len(current["unique"]) > 50:
                    current["unique_overflow"] = True
                    current["unique"] = set(
                        list(current["unique"])[:50]
                    )

    print("\nFEATURE VALUE REPORT")

    for column in columns:
        result = stats[column]

        print("\nFeature:", column)
        print("Rows:", f"{result['rows']:,}")
        print("Missing values:", f"{result['nulls']:,}")

        if result["minimum"] == np.inf:
            print("No numeric values found.")
            continue

        print("Minimum:", result["minimum"])
        print("Maximum:", result["maximum"])

        if result["unique_overflow"]:
            print("Unique values: more than 50")
        else:
            print("Unique values:", len(result["unique"]))

        print(
            "Observed values:",
            sorted(result["unique"])[:20],
        )


def main():
    print("=" * 80)
    print("SENTINELX STEP 37")
    print("TRACE FLAG FEATURE VALUES THROUGH THE PIPELINE")
    print("=" * 80)

    for description, path in FILES:
        inspect_file(description, path)

    print("\n" + "=" * 80)
    print("STEP 37 COMPLETED")
    print("=" * 80)

    print("All files were inspected read-only.")
    print("No preprocessing was performed.")
    print("No features were removed.")
    print("No model was retrained.")


if __name__ == "__main__":
    main()
