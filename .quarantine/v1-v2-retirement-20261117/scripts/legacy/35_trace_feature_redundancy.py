
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Datasets"

PAIRS = [
    ("CWE Flag Count__max", "Fwd URG Flags__max"),
    ("CWE Flag Count__std", "Fwd URG Flags__std"),
]

SPLITS = ["train", "validation", "test"]

FILES = {
    "expanded": "sentinelx_expanded_{split}_v1.csv",
    "preprocessed": "sentinelx_expanded_{split}_preprocessed_v1.csv",
}


def inspect_file(path, description):
    print("\n" + "-" * 75)
    print(description)
    print(path.name)

    if not path.exists():
        print("FILE NOT FOUND — skipped")
        return

    header = pd.read_csv(path, nrows=0).columns.tolist()

    required = set()
    for a, b in PAIRS:
        required.update([a, b])

    available = [c for c in required if c in header]

    if len(available) != len(required):
        print("Some expected feature columns are missing.")
        print("Available:", available)
        return

    extra = [
        c for c in ["Timestamp", "Label", "Segment ID", "CaptureDay"]
        if c in header
    ]

    df = pd.read_csv(
        path,
        usecols=available + extra,
    )

    print(f"Rows: {len(df):,}")

    if "Segment ID" in df.columns:
        print(f"Unique segments: {df['Segment ID'].nunique():,}")

    for first, second in PAIRS:
        a = pd.to_numeric(df[first], errors="coerce")
        b = pd.to_numeric(df[second], errors="coerce")

        exact = a.equals(b)

        difference = (a - b).abs()
        max_difference = difference.max()

        print(f"\n{first} VS {second}")
        print(f"Exactly identical: {exact}")
        print(f"Maximum absolute difference: {max_difference}")
        print(f"{first} unique values: {a.nunique(dropna=False)}")
        print(f"{second} unique values: {b.nunique(dropna=False)}")

        print("First 5 paired values:")
        print(
            pd.DataFrame({
                first: a.head(),
                second: b.head(),
            }).to_string(index=False)
        )


def inspect_sequence_files(split):
    print("\n" + "=" * 75)
    print(f"{split.upper()} SEQUENCE FILE CHECK")
    print("=" * 75)

    x_path = DATA / f"sentinelx_{split}_X_seq10_v1.npy"
    y_path = DATA / f"sentinelx_{split}_y_next_v1.npy"
    meta_path = DATA / (
        f"sentinelx_{split}_sequence_metadata_v1.csv"
    )

    for path in [x_path, y_path, meta_path]:
        if not path.exists():
            print(f"MISSING: {path.name}")
            return

    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    metadata = pd.read_csv(meta_path)

    print(f"X samples: {X.shape[0]:,}")
    print(f"y samples: {y.shape[0]:,}")
    print(f"Metadata rows: {len(metadata):,}")

    if "Segment ID" in metadata.columns:
        counts = metadata["Segment ID"].value_counts()

        print(f"Metadata segments: {len(counts):,}")
        print(f"Largest segment sample count: {counts.max():,}")
        print(f"Smallest segment sample count: {counts.min():,}")

    if "Target State Row" in metadata.columns:
        print(
            "Target State Row range:",
            metadata["Target State Row"].min(),
            "to",
            metadata["Target State Row"].max(),
        )


def main():
    print("=" * 75)
    print("SENTINELX STEP 35")
    print("TRACE FEATURE REDUNDANCY TO SOURCE")
    print("=" * 75)

    for split in SPLITS:
        for kind, pattern in FILES.items():
            path = DATA / pattern.format(split=split)
            inspect_file(path, f"{split.upper()} — {kind.upper()}")

        inspect_sequence_files(split)

    print("\n" + "=" * 75)
    print("STEP 35 SOURCE TRACE COMPLETED")
    print("=" * 75)

    print("No files were modified.")
    print("No features were removed.")
    print("No sequences were regenerated.")
    print("No model was retrained.")


if __name__ == "__main__":
    main()
