from pathlib import Path
from zipfile import ZipFile
import csv

ROOT = Path(r"F:\Projects\SentinelX")
ARCHIVE = ROOT / "Datasets" / "archive.zip"

if not ARCHIVE.exists():
    raise FileNotFoundError(f"Archive not found: {ARCHIVE}")

# These are the three capture days already used.
already_used = {
    "02-14-2018.csv",
    "02-15-2018.csv",
    "03-02-2018.csv"
}

print("=" * 75)
print("SENTINELX — CIC-IDS-2018 ARCHIVE INSPECTION")
print("=" * 75)
print("Archive:", ARCHIVE)
print(f"Archive size: {ARCHIVE.stat().st_size / (1024**3):.2f} GB")

with ZipFile(ARCHIVE, "r") as archive:
    entries = [
        item for item in archive.infolist()
        if not item.is_dir() and item.filename.lower().endswith(".csv")
    ]

    if not entries:
        print("\nNo CSV files found in the archive.")
        raise SystemExit(0)

    rows = []

    for item in entries:
        filename = Path(item.filename).name
        rows.append({
            "Archive_Path": item.filename,
            "Filename": filename,
            "Uncompressed_GB": item.file_size / (1024**3),
            "Compressed_GB": item.compress_size / (1024**3),
            "Already_Used": filename in already_used
        })

    rows.sort(key=lambda row: row["Filename"])

    print(f"\nCSV files found: {len(rows)}")
    print("\nCSV inventory:")
    print(
        f"{'Filename':42} {'Size GB':>10} {'Used':>8}"
    )
    print("-" * 65)

    for row in rows:
        used = "YES" if row["Already_Used"] else "NO"
        print(
            f"{row['Filename'][:42]:42} "
            f"{row['Uncompressed_GB']:10.3f} "
            f"{used:>8}"
        )

    total_uncompressed = sum(r["Uncompressed_GB"] for r in rows)
    total_compressed = sum(r["Compressed_GB"] for r in rows)
    unused = [r for r in rows if not r["Already_Used"]]

    print("\nSUMMARY")
    print("-" * 65)
    print("Total CSV files:", len(rows))
    print("Already-used CSV files:", len(rows) - len(unused))
    print("Additional CSV files:", len(unused))
    print(f"Total uncompressed CSV size: {total_uncompressed:.2f} GB")
    print(f"Total compressed CSV size: {total_compressed:.2f} GB")

    print("\nADDITIONAL CSV FILES TO INVESTIGATE")
    print("-" * 65)

    for row in unused:
        print(
            f"{row['Filename']} | "
            f"{row['Uncompressed_GB']:.3f} GB"
        )

    print("\nNo files were extracted or modified.")
