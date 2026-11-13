import zipfile
import csv
from pathlib import Path

root = Path(r"F:\Projects\SentinelX")
zip_path = root / "Datasets" / "archive.zip"

print("=" * 75)
print("SENTINELX — SOURCE CSV SCHEMA VERIFICATION")
print("=" * 75)

with zipfile.ZipFile(zip_path, "r") as archive:
    entries = {
        Path(info.filename).name: info
        for info in archive.infolist()
        if info.filename.lower().endswith(".csv")
    }

    reference_header = None
    all_compatible = True

    for filename in sorted(entries):
        info = entries[filename]

        with archive.open(info) as file:
            text = (line.decode("utf-8-sig", errors="replace")
                    for line in file)
            reader = csv.reader(text)
            header = next(reader, None)

        if header is None:
            print(f"\n{filename}: EMPTY FILE")
            all_compatible = False
            continue

        normalized = [column.strip() for column in header]

        if reference_header is None:
            reference_header = normalized
            status = "REFERENCE"
        elif normalized == reference_header:
            status = "MATCH"
        else:
            status = "MISMATCH"
            all_compatible = False

        print(f"\nFile: {filename}")
        print(f"Uncompressed size: {info.file_size / (1024**3):.3f} GB")
        print(f"Column count: {len(normalized)}")
        print(f"Schema status: {status}")

        if status == "MISMATCH":
            missing = sorted(set(reference_header) - set(normalized))
            extra = sorted(set(normalized) - set(reference_header))

            print("Missing columns:", missing)
            print("Extra columns:", extra)

            if not missing and not extra:
                for index, (a, b) in enumerate(
                    zip(reference_header, normalized)
                ):
                    if a != b:
                        print(
                            f"Order difference at position {index}: "
                            f"{a!r} vs {b!r}"
                        )

print("\n" + "=" * 75)
print("OVERALL RESULT:",
      "ALL SCHEMAS MATCH" if all_compatible else "SCHEMA DIFFERENCES FOUND")
print("=" * 75)

# Verify the existing chronological dataset separately.
existing = root / "Datasets" / "sentinelx_multiday_chronological_v2.csv"
if existing.exists():
    print(f"\nExisting chronological dataset: {existing.name}")
    print(f"Size: {existing.stat().st_size / (1024**3):.3f} GB")
else:
    print("\nExisting chronological dataset was not found.")
