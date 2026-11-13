import zipfile
import csv
from pathlib import Path

zip_path = Path(r"F:\Projects\SentinelX\Datasets\archive.zip")

with zipfile.ZipFile(zip_path) as archive:
    names = {Path(n).name: n for n in archive.namelist() if n.endswith(".csv")}

    def read_sample(filename):
        with archive.open(names[filename]) as f:
            reader = csv.reader(
                (line.decode("utf-8-sig", errors="replace") for line in f)
            )
            header = next(reader)
            rows = [next(reader) for _ in range(5)]
        return header, rows

    reference_header, _ = read_sample("02-14-2018.csv")
    feb20_header, feb20_rows = read_sample("02-20-2018.csv")

    reference = [c.strip() for c in reference_header]
    feb20 = [c.strip() for c in feb20_header]

    common_reference = [c for c in reference if c in feb20]
    common_feb20 = [c for c in feb20 if c in reference]

    print("Reference schema columns:", len(reference))
    print("Feb 20 schema columns:", len(feb20))
    print("Common columns:", len(common_reference))
    print("Common columns preserve same order:",
          common_reference == common_feb20)
    print("Reference columns absent from Feb 20:",
          sorted(set(reference) - set(feb20)))
    print("Feb 20 extra columns:",
          sorted(set(feb20) - set(reference)))

    print("\nExtra-column sample values:")
    for col in ["Flow ID", "Src IP", "Src Port", "Dst IP"]:
        idx = feb20.index(col)
        print(f"\n{col}:")
        for row in feb20_rows:
            print(" ", row[idx])

    print("\nFirst 10 common columns:")
    print(common_reference[:10])

    print("\nFinal 10 common columns:")
    print(common_reference[-10:])
