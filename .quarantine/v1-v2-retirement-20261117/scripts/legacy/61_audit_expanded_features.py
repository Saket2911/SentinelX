from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(r"F:\Projects\SentinelX\Datasets")
SOURCE = DATA_DIR / "sentinelx_expanded_chronological_v1.csv"
REPORT = DATA_DIR / "sentinelx_expanded_feature_audit_v1.txt"

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

metadata = {"Timestamp", "CaptureDay", "Label"}
header = pd.read_csv(SOURCE, nrows=0).columns.tolist()
features = [c for c in header if c not in metadata]

stats = {
    c: {
        "min": np.inf,
        "max": -np.inf,
        "missing": 0,
        "positive_inf": 0,
        "negative_inf": 0,
        "finite": 0,
        "non_numeric": 0,
    }
    for c in features
}

total_rows = 0

print("=" * 70)
print("SENTINELX EXPANDED FEATURE AUDIT")
print("=" * 70)
print("Feature columns:", len(features))
print("Streaming CSV in chunks; this may take several minutes.")

reader = pd.read_csv(
    SOURCE,
    chunksize=100_000,
    low_memory=False,
    dtype={
        "Timestamp": "string",
        "CaptureDay": "string",
        "Label": "string",
    },
)

for chunk_no, chunk in enumerate(reader, start=1):
    total_rows += len(chunk)

    for col in features:
        raw = chunk[col]
        values = pd.to_numeric(raw, errors="coerce")

        s = stats[col]
        s["non_numeric"] += int(
            (values.isna() & raw.notna()).sum()
        )
        s["missing"] += int(values.isna().sum())

        s["positive_inf"] += int(np.isposinf(values).sum())
        s["negative_inf"] += int(np.isneginf(values).sum())

        finite = values[np.isfinite(values)]

        if len(finite):
            s["min"] = min(s["min"], float(finite.min()))
            s["max"] = max(s["max"], float(finite.max()))
            s["finite"] += len(finite)

    if chunk_no % 20 == 0:
        print(f"Chunks: {chunk_no:,} | Rows scanned: {total_rows:,}")

constant = []
variable = []
all_missing = []

lines = [
    "SENTINELX EXPANDED FEATURE AUDIT",
    "=" * 60,
    f"Input: {SOURCE}",
    f"Rows scanned: {total_rows:,}",
    f"Feature columns: {len(features)}",
    "",
]

for col in features:
    s = stats[col]

    if s["finite"] == 0:
        all_missing.append(col)
        category = "NO FINITE VALUES"
    elif s["min"] == s["max"]:
        constant.append(col)
        category = "CONSTANT"
    else:
        variable.append(col)
        category = "VARIABLE"

    lines.append(
        f"{category}: {col} | "
        f"min={s['min']} | max={s['max']} | "
        f"finite={s['finite']:,} | missing={s['missing']:,} | "
        f"+inf={s['positive_inf']:,} | -inf={s['negative_inf']:,} | "
        f"non_numeric={s['non_numeric']:,}"
    )

lines.extend([
    "",
    f"Constant features: {len(constant)}",
    f"Variable features: {len(variable)}",
    f"Features with no finite values: {len(all_missing)}",
    "",
    "CONSTANT FEATURE NAMES:",
    *constant,
    "",
    "VARIABLE FEATURE COUNT:",
    str(len(variable)),
])

REPORT.write_text("\n".join(lines), encoding="utf-8")

print("\n--- Audit summary ---")
print("Rows scanned:", f"{total_rows:,}")
print("Total features:", len(features))
print("Constant features:", len(constant))
print("Variable features:", len(variable))
print("All-missing features:", len(all_missing))
print("\nReport saved:", REPORT)
print("\nAudit complete.")
