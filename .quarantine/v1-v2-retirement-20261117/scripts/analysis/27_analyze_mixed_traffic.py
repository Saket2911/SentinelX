from pathlib import Path
from collections import defaultdict
import pandas as pd

ROOT = Path(r"F:\Projects\SentinelX")

INPUT_FILE = (
    ROOT / "Datasets"
    / "sentinelx_multiday_chronological_v2.csv"
)

OUTPUT_DIR = (
    ROOT / "Datasets"
    / "diagnostics"
    / "mixed_traffic_analysis_v1"
)

WINDOWS_FILE = OUTPUT_DIR / "per_second_traffic_composition.csv"
DAILY_FILE = OUTPUT_DIR / "daily_mixed_traffic_summary.csv"
LABEL_FILE = OUTPUT_DIR / "overall_flow_label_counts.csv"
REPORT_FILE = OUTPUT_DIR / "mixed_traffic_analysis_report.txt"

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

outputs = [WINDOWS_FILE, DAILY_FILE, LABEL_FILE, REPORT_FILE]
existing = [str(path) for path in outputs if path.exists()]

if existing:
    raise FileExistsError(
        "Output files already exist. Nothing was overwritten:\n"
        + "\n".join(existing)
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Accumulate flow counts by one-second window and original flow label.
window_label_counts = defaultdict(lambda: defaultdict(int))
overall_label_counts = defaultdict(int)

rows_scanned = 0
chunksize = 200_000

print("=" * 70)
print("SENTINELX - MIXED TRAFFIC ANALYSIS")
print("=" * 70)
print("Input:", INPUT_FILE)
print("Reading original flow records in chunks...")

for chunk in pd.read_csv(
    INPUT_FILE,
    usecols=["Timestamp", "Label"],
    chunksize=chunksize
):
    rows_scanned += len(chunk)

    chunk["Timestamp"] = pd.to_datetime(
        chunk["Timestamp"], errors="coerce"
    )
    chunk = chunk.dropna(subset=["Timestamp", "Label"])

    chunk["Window"] = chunk["Timestamp"].dt.floor("1s")
    chunk["Label"] = chunk["Label"].astype(str).str.strip()

    grouped = (
        chunk.groupby(["Window", "Label"], sort=False)
        .size()
    )

    for (window, label), count in grouped.items():
        count = int(count)
        window_label_counts[window][label] += count
        overall_label_counts[label] += count

    if rows_scanned % 1_000_000 < chunksize:
        print(f"Scanned {rows_scanned:,} rows...")

print("\nBuilding per-second composition table...")

records = []

for window, label_counts in sorted(window_label_counts.items()):
    benign_count = sum(
        count for label, count in label_counts.items()
        if label.lower() == "benign"
    )

    attack_counts = {
        label: count
        for label, count in label_counts.items()
        if label.lower() != "benign"
    }

    attack_count = sum(attack_counts.values())
    total_count = benign_count + attack_count

    has_benign = benign_count > 0
    has_attack = attack_count > 0

    records.append({
        "Timestamp": window,
        "CaptureDay": window.strftime("%Y-%m-%d"),
        "Flow Count": total_count,
        "Benign Flow Count": benign_count,
        "Attack Flow Count": attack_count,
        "Attack Flow Ratio": (
            attack_count / total_count if total_count else 0.0
        ),
        "Has Benign": int(has_benign),
        "Has Attack": int(has_attack),
        "Mixed Benign Attack": int(has_benign and has_attack),
        "Distinct Attack Labels": len(attack_counts),
        "Multiple Attack Types": int(len(attack_counts) > 1),
        "Attack Label Counts": "; ".join(
            f"{label}={count}"
            for label, count in sorted(attack_counts.items())
        )
    })

per_second = pd.DataFrame(records)

if per_second.empty:
    raise RuntimeError("No valid one-second windows were produced.")

per_second.to_csv(WINDOWS_FILE, index=False)

print("Building daily summary...")

daily = (
    per_second.groupby("CaptureDay")
    .agg(
        Observed_Seconds=("Timestamp", "count"),
        Total_Flows=("Flow Count", "sum"),
        Mixed_Seconds=("Mixed Benign Attack", "sum"),
        Attack_Seconds=("Has Attack", "sum"),
        Benign_Only_Seconds=(
            "Mixed Benign Attack",
            lambda values: 0
        ),
        Multiple_Attack_Type_Seconds=(
            "Multiple Attack Types", "sum"
        ),
        Mean_Attack_Flow_Ratio=("Attack Flow Ratio", "mean")
    )
    .reset_index()
)

# Benign-only seconds have flows but no attack flows.
daily["Benign_Only_Seconds"] = (
    per_second.groupby("CaptureDay")
    .apply(
        lambda group: (
            (group["Has Benign"] == 1)
            & (group["Has Attack"] == 0)
        ).sum(),
        include_groups=False
    )
    .to_numpy()
)

daily["Mixed_Seconds_Percent"] = (
    daily["Mixed_Seconds"] / daily["Observed_Seconds"] * 100
)

daily["Attack_Seconds_Percent"] = (
    daily["Attack_Seconds"] / daily["Observed_Seconds"] * 100
)

daily["Mean_Attack_Flow_Ratio_Percent"] = (
    daily["Mean_Attack_Flow_Ratio"] * 100
)

daily.to_csv(DAILY_FILE, index=False)

print("Saving overall flow-label counts...")

label_table = pd.DataFrame([
    {"Label": label, "Flow Count": count}
    for label, count in sorted(
        overall_label_counts.items(),
        key=lambda item: item[1],
        reverse=True
    )
])

label_table.to_csv(LABEL_FILE, index=False)

# Overall summary.
observed_seconds = len(per_second)
mixed_seconds = int(per_second["Mixed Benign Attack"].sum())
attack_seconds = int(per_second["Has Attack"].sum())
benign_only_seconds = int(
    (
        (per_second["Has Benign"] == 1)
        & (per_second["Has Attack"] == 0)
    ).sum()
)
attack_only_seconds = int(
    (
        (per_second["Has Benign"] == 0)
        & (per_second["Has Attack"] == 1)
    ).sum()
)
multiple_attack_seconds = int(
    per_second["Multiple Attack Types"].sum()
)

report = [
    "SENTINELX - MIXED TRAFFIC ANALYSIS REPORT",
    "=" * 55,
    "",
    f"Input file: {INPUT_FILE}",
    f"Rows scanned: {rows_scanned:,}",
    f"Observed one-second windows: {observed_seconds:,}",
    "",
    "OVERALL WINDOW COMPOSITION",
    "-" * 55,
    f"Benign-only windows: {benign_only_seconds:,}",
    f"Attack-only windows: {attack_only_seconds:,}",
    f"Mixed Benign + Attack windows: {mixed_seconds:,}",
    f"Windows containing any attack: {attack_seconds:,}",
    f"Windows with multiple attack types: {multiple_attack_seconds:,}",
    "",
    f"Mixed windows (% of all observed windows): "
    f"{mixed_seconds / observed_seconds * 100:.3f}%",
    f"Attack-containing windows (% of all observed windows): "
    f"{attack_seconds / observed_seconds * 100:.3f}%",
    "",
    "FLOW LABEL COUNTS",
    "-" * 55,
]

for _, row in label_table.iterrows():
    report.append(
        f"{row['Label']}: {int(row['Flow Count']):,}"
    )

report.extend([
    "",
    "DAILY SUMMARY",
    "-" * 55,
])

for _, row in daily.iterrows():
    report.append(
        f"{row['CaptureDay']}: "
        f"{int(row['Observed_Seconds']):,} observed seconds, "
        f"{int(row['Mixed_Seconds']):,} mixed seconds "
        f"({row['Mixed_Seconds_Percent']:.3f}%), "
        f"{int(row['Attack_Seconds']):,} attack-containing seconds"
    )

report.extend([
    "",
    "INTERPRETATION",
    "-" * 55,
    "A mixed window contains at least one Benign flow and at least",
    "one non-Benign flow within the same one-second interval.",
    "The current state-label rule assigns an attack label to such",
    "a window, even if Benign flows are more numerous.",
    "",
    "This report is diagnostic. It does not change model inputs,",
    "state labels, preprocessing, sequences, or trained models.",
])

REPORT_FILE.write_text("\n".join(report), encoding="utf-8")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"Rows scanned: {rows_scanned:,}")
print(f"Observed seconds: {observed_seconds:,}")
print(
    f"Mixed Benign + Attack seconds: {mixed_seconds:,} "
    f"({mixed_seconds / observed_seconds * 100:.3f}%)"
)
print(
    f"Attack-containing seconds: {attack_seconds:,} "
    f"({attack_seconds / observed_seconds * 100:.3f}%)"
)
print("\nDaily summary:")
print(daily.to_string(index=False))

print("\nOutput files:")
for path in outputs:
    print(path)
