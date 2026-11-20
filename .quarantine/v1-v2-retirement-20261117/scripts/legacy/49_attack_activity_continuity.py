from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(r"F:\Projects\SentinelX")
DATA = ROOT / "Datasets"

SOURCE = (
    DATA / "diagnostics" / "mixed_traffic_analysis_v1"
    / "per_second_traffic_composition.csv"
)

OUT = DATA / "diagnostics" / "attack_activity_continuity_v1"

if OUT.exists():
    raise FileExistsError(
        f"Output folder already exists: {OUT}. "
        "Inspect it before rerunning."
    )

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

OUT.mkdir(parents=True)

df = pd.read_csv(SOURCE)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

required = [
    "Timestamp", "CaptureDay", "Attack Flow Count",
    "Has Attack", "Attack Label Counts"
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if df["Timestamp"].isna().any():
    raise ValueError("Invalid timestamps found.")

df = df.sort_values(["CaptureDay", "Timestamp"]).reset_index(drop=True)

# Identify attack-positive seconds.
df["Has Attack"] = df["Has Attack"].fillna(0).astype(int)
df["Attack Flow Count"] = (
    df["Attack Flow Count"].fillna(0).astype(int)
)

active = df[df["Has Attack"] == 1].copy()

# Parse attack label counts such as Bot=2 or FTP-BruteForce=3.
def parse_attack_labels(value):
    if pd.isna(value):
        return []

    matches = re.findall(
        r"([^,;|]+?)\s*=\s*(\d+)",
        str(value)
    )

    return [
        (name.strip(), int(count))
        for name, count in matches
        if name.strip()
    ]

active["Parsed_Attack_Labels"] = (
    active["Attack Label Counts"].apply(parse_attack_labels)
)

# Overall gaps between attack-positive seconds, separately per day.
gap_rows = []

for day, group in active.groupby("CaptureDay"):
    group = group.sort_values("Timestamp")
    times = group["Timestamp"].to_numpy()

    for i in range(1, len(times)):
        gap_seconds = (
            pd.Timestamp(times[i]) -
            pd.Timestamp(times[i - 1])
        ).total_seconds() - 1

        gap_rows.append({
            "CaptureDay": day,
            "Previous_Attack_Second": times[i - 1],
            "Next_Attack_Second": times[i],
            "Inactive_Gap_Seconds": gap_seconds
        })

gaps = pd.DataFrame(gap_rows)

# Build attack-type-specific active-second series.
type_rows = []

for _, row in active.iterrows():
    for attack_name, count in row["Parsed_Attack_Labels"]:
        type_rows.append({
            "CaptureDay": row["CaptureDay"],
            "Timestamp": row["Timestamp"],
            "Attack_Type": attack_name,
            "Attack_Flow_Count": count
        })

type_df = pd.DataFrame(type_rows)

type_gap_rows = []

if not type_df.empty:
    for (day, attack_type), group in type_df.groupby(
        ["CaptureDay", "Attack_Type"]
    ):
        group = group.sort_values("Timestamp")
        times = group["Timestamp"].to_numpy()

        for i in range(1, len(times)):
            gap_seconds = (
                pd.Timestamp(times[i]) -
                pd.Timestamp(times[i - 1])
            ).total_seconds() - 1

            type_gap_rows.append({
                "CaptureDay": day,
                "Attack_Type": attack_type,
                "Inactive_Gap_Seconds": gap_seconds
            })

type_gaps = pd.DataFrame(type_gap_rows)

# Gap distribution summary.
def summarize_gaps(frame, group_columns):
    rows = []

    if frame.empty:
        return pd.DataFrame()

    grouped = (
        frame.groupby(group_columns)
        if group_columns else [(("Overall",), frame)]
    )

    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)

        row = dict(zip(group_columns, key))
        values = group["Inactive_Gap_Seconds"]

        row.update({
            "Gap_Count": len(values),
            "Mean_Gap_Seconds": values.mean(),
            "Median_Gap_Seconds": values.median(),
            "P90_Gap_Seconds": values.quantile(0.90),
            "P95_Gap_Seconds": values.quantile(0.95),
            "P99_Gap_Seconds": values.quantile(0.99),
            "Maximum_Gap_Seconds": values.max()
        })

        rows.append(row)

    return pd.DataFrame(rows)

overall_gap_summary = summarize_gaps(gaps, ["CaptureDay"])
type_gap_summary = summarize_gaps(
    type_gaps, ["CaptureDay", "Attack_Type"]
)

# Episode sensitivity: merge positive seconds if the number of
# inactive seconds between them is <= the chosen tolerance.
tolerances = [0, 1, 2, 5, 10, 30]

episode_rows = []

def count_episodes(timestamps, tolerance):
    if len(timestamps) == 0:
        return 0

    timestamps = sorted(pd.to_datetime(timestamps))
    episodes = 1

    for i in range(1, len(timestamps)):
        inactive_gap = (
            timestamps[i] - timestamps[i - 1]
        ).total_seconds() - 1

        if inactive_gap > tolerance:
            episodes += 1

    return episodes

for day, group in active.groupby("CaptureDay"):
    for tolerance in tolerances:
        episode_rows.append({
            "CaptureDay": day,
            "Activity": "All_Attack_Types",
            "Allowed_Inactive_Gap_Seconds": tolerance,
            "Active_Seconds": len(group),
            "Episode_Count": count_episodes(
                group["Timestamp"], tolerance
            )
        })

if not type_df.empty:
    for (day, attack_type), group in type_df.groupby(
        ["CaptureDay", "Attack_Type"]
    ):
        for tolerance in tolerances:
            episode_rows.append({
                "CaptureDay": day,
                "Activity": attack_type,
                "Allowed_Inactive_Gap_Seconds": tolerance,
                "Active_Seconds": group["Timestamp"].nunique(),
                "Episode_Count": count_episodes(
                    group["Timestamp"].unique(), tolerance
                )
            })

episode_sensitivity = pd.DataFrame(episode_rows)

# Overall daily activity summary.
daily_summary = (
    df.groupby("CaptureDay")
    .agg(
        Observed_Seconds=("Timestamp", "size"),
        Attack_Seconds=("Has Attack", "sum"),
        Total_Attack_Flows=("Attack Flow Count", "sum")
    )
    .reset_index()
)

daily_summary["Attack_Second_Percentage"] = (
    100 * daily_summary["Attack_Seconds"] /
    daily_summary["Observed_Seconds"]
)

# Save results.
outputs = {
    "attack_activity_gaps.csv": gaps,
    "attack_gap_summary_by_day.csv": overall_gap_summary,
    "attack_type_gap_summary.csv": type_gap_summary,
    "episode_gap_sensitivity.csv": episode_sensitivity,
    "daily_attack_activity_summary.csv": daily_summary
}

for filename, frame in outputs.items():
    frame.to_csv(OUT / filename, index=False)

report = [
    "SENTINELX ATTACK ACTIVITY CONTINUITY ANALYSIS",
    "=" * 70,
    "",
    f"Total observed seconds: {len(df)}",
    f"Attack-positive seconds: {len(active)}",
    f"Attack-positive flow total: {int(active['Attack Flow Count'].sum())}",
    "",
    "DAILY ATTACK ACTIVITY",
    "-" * 70,
    daily_summary.to_string(index=False),
    "",
    "OVERALL ATTACK GAP DISTRIBUTION BY DAY",
    "-" * 70,
    overall_gap_summary.to_string(index=False),
    "",
    "ATTACK-TYPE GAP DISTRIBUTION",
    "-" * 70,
    type_gap_summary.to_string(index=False),
    "",
    "EPISODE COUNT SENSITIVITY",
    "-" * 70,
    episode_sensitivity.to_string(index=False),
    "",
    "INTERPRETATION",
    "-" * 70,
    "An attack-positive second contains at least one labelled attack flow.",
    "An inactive gap is the number of seconds between two positive seconds.",
    "Episode counts are exploratory and depend on the chosen gap tolerance.",
    "They do not prove that separated activity belongs to one real campaign.",
    "Original per-second labels and source records are not modified."
]

report_path = OUT / "attack_activity_continuity_report.txt"
report_path.write_text("\n".join(report), encoding="utf-8")

print("\n" + "=" * 70)
print("ATTACK ACTIVITY CONTINUITY ANALYSIS COMPLETE")
print("=" * 70)

print("\nDaily activity:")
print(daily_summary.to_string(index=False))

print("\nGap distribution by day:")
print(overall_gap_summary.to_string(index=False))

print("\nAttack-type gap distribution:")
print(type_gap_summary.to_string(index=False))

print("\nEpisode sensitivity:")
print(episode_sensitivity.to_string(index=False))

print("\nOutput folder:", OUT)
print("Report:", report_path)
