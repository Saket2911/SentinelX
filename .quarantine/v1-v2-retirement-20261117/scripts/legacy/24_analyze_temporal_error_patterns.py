from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"F:\Projects\SentinelX")

INPUT = (
    ROOT / "Models" / "lstm_world_model_v1"
    / "anomaly_experiment_v1"
    / "test_anomaly_predictions.csv"
)

OUTPUT_DIR = (
    ROOT / "Models" / "lstm_world_model_v1"
    / "temporal_error_analysis_v1"
)

if not INPUT.exists():
    raise FileNotFoundError(INPUT)

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Output already exists: {OUTPUT_DIR}"
    )

OUTPUT_DIR.mkdir(parents=True)

df = pd.read_csv(INPUT)

required = {
    "timestamp",
    "actual_label",
    "prediction_error"
}

if not required.issubset(df.columns):
    raise ValueError(
        f"Missing columns: {required - set(df.columns)}"
    )

df["timestamp"] = pd.to_datetime(
    df["timestamp"], errors="raise"
)

df["prediction_error"] = pd.to_numeric(
    df["prediction_error"], errors="raise"
)

if not np.isfinite(df["prediction_error"]).all():
    raise ValueError("Non-finite prediction errors found.")

if set(df["actual_label"].unique()) != {"Benign", "Bot"}:
    raise ValueError("Expected only Benign and Bot labels.")

df = df.sort_values("timestamp").reset_index(drop=True)

df["is_bot"] = (
    df["actual_label"] == "Bot"
).astype(int)

print("=" * 65)
print("SENTINELX - TEMPORAL ERROR ANALYSIS")
print("=" * 65)

print("Total rows:", len(df))
print("Start:", df["timestamp"].min())
print("End:", df["timestamp"].max())

print(
    "Duplicate timestamps:",
    int(df["timestamp"].duplicated().sum())
)

time_diffs = df["timestamp"].diff().dt.total_seconds()

print(
    "Non-one-second intervals:",
    int((time_diffs.dropna() != 1).sum())
)

print(
    "Chronological ordering violations:",
    int((time_diffs.dropna() < 0).sum())
)


# -------------------------------------------------
# 1. Overall temporal distribution
# -------------------------------------------------

print("\nOVERALL LABEL DISTRIBUTION")

print(
    df["actual_label"].value_counts().to_string()
)

print("\nOVERALL ERROR BY LABEL")

overall = df.groupby("actual_label")[
    "prediction_error"
].agg(
    ["count", "mean", "median", "std", "min", "max"]
)

print(overall.to_string())


# -------------------------------------------------
# 2. Five-minute intervals
# -------------------------------------------------

df["time_bin"] = df["timestamp"].dt.floor("5min")

temporal = df.groupby("time_bin").agg(
    total_states=("actual_label", "size"),
    bot_count=("is_bot", "sum"),
    bot_fraction=("is_bot", "mean"),
    mean_error=("prediction_error", "mean"),
    median_error=("prediction_error", "median")
)

for label in ["Benign", "Bot"]:
    subset = df[df["actual_label"] == label]

    grouped = subset.groupby("time_bin")[
        "prediction_error"
    ].agg(["count", "mean", "median"])

    grouped = grouped.rename(columns={
        "count": f"{label}_count",
        "mean": f"{label}_mean_error",
        "median": f"{label}_median_error"
    })

    temporal = temporal.join(grouped)

temporal = temporal.reset_index()

temporal_path = OUTPUT_DIR / "five_minute_temporal_analysis.csv"

temporal.to_csv(temporal_path, index=False)

print("\nFIVE-MINUTE INTERVAL SUMMARY")
print("Intervals:", len(temporal))
print("\nFirst 12 intervals:")
print(temporal.head(12).to_string(index=False))

print("\nLast 12 intervals:")
print(temporal.tail(12).to_string(index=False))


# -------------------------------------------------
# 3. Compare periods
# -------------------------------------------------

print("\n" + "=" * 65)
print("ERROR DIFFERENCE OVER TIME")
print("=" * 65)

valid = temporal.dropna(
    subset=["Benign_mean_error", "Bot_mean_error"]
).copy()

valid["benign_minus_bot_error"] = (
    valid["Benign_mean_error"]
    - valid["Bot_mean_error"]
)

print("Intervals containing both labels:", len(valid))

if len(valid):
    print(
        "Intervals where Benign error > Bot error:",
        int((valid["benign_minus_bot_error"] > 0).sum())
    )

    print(
        "Intervals where Bot error > Benign error:",
        int((valid["benign_minus_bot_error"] < 0).sum())
    )

    print(
        "Mean difference (Benign - Bot):",
        float(valid["benign_minus_bot_error"].mean())
    )

    print(
        "Median difference (Benign - Bot):",
        float(valid["benign_minus_bot_error"].median())
    )


# -------------------------------------------------
# 4. Consecutive label runs
# -------------------------------------------------

df["new_run"] = (
    df["actual_label"] != df["actual_label"].shift()
).cumsum()

runs = df.groupby("new_run").agg(
    start_time=("timestamp", "min"),
    end_time=("timestamp", "max"),
    label=("actual_label", "first"),
    states=("actual_label", "size"),
    mean_error=("prediction_error", "mean")
).reset_index(drop=True)

runs_path = OUTPUT_DIR / "consecutive_label_runs.csv"
runs.to_csv(runs_path, index=False)

print("\n" + "=" * 65)
print("CONSECUTIVE LABEL RUNS")
print("=" * 65)

print("Total runs:", len(runs))
print("Label transitions:", max(0, len(runs) - 1))

print("\nRun counts by label:")
print(runs["label"].value_counts().to_string())

print("\nMedian run length by label:")
print(
    runs.groupby("label")["states"]
    .median()
    .to_string()
)

print("\nFirst 10 runs:")
print(runs.head(10).to_string(index=False))


# -------------------------------------------------
# 5. Correlation - descriptive only
# -------------------------------------------------

correlation = df[
    ["is_bot", "prediction_error"]
].corr(method="spearman").iloc[0, 1]

print("\n" + "=" * 65)
print("DESCRIPTIVE CORRELATION")
print("=" * 65)

print(
    "Spearman correlation between Bot label and error:",
    float(correlation)
)


# -------------------------------------------------
# 6. Save report
# -------------------------------------------------

report = [
    "SENTINELX - TEMPORAL ERROR ANALYSIS",
    "",
    f"Total states: {len(df)}",
    f"Start: {df['timestamp'].min()}",
    f"End: {df['timestamp'].max()}",
    f"Duplicate timestamps: {df['timestamp'].duplicated().sum()}",
    f"Non-one-second intervals: {(time_diffs.dropna() != 1).sum()}",
    "",
    "OVERALL ERROR BY LABEL",
    overall.to_string(),
    "",
    "FIVE-MINUTE ANALYSIS",
    f"Intervals: {len(temporal)}",
    f"Intervals containing both labels: {len(valid)}",
    "",
    "ERROR DIFFERENCE (BENIGN - BOT)",
    (
        valid["benign_minus_bot_error"].describe().to_string()
        if len(valid) else "No intervals contain both labels."
    ),
    "",
    "CONSECUTIVE LABEL RUNS",
    f"Total runs: {len(runs)}",
    f"Label transitions: {max(0, len(runs) - 1)}",
    runs.groupby("label")["states"].median().to_string(),
    "",
    f"Spearman correlation: {correlation}",
    "",
    "NOTE: These are descriptive test-set analyses.",
    "Do not use these results to tune thresholds or select models.",
    "Temporal clustering and adjacent sequence overlap mean that",
    "individual rows should not be treated as independent samples."
]

report_path = OUTPUT_DIR / "temporal_analysis_report.txt"

report_path.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("\n" + "=" * 65)
print("ANALYSIS COMPLETED")
print("=" * 65)

print("Output directory:", OUTPUT_DIR)
print("Temporal intervals:", temporal_path)
print("Label runs:", runs_path)
print("Report:", report_path)
