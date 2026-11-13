from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"F:\Projects\SentinelX")
DATA = ROOT / "Datasets"
MODEL_DIR = ROOT / "Models" / "lstm_world_model_v1"

labels_path = DATA / "sentinelx_test_target_labels.npy"
timestamps_path = DATA / "sentinelx_test_target_timestamps.npy"
segments_path = DATA / "sentinelx_test_target_segment_ids.npy"

labels = np.load(labels_path).astype(str)
timestamps = pd.to_datetime(np.load(timestamps_path))
segments = np.load(segments_path)

if not (len(labels) == len(timestamps) == len(segments)):
    raise ValueError("Array length mismatch.")

df = pd.DataFrame({
    "Timestamp": timestamps,
    "Label": labels,
    "Segment": segments
})

# A new run begins whenever the label or segment changes.
df["New_Run"] = (
    (df["Label"] != df["Label"].shift()) |
    (df["Segment"] != df["Segment"].shift())
)
df["Run_ID"] = df["New_Run"].cumsum()

runs = (
    df.groupby(["Run_ID", "Label", "Segment"], as_index=False)
    .agg(
        Start=("Timestamp", "first"),
        End=("Timestamp", "last"),
        Duration_Seconds=("Timestamp", "size")
    )
)

thresholds = [1, 2, 5, 10, 30, 60, 300]

duration_rows = []

for label in sorted(df["Label"].unique()):
    subset = runs[runs["Label"] == label]
    total_states = int(subset["Duration_Seconds"].sum())

    for threshold in thresholds:
        qualifying = subset[
            subset["Duration_Seconds"] >= threshold
        ]
        qualifying_states = int(
            qualifying["Duration_Seconds"].sum()
        )

        duration_rows.append({
            "Label": label,
            "Minimum_Run_Seconds": threshold,
            "Number_Of_Runs": len(qualifying),
            "States_In_Qualifying_Runs": qualifying_states,
            "Total_States_For_Label": total_states,
            "Percentage_Of_Label_States": (
                100 * qualifying_states / total_states
                if total_states else 0
            )
        })

duration_summary = pd.DataFrame(duration_rows)

run_summary = (
    runs.groupby("Label")["Duration_Seconds"]
    .agg(
        Run_Count="count",
        Mean="mean",
        Median="median",
        Minimum="min",
        Maximum="max"
    )
    .reset_index()
)

histogram_rows = []

bins = [
    (1, 1, "1 second"),
    (2, 2, "2 seconds"),
    (3, 5, "3-5 seconds"),
    (6, 10, "6-10 seconds"),
    (11, 30, "11-30 seconds"),
    (31, 60, "31-60 seconds"),
    (61, 300, "61-300 seconds"),
    (301, float("inf"), "Over 300 seconds")
]

for label in sorted(df["Label"].unique()):
    subset = runs[runs["Label"] == label]

    for low, high, name in bins:
        selected = subset[
            (subset["Duration_Seconds"] >= low) &
            (subset["Duration_Seconds"] <= high)
        ]

        histogram_rows.append({
            "Label": label,
            "Duration_Category": name,
            "Run_Count": len(selected),
            "States_In_Runs": int(
                selected["Duration_Seconds"].sum()
            )
        })

histogram = pd.DataFrame(histogram_rows)

out_dir = MODEL_DIR / "label_run_structure_v1"
out_dir.mkdir(parents=True, exist_ok=True)

outputs = [
    out_dir / "all_label_runs.csv",
    out_dir / "label_run_duration_summary.csv",
    out_dir / "label_run_threshold_summary.csv",
    out_dir / "label_run_histogram.csv",
    out_dir / "label_run_structure_report.txt"
]

if any(path.exists() for path in outputs):
    raise FileExistsError(
        "An output already exists. Inspect the output folder before rerunning."
    )

runs.to_csv(outputs[0], index=False)
run_summary.to_csv(outputs[1], index=False)
duration_summary.to_csv(outputs[2], index=False)
histogram.to_csv(outputs[3], index=False)

report = []
report.append("SENTINELX LABEL RUN STRUCTURE ANALYSIS")
report.append("=" * 65)
report.append("")
report.append(f"Total states: {len(df)}")
report.append(f"Total runs: {len(runs)}")
report.append(f"Timestamp range: {timestamps[0]} to {timestamps[-1]}")
report.append("")
report.append("RUN DURATION SUMMARY")
report.append("-" * 65)
report.append(run_summary.to_string(index=False))
report.append("")
report.append("RUN LENGTH HISTOGRAM")
report.append("-" * 65)
report.append(histogram.to_string(index=False))
report.append("")
report.append("PERCENTAGE OF STATES IN RUNS AT LEAST EACH DURATION")
report.append("-" * 65)
report.append(duration_summary.to_string(index=False))
report.append("")
report.append("Interpretation:")
report.append(
    "A run is a consecutive sequence of states with the same label "
    "inside the same segment."
)
report.append(
    "A long run describes label persistence in this dataset; it does "
    "not necessarily represent a continuous real-world attack episode."
)
report.append(
    "The results describe the test partition only."
)

outputs[4].write_text("\n".join(report), encoding="utf-8")

print("\n" + "=" * 65)
print("LABEL RUN STRUCTURE ANALYSIS COMPLETE")
print("=" * 65)

print("\nRun duration summary:")
print(run_summary.to_string(index=False))

print("\nRun length histogram:")
print(histogram.to_string(index=False))

print("\nStates in runs meeting duration thresholds:")
print(duration_summary.to_string(index=False))

print("\nOutput folder:", out_dir)
print("Report:", outputs[4])
