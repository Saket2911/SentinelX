import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(r"F:\Projects\SentinelX")
DATA = ROOT / "Datasets"
OUTPUT = DATA / "diagnostics" / "label_transition_integrity_v1"

if OUTPUT.exists():
    raise FileExistsError(
        f"Output already exists: {OUTPUT}. "
        "Inspect it before choosing another output folder."
    )

OUTPUT.mkdir(parents=True)

composition_path = (
    DATA / "diagnostics" / "mixed_traffic_analysis_v1"
    / "per_second_traffic_composition.csv"
)

labels_path = DATA / "sentinelx_test_target_labels.npy"
timestamps_path = DATA / "sentinelx_test_target_timestamps.npy"

for path in [composition_path, labels_path, timestamps_path]:
    if not path.exists():
        raise FileNotFoundError(path)

print("Loading traffic composition and test labels...")

composition = pd.read_csv(composition_path)
labels = np.load(labels_path)
timestamps = np.load(timestamps_path)

test = pd.DataFrame({
    "Timestamp": pd.to_datetime(timestamps),
    "Model_State_Label": labels.astype(str)
})

composition["Timestamp"] = pd.to_datetime(
    composition["Timestamp"], errors="coerce"
)

# Keep only the fields needed for this diagnostic.
fields = [
    "Timestamp", "Flow Count", "Benign Flow Count",
    "Attack Flow Count", "Attack Flow Ratio",
    "Has Attack", "Mixed Benign Attack",
    "Attack Label Counts"
]

merged = test.merge(
    composition[fields],
    on="Timestamp",
    how="left",
    validate="one_to_one"
)

if merged["Attack Flow Count"].isna().any():
    missing = int(merged["Attack Flow Count"].isna().sum())
    raise ValueError(f"{missing} test timestamps did not match composition data.")

merged["Composition_Label"] = np.where(
    merged["Has Attack"] == 1, "Bot", "Benign"
)

merged["Label_Match"] = (
    merged["Model_State_Label"] == merged["Composition_Label"]
)

# Consecutive runs of the same model state label.
merged["Run_ID"] = (
    merged["Model_State_Label"] !=
    merged["Model_State_Label"].shift()
).cumsum()

runs = (
    merged.groupby(["Run_ID", "Model_State_Label"], as_index=False)
    .agg(
        Start_Timestamp=("Timestamp", "first"),
        End_Timestamp=("Timestamp", "last"),
        Duration_Seconds=("Timestamp", "size"),
        Mean_Attack_Ratio=("Attack Flow Ratio", "mean"),
        Total_Attack_Flows=("Attack Flow Count", "sum"),
        Total_Flows=("Flow Count", "sum")
    )
)

# Identify each label transition and attach traffic composition.
merged["Previous_Label"] = merged["Model_State_Label"].shift()
transitions = merged[
    merged["Model_State_Label"] != merged["Previous_Label"]
].copy()

transitions["Transition"] = (
    transitions["Previous_Label"].fillna("START")
    + "_to_"
    + transitions["Model_State_Label"]
)

transition_columns = [
    "Timestamp", "Transition", "Previous_Label",
    "Model_State_Label", "Flow Count",
    "Benign Flow Count", "Attack Flow Count",
    "Attack Flow Ratio", "Attack Label Counts"
]

transitions[transition_columns].to_csv(
    OUTPUT / "transition_traffic_details.csv", index=False
)

merged.to_csv(OUTPUT / "test_label_composition_join.csv", index=False)
runs.to_csv(OUTPUT / "consecutive_label_runs.csv", index=False)

summary_rows = []

for label, group in merged.groupby("Model_State_Label"):
    summary_rows.append({
        "Label": label,
        "States": len(group),
        "Mean_Flow_Count": group["Flow Count"].mean(),
        "Mean_Attack_Flow_Count": group["Attack Flow Count"].mean(),
        "Mean_Attack_Flow_Ratio": group["Attack Flow Ratio"].mean(),
        "Median_Attack_Flow_Ratio": group["Attack Flow Ratio"].median(),
        "Zero_Attack_Flow_States": int(
            (group["Attack Flow Count"] == 0).sum()
        ),
        "Mean_Benign_Flow_Count": group["Benign Flow Count"].mean()
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT / "traffic_composition_by_label.csv", index=False)

# Run-length statistics.
run_summary = (
    runs.groupby("Model_State_Label")["Duration_Seconds"]
    .agg(["count", "mean", "median", "min", "max"])
    .reset_index()
)
run_summary.to_csv(OUTPUT / "run_length_summary.csv", index=False)

# Overall transition counts.
transition_counts = (
    transitions["Transition"].value_counts()
    .rename_axis("Transition")
    .reset_index(name="Count")
)
transition_counts.to_csv(
    OUTPUT / "transition_counts.csv", index=False
)

# Report.
report_path = OUTPUT / "label_transition_integrity_report.txt"

with report_path.open("w", encoding="utf-8") as f:
    f.write("SENTINELX - LABEL TRANSITION INTEGRITY ANALYSIS\n")
    f.write("=" * 65 + "\n\n")

    f.write(f"Test states: {len(merged)}\n")
    f.write(f"Matched composition records: {len(merged)}\n")
    f.write(f"Unmatched timestamps: {merged['Attack Flow Count'].isna().sum()}\n")
    f.write(f"Label/composition matches: {merged['Label_Match'].sum()}\n")
    f.write(f"Label/composition mismatches: {(~merged['Label_Match']).sum()}\n")
    f.write(f"Total label transitions: {len(transitions)}\n")
    f.write(f"Total consecutive runs: {len(runs)}\n\n")

    f.write("TRAFFIC COMPOSITION BY MODEL LABEL\n")
    f.write("-" * 65 + "\n")
    f.write(summary.to_string(index=False))
    f.write("\n\nRUN LENGTH SUMMARY (SECONDS)\n")
    f.write("-" * 65 + "\n")
    f.write(run_summary.to_string(index=False))
    f.write("\n\nTRANSITION COUNTS\n")
    f.write("-" * 65 + "\n")
    f.write(transition_counts.to_string(index=False))
    f.write("\n\nINTERPRETATION\n")
    f.write("-" * 65 + "\n")
    f.write(
        "This analysis checks whether the state labels correspond to "
        "the presence of attack flows in each second.\n"
    )
    f.write(
        "A Bot label indicates attack-flow presence under the existing "
        "any-attack labeling rule; it does not indicate that Bot traffic "
        "is the majority of flows.\n"
    )
    f.write(
        "Rapid label transitions may reflect intermittent attack-flow "
        "presence rather than repeated starts and stops of a real attack.\n"
    )
    f.write(
        "This is a dataset-label diagnostic, not an attack-detection "
        "performance evaluation.\n"
    )

print("\n" + "=" * 65)
print("LABEL TRANSITION INTEGRITY ANALYSIS COMPLETE")
print("=" * 65)

print("\nLabel/composition matches:",
      int(merged["Label_Match"].sum()))
print("Label/composition mismatches:",
      int((~merged["Label_Match"]).sum()))
print("Transitions:", len(transitions))

print("\nTraffic composition by label:")
print(summary.to_string(index=False))

print("\nRun length summary:")
print(run_summary.to_string(index=False))

print("\nTransition counts:")
print(transition_counts.to_string(index=False))

print("\nOutput folder:", OUTPUT)
print("Report:", report_path)
