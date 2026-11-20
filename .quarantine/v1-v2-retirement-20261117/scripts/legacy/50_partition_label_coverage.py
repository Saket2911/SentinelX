from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"F:\Projects\SentinelX")
DATA = ROOT / "Datasets"
OUT = DATA / "diagnostics" / "partition_label_coverage_v1"

if OUT.exists():
    raise FileExistsError(
        f"Output already exists: {OUT}. Inspect before rerunning."
    )

OUT.mkdir(parents=True)

partitions = {
    "Train": DATA / "sentinelx_train_states.csv",
    "Validation": DATA / "sentinelx_validation_states.csv",
    "Test": DATA / "sentinelx_test_states.csv"
}

frames = {}

for name, path in partitions.items():
    if not path.exists():
        raise FileNotFoundError(path)

    print(f"Loading {name}...")
    df = pd.read_csv(path, usecols=["Timestamp", "Label", "Segment ID"])
    df["Partition"] = name
    frames[name] = df

all_rows = []
coverage_rows = []
transition_rows = []

for name, df in frames.items():
    counts = df["Label"].fillna("MISSING").value_counts()

    total = len(df)

    for label, count in counts.items():
        all_rows.append({
            "Partition": name,
            "Label": label,
            "State_Count": int(count),
            "Percentage": 100 * count / total
        })

    labels = df["Label"].fillna("MISSING").astype(str)
    unique_labels = set(labels.unique())

    for label in sorted(unique_labels):
        coverage_rows.append({
            "Partition": name,
            "Label": label,
            "Present": True,
            "State_Count": int((labels == label).sum())
        })

    # Count label transitions without crossing segment boundaries.
    transitions = 0
    segment_groups = df.groupby("Segment ID", sort=False)

    for _, group in segment_groups:
        values = group["Label"].fillna("MISSING").astype(str).to_numpy()
        if len(values) > 1:
            transitions += int(np.sum(values[1:] != values[:-1]))

    transition_rows.append({
        "Partition": name,
        "States": total,
        "Segments": int(df["Segment ID"].nunique()),
        "Label_Transitions": transitions,
        "Distinct_Labels": len(unique_labels)
    })

count_df = pd.DataFrame(all_rows)
coverage_df = pd.DataFrame(coverage_rows)
transition_df = pd.DataFrame(transition_rows)

# Build a complete class-presence matrix.
all_labels = sorted(
    set().union(*[
        set(df["Label"].fillna("MISSING").astype(str).unique())
        for df in frames.values()
    ])
)

matrix_rows = []

for label in all_labels:
    row = {"Label": label}

    for partition, df in frames.items():
        count = int(
            (df["Label"].fillna("MISSING").astype(str) == label).sum()
        )
        row[partition] = count
        row[f"{partition}_Present"] = count > 0

    matrix_rows.append(row)

matrix = pd.DataFrame(matrix_rows)

train_labels = set(
    frames["Train"]["Label"].fillna("MISSING").astype(str).unique()
)

validation_labels = set(
    frames["Validation"]["Label"].fillna("MISSING").astype(str).unique()
)

test_labels = set(
    frames["Test"]["Label"].fillna("MISSING").astype(str).unique()
)

unseen_validation = sorted(validation_labels - train_labels)
unseen_test = sorted(test_labels - train_labels)

# Save files.
count_df.to_csv(OUT / "partition_label_counts.csv", index=False)
coverage_df.to_csv(OUT / "partition_label_coverage_long.csv", index=False)
transition_df.to_csv(OUT / "partition_transition_summary.csv", index=False)
matrix.to_csv(OUT / "partition_label_matrix.csv", index=False)

report = [
    "SENTINELX PARTITION LABEL COVERAGE",
    "=" * 70,
    "",
    "LABEL DISTRIBUTION",
    "-" * 70,
    count_df.to_string(index=False),
    "",
    "LABEL PRESENCE MATRIX",
    "-" * 70,
    matrix.to_string(index=False),
    "",
    "TRANSITION SUMMARY",
    "-" * 70,
    transition_df.to_string(index=False),
    "",
    "VALIDATION LABELS ABSENT FROM TRAINING",
    "-" * 70,
    str(unseen_validation),
    "",
    "TEST LABELS ABSENT FROM TRAINING",
    "-" * 70,
    str(unseen_test),
    "",
    "INTERPRETATION",
    "-" * 70,
    "A class absent from a partition cannot be evaluated on that partition.",
    "A class present in test but absent from training is an unseen class.",
    "This report describes label coverage, not model performance.",
    "No source dataset or model files are modified."
]

report_path = OUT / "partition_label_coverage_report.txt"
report_path.write_text("\n".join(report), encoding="utf-8")

print("\n" + "=" * 70)
print("PARTITION LABEL COVERAGE COMPLETE")
print("=" * 70)

print("\nLabel distribution:")
print(count_df.to_string(index=False))

print("\nLabel matrix:")
print(matrix.to_string(index=False))

print("\nTransition summary:")
print(transition_df.to_string(index=False))

print("\nValidation labels absent from training:", unseen_validation)
print("Test labels absent from training:", unseen_test)

print("\nOutput folder:", OUT)
print("Report:", report_path)
