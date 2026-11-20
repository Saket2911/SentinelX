from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"F:\Projects\SentinelX")
DATA = ROOT / "Datasets"
MODEL_DIR = ROOT / "Models" / "lstm_world_model_v1"

labels = np.load(
    DATA / "sentinelx_test_target_labels.npy"
).astype(str)

timestamps = pd.to_datetime(
    np.load(DATA / "sentinelx_test_target_timestamps.npy")
)

segments = np.load(
    DATA / "sentinelx_test_target_segment_ids.npy"
)

if not (len(labels) == len(timestamps) == len(segments)):
    raise ValueError("Array length mismatch.")

# Identify transitions between consecutive states.
transition_indices = np.where(
    (labels[1:] != labels[:-1]) |
    (segments[1:] != segments[:-1])
)[0] + 1

# For every origin, identify the next transition strictly after it.
next_transition = np.full(len(labels), -1, dtype=int)

for transition_idx in transition_indices:
    previous = next_transition[:transition_idx]
    unset = np.where(previous == -1)[0]
    next_transition[unset] = transition_idx

# Define lead-time groups using the next future transition.
lead_groups = [
    (1, 5, "1-5 seconds"),
    (6, 10, "6-10 seconds"),
    (11, 30, "11-30 seconds"),
    (31, None, "Over 30 seconds"),
]

horizons = [1, 2, 5, 10, 30]
rows = []

for horizon in horizons:
    for low, high, group_name in lead_groups:
        y_true = []
        y_pred = []

        for origin in range(len(labels) - horizon):
            target = origin + horizon

            transition_idx = next_transition[origin]

            if transition_idx == -1:
                continue

            # Only evaluate origins with a future transition
            # in the same segment.
            if segments[origin] != segments[transition_idx]:
                continue

            lead_seconds = (
                timestamps[transition_idx] -
                timestamps[origin]
            ).total_seconds()

            if lead_seconds < low:
                continue

            if high is not None and lead_seconds > high:
                continue

            if segments[origin] != segments[target]:
                continue

            elapsed = (
                timestamps[target] -
                timestamps[origin]
            ).total_seconds()

            if elapsed != horizon:
                continue

            y_pred.append(labels[origin])
            y_true.append(labels[target])

        y_true = np.asarray(y_true, dtype=str)
        y_pred = np.asarray(y_pred, dtype=str)

        if len(y_true) == 0:
            continue

        tn = int(np.sum(
            (y_true == "Benign") & (y_pred == "Benign")
        ))
        fp = int(np.sum(
            (y_true == "Benign") & (y_pred == "Bot")
        ))
        fn = int(np.sum(
            (y_true == "Bot") & (y_pred == "Benign")
        ))
        tp = int(np.sum(
            (y_true == "Bot") & (y_pred == "Bot")
        ))

        benign_recall = tn / (tn + fp) if tn + fp else 0
        bot_recall = tp / (tp + fn) if tp + fn else 0

        precision = tp / (tp + fp) if tp + fp else 0
        f1 = (
            2 * precision * bot_recall /
            (precision + bot_recall)
            if precision + bot_recall else 0
        )

        rows.append({
            "Horizon_Seconds": horizon,
            "Time_Until_Next_Transition": group_name,
            "Samples": len(y_true),
            "Accuracy": (tn + tp) / len(y_true),
            "Balanced_Accuracy": (
                benign_recall + bot_recall
            ) / 2,
            "Bot_Precision": precision,
            "Bot_Recall": bot_recall,
            "Bot_F1": f1,
            "Benign_Recall": benign_recall,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        })

results = pd.DataFrame(rows)

out_dir = MODEL_DIR / "pre_transition_persistence_v1"
out_dir.mkdir(parents=True, exist_ok=True)

csv_path = out_dir / "pre_transition_persistence.csv"
report_path = out_dir / "pre_transition_persistence_report.txt"

if csv_path.exists() or report_path.exists():
    raise FileExistsError(
        f"Output already exists in {out_dir}. "
        "Inspect it before rerunning."
    )

results.to_csv(csv_path, index=False)

report = [
    "SENTINELX PRE-TRANSITION LABEL PERSISTENCE ANALYSIS",
    "=" * 70,
    "",
    f"Test states: {len(labels)}",
    f"Future transitions: {len(transition_indices)}",
    "",
    "Origins are grouped by time until their next future transition.",
    "Only transitions within the same segment are considered.",
    "The prediction repeats the origin label.",
    "",
    results.to_string(index=False),
    "",
    "IMPORTANT:",
    "This is a descriptive label-persistence experiment.",
    "It is not LSTM performance or attack-warning performance.",
    "Some origins may contribute to multiple horizon evaluations.",
    "The lead-time groups are based on the observed future timeline."
]

report_path.write_text("\n".join(report), encoding="utf-8")

print("\n" + "=" * 70)
print("PRE-TRANSITION PERSISTENCE ANALYSIS COMPLETE")
print("=" * 70)
print(results.to_string(index=False))
print("\nCSV:", csv_path)
print("Report:", report_path)
