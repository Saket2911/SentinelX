from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"F:\Projects\SentinelX")
DATA = ROOT / "Datasets"
MODEL_DIR = ROOT / "Models" / "lstm_world_model_v1"

labels = np.load(
    DATA / "sentinelx_test_target_labels.npy"
).astype(str)

timestamps = pd.to_datetime(np.load(
    DATA / "sentinelx_test_target_timestamps.npy"
))

segments = np.load(
    DATA / "sentinelx_test_target_segment_ids.npy"
)

if not (len(labels) == len(timestamps) == len(segments)):
    raise ValueError("Array length mismatch.")

# A transition occurs at the first state with a changed label
# or a changed segment ID.
transition_indices = np.where(
    (labels[1:] != labels[:-1]) |
    (segments[1:] != segments[:-1])
)[0] + 1

if len(transition_indices) == 0:
    raise ValueError("No transitions found.")

# Distance in seconds to the nearest transition.
# The test timeline has already been checked for one-second intervals.
all_indices = np.arange(len(labels))
nearest_distance = np.min(
    np.abs(
        all_indices[:, None] -
        transition_indices[None, :]
    ),
    axis=1
)

NEAR_WINDOW = 5
near_transition = nearest_distance <= NEAR_WINDOW

print("Total test states:", len(labels))
print("Transitions:", len(transition_indices))
print("States within 5 seconds of a transition:",
      int(near_transition.sum()))
print("States away from transitions:",
      int((~near_transition).sum()))

horizons = [1, 2, 5, 10, 30]
rows = []

for horizon in horizons:
    for group_name, group_mask in [
        ("Near_Transition", near_transition),
        ("Away_From_Transition", ~near_transition)
    ]:
        y_true = []
        y_pred = []

        for i in range(1, len(labels) - horizon + 1):
            origin = i - 1
            target = i + horizon - 1

            if segments[origin] != segments[target]:
                continue

            elapsed = (
                timestamps[target] - timestamps[origin]
            ).total_seconds()

            if elapsed != horizon:
                continue

            if not group_mask[origin]:
                continue

            # Persistence prediction: repeat the origin label.
            y_pred.append(labels[origin])
            y_true.append(labels[target])

        y_true = np.asarray(y_true, dtype=str)
        y_pred = np.asarray(y_pred, dtype=str)

        if len(y_true) == 0:
            continue

        benign_mask = y_true == "Benign"
        bot_mask = y_true == "Bot"

        tn = int(np.sum((y_true == "Benign") & (y_pred == "Benign")))
        fp = int(np.sum((y_true == "Benign") & (y_pred == "Bot")))
        fn = int(np.sum((y_true == "Bot") & (y_pred == "Benign")))
        tp = int(np.sum((y_true == "Bot") & (y_pred == "Bot")))

        benign_recall = (
            tn / (tn + fp) if (tn + fp) else 0.0
        )
        bot_recall = (
            tp / (tp + fn) if (tp + fn) else 0.0
        )

        accuracy = (tn + tp) / len(y_true)
        balanced_accuracy = (
            benign_recall + bot_recall
        ) / 2

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (
            2 * precision * bot_recall / (precision + bot_recall)
            if (precision + bot_recall) else 0.0
        )

        rows.append({
            "Horizon_Seconds": horizon,
            "Group": group_name,
            "Samples": len(y_true),
            "Accuracy": accuracy,
            "Balanced_Accuracy": balanced_accuracy,
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

out_dir = MODEL_DIR / "transition_conditioned_persistence_v1"
out_dir.mkdir(parents=True, exist_ok=True)

outputs = [
    out_dir / "transition_conditioned_persistence.csv",
    out_dir / "transition_conditioned_persistence_report.txt"
]

if any(p.exists() for p in outputs):
    raise FileExistsError(
        f"An output already exists in {out_dir}. "
        "Inspect it before rerunning."
    )

results.to_csv(outputs[0], index=False)

report = [
    "SENTINELX TRANSITION-CONDITIONED LABEL PERSISTENCE",
    "=" * 70,
    "",
    f"Test states: {len(labels)}",
    f"Label/segment transitions: {len(transition_indices)}",
    f"Near-transition window: {NEAR_WINDOW} seconds from transition",
    f"Near-transition states: {int(near_transition.sum())}",
    f"Away-from-transition states: {int((~near_transition).sum())}",
    "",
    "METRICS",
    "-" * 70,
    results.to_string(index=False),
    "",
    "INTERPRETATION",
    "-" * 70,
    "The persistence predictor repeats the label at the origin state.",
    "Near-transition and away-from-transition groups are defined",
    "using the origin state's distance to the nearest transition.",
    "The near-transition group may include origins immediately before",
    "or after a transition.",
    "This is a label-persistence diagnostic, not LSTM performance",
    "and not a real-world attack-detection evaluation.",
    "The five-second window is an analysis choice, not a learned threshold."
]

outputs[1].write_text("\n".join(report), encoding="utf-8")

print("\n" + "=" * 70)
print("TRANSITION-CONDITIONED PERSISTENCE ANALYSIS COMPLETE")
print("=" * 70)
print(results.to_string(index=False))
print("\nCSV:", outputs[0])
print("Report:", outputs[1])
