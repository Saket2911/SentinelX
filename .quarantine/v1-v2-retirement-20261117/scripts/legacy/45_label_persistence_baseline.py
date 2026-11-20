from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(r"F:\Projects\SentinelX")
DATA = ROOT / "Datasets"
MODEL_DIR = ROOT / "Models" / "lstm_world_model_v1"

LABEL_PATH = DATA / "sentinelx_test_target_labels.npy"
TIME_PATH = DATA / "sentinelx_test_target_timestamps.npy"
SEGMENT_PATH = DATA / "sentinelx_test_target_segment_ids.npy"

# If the test arrays are stored elsewhere, locate them in the Datasets folder.
for path in [LABEL_PATH, TIME_PATH, SEGMENT_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

labels = np.load(LABEL_PATH, allow_pickle=False).astype(str)
timestamps = pd.to_datetime(np.load(TIME_PATH, allow_pickle=False))
segments = np.load(SEGMENT_PATH, allow_pickle=False)

if not (len(labels) == len(timestamps) == len(segments)):
    raise ValueError("Labels, timestamps and segment IDs have different lengths.")

if len(labels) < 31:
    raise ValueError("Not enough test states for a 30-second horizon.")

print("Loaded test timeline:", len(labels), "states")
print("Label counts:")
print(pd.Series(labels).value_counts().to_string())
print("Timestamp range:", timestamps[0], "to", timestamps[-1])

horizons = [1, 2, 5, 10, 30]
rows = []

for horizon in horizons:
    y_true = []
    y_pred = []
    skipped = 0

    # Origin is state i-1. Target is state i+horizon-1.
    for i in range(1, len(labels) - horizon + 1):
        origin_idx = i - 1
        target_idx = i + horizon - 1

        # Do not compare across independent segments.
        if segments[origin_idx] != segments[target_idx]:
            skipped += 1
            continue

        # Confirm that the target is exactly horizon seconds after origin.
        elapsed = (timestamps[target_idx] - timestamps[origin_idx]).total_seconds()
        if elapsed != horizon:
            skipped += 1
            continue

        y_pred.append(labels[origin_idx])
        y_true.append(labels[target_idx])

    y_true = np.asarray(y_true, dtype=str)
    y_pred = np.asarray(y_pred, dtype=str)

    if len(y_true) == 0:
        raise RuntimeError(f"No valid samples for horizon {horizon}.")

    accuracy = accuracy_score(y_true, y_pred)
    balanced = balanced_accuracy_score(y_true, y_pred)

    # Bot is treated as the positive class.
    precision = precision_score(
        y_true, y_pred, pos_label="Bot", zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, pos_label="Bot", zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, pos_label="Bot", zero_division=0
    )

    matrix = confusion_matrix(
        y_true, y_pred, labels=["Benign", "Bot"]
    )

    benign_recall = (
        matrix[0, 0] / matrix[0].sum()
        if matrix[0].sum() else 0.0
    )
    bot_recall = (
        matrix[1, 1] / matrix[1].sum()
        if matrix[1].sum() else 0.0
    )

    rows.append({
        "Horizon_Seconds": horizon,
        "Samples": len(y_true),
        "Skipped_Boundary_or_Time": skipped,
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced,
        "Bot_Precision": precision,
        "Bot_Recall": recall,
        "Bot_F1": f1,
        "Benign_Recall": benign_recall,
        "Bot_Recall_Check": bot_recall,
        "TN_Benign_Predicted_Benign": int(matrix[0, 0]),
        "FP_Benign_Predicted_Bot": int(matrix[0, 1]),
        "FN_Bot_Predicted_Benign": int(matrix[1, 0]),
        "TP_Bot_Predicted_Bot": int(matrix[1, 1]),
    })

results = pd.DataFrame(rows)

out_dir = MODEL_DIR / "label_persistence_baseline_v1"
out_dir.mkdir(parents=True, exist_ok=True)

csv_path = out_dir / "label_persistence_metrics.csv"
report_path = out_dir / "label_persistence_report.txt"

if csv_path.exists() or report_path.exists():
    raise FileExistsError(
        f"Output already exists in {out_dir}. "
        "Move or rename the previous output before rerunning."
    )

results.to_csv(csv_path, index=False)

report = []
report.append("SENTINELX LABEL PERSISTENCE BASELINE")
report.append("=" * 65)
report.append("")
report.append("Purpose:")
report.append(
    "Measure how well the previous second's Benign/Bot label "
    "persists into a future target state."
)
report.append("")
report.append("This is a label-persistence reference, not the LSTM's")
report.append("numerical forecasting score and not an attack detector.")
report.append("")
report.append(f"Test states: {len(labels)}")
report.append(f"Timestamp range: {timestamps[0]} to {timestamps[-1]}")
report.append("")
report.append("Evaluation alignment:")
report.append("Origin label = labels[i-1]")
report.append("Future label = labels[i+horizon-1]")
report.append("Only same-segment samples exactly horizon seconds apart are used.")
report.append("")
report.append("Metrics:")
report.append(results.to_string(index=False))
report.append("")
report.append("Interpretation notes:")
report.append(
    "- Accuracy can be misleading because Bot is the majority label."
)
report.append(
    "- Balanced accuracy gives equal weight to Benign and Bot recall."
)
report.append(
    "- Bot precision/recall/F1 evaluate persistence of the Bot label only."
)
report.append(
    "- This experiment does not evaluate the LSTM's predicted numeric states."
)
report.append(
    "- It does not establish early warning or real-world attack detection."
)

report_text = "\n".join(report)
report_path.write_text(report_text, encoding="utf-8")

print("\n" + "=" * 65)
print("LABEL PERSISTENCE BASELINE COMPLETE")
print("=" * 65)
print(results.to_string(index=False))
print("\nCSV:", csv_path)
print("Report:", report_path)
