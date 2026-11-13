from pathlib import Path
from collections import Counter

import csv
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

ROOT = Path(r"F:\Projects\SentinelX")
DATA = ROOT / "Datasets"
MODEL_DIR = ROOT / "Models" / "lstm_world_model_v1"

OUTPUT = MODEL_DIR / "anomaly_experiment_v1"
CHECKPOINT = MODEL_DIR / "best_model.pt"

BATCH_SIZE = 64
FEATURES = 206
HIDDEN = 128

if OUTPUT.exists():
    raise FileExistsError(
        f"Output already exists: {OUTPUT}. Not overwriting."
    )

OUTPUT.mkdir(parents=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------
# 2. Model architecture
# ---------------------------------------------------------

class SentinelXLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=FEATURES,
            hidden_size=HIDDEN,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.2)
        self.output_layer = nn.Linear(HIDDEN, FEATURES)

    def forward(self, x):
        output, _ = self.lstm(x)
        final_state = output[:, -1, :]
        return self.output_layer(self.dropout(final_state))


model = SentinelXLSTM().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# ---------------------------------------------------------
# 3. Load data
# ---------------------------------------------------------

def load_array(name):
    path = DATA / name

    if not path.exists():
        raise FileNotFoundError(path)

    return np.load(path, mmap_mode="r")


def decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


X_val = load_array("sentinelx_validation_X.npy")
y_val = load_array("sentinelx_validation_y.npy")
labels_val = load_array("sentinelx_validation_target_labels.npy")

X_test = load_array("sentinelx_test_X.npy")
y_test = load_array("sentinelx_test_y.npy")
labels_test = load_array("sentinelx_test_target_labels.npy")
timestamps_test = load_array(
    "sentinelx_test_target_timestamps.npy"
)

print("=" * 65)
print("SENTINELX - ANOMALY DETECTION EXPERIMENT")
print("=" * 65)

print("Device:", DEVICE)
print("Validation sequences:", len(X_val))
print("Test sequences:", len(X_test))


# ---------------------------------------------------------
# 4. Calculate sequence-level prediction errors
# ---------------------------------------------------------

@torch.no_grad()
def calculate_scores(X, y, split_name):

    scores = np.empty(len(X), dtype=np.float32)

    for start in range(0, len(X), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(X))

        xb_np = np.array(
            X[start:end], dtype=np.float32, copy=True
        )

        yb_np = np.array(
            y[start:end], dtype=np.float32, copy=True
        )

        xb = torch.from_numpy(xb_np).to(DEVICE)

        predictions = model(xb).cpu().numpy()

        errors = np.abs(predictions - yb_np)

        # Mean absolute prediction error across 206 features.
        scores[start:end] = errors.mean(axis=1)

        if end % 2048 < BATCH_SIZE or end == len(X):
            print(f"{split_name}: {end}/{len(X)}")

    return scores


print("\nCalculating validation scores...")
val_scores = calculate_scores(X_val, y_val, "Validation")

print("\nCalculating test scores...")
test_scores = calculate_scores(X_test, y_test, "Test")


# ---------------------------------------------------------
# 5. Calibrate thresholds using BENIGN validation only
# ---------------------------------------------------------

val_labels = np.array(
    [decode(v) for v in labels_val]
)

benign_mask = val_labels == "Benign"

benign_scores = val_scores[benign_mask]

if len(benign_scores) < 100:
    raise ValueError("Insufficient benign validation samples.")

threshold_95 = float(
    np.percentile(benign_scores, 95)
)

threshold_99 = float(
    np.percentile(benign_scores, 99)
)

print("\n" + "=" * 65)
print("THRESHOLD CALIBRATION")
print("=" * 65)

print("Benign validation samples:", len(benign_scores))
print(f"95th percentile threshold: {threshold_95:.8f}")
print(f"99th percentile threshold: {threshold_99:.8f}")


# ---------------------------------------------------------
# 6. Test evaluation
# ---------------------------------------------------------

test_labels = np.array(
    [decode(v) for v in labels_test]
)

allowed_labels = {"Benign", "Bot"}

unexpected = set(test_labels) - allowed_labels

if unexpected:
    raise ValueError(
        f"Unexpected test labels: {unexpected}. "
        "Review the evaluation design before proceeding."
    )


def evaluate_threshold(threshold, name):

    # High prediction error = anomaly.
    predicted = test_scores > threshold

    # Bot is the positive class for this experiment.
    actual = test_labels == "Bot"

    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    tn = int(np.sum(~predicted & ~actual))
    fn = int(np.sum(~predicted & actual))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    fpr = fp / (fp + tn) if fp + tn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / len(test_labels)

    results = {
        "threshold": threshold,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "specificity": specificity,
        "accuracy": accuracy
    }

    print("\n" + "-" * 50)
    print(name)
    print("-" * 50)

    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    return results, predicted


results_95, predictions_95 = evaluate_threshold(
    threshold_95, "95th Percentile Threshold"
)

results_99, predictions_99 = evaluate_threshold(
    threshold_99, "99th Percentile Threshold"
)


# ---------------------------------------------------------
# 7. Save threshold evaluation
# ---------------------------------------------------------

metrics_path = OUTPUT / "anomaly_detection_metrics.csv"

with metrics_path.open("w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(results_95.keys())
    )

    writer.writeheader()
    writer.writerow(results_95)
    writer.writerow(results_99)


# ---------------------------------------------------------
# 8. Save individual test predictions
# ---------------------------------------------------------

predictions_path = OUTPUT / "test_anomaly_predictions.csv"

with predictions_path.open(
    "w", newline="", encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "timestamp",
        "actual_label",
        "prediction_error",
        "anomaly_95",
        "anomaly_99"
    ])

    for i in range(len(test_scores)):

        writer.writerow([
            decode(timestamps_test[i]),
            test_labels[i],
            f"{test_scores[i]:.8f}",
            bool(predictions_95[i]),
            bool(predictions_99[i])
        ])


# ---------------------------------------------------------
# 9. Save experiment report
# ---------------------------------------------------------

report = f"""
SENTINELX - ANOMALY DETECTION EXPERIMENT

Model checkpoint epoch: {checkpoint['epoch']}

CALIBRATION:
Validation benign samples: {len(benign_scores)}
95th percentile threshold: {threshold_95}
99th percentile threshold: {threshold_99}

TEST:
Total sequences: {len(test_scores)}
Label counts: {dict(Counter(test_labels))}

95TH PERCENTILE RESULTS:
{results_95}

99TH PERCENTILE RESULTS:
{results_99}

SCORING:
Sequence anomaly score = mean absolute prediction error
across all 206 preprocessed network-state features.

Anomaly decision:
Score > threshold means anomaly.

POSITIVE CLASS:
Bot

IMPORTANT LIMITATIONS:
This is an experimental evaluation on the available CIC-IDS-2018
test partition. The test partition contains Benign and Bot labels.
A high prediction error is not proof of malicious activity.

The threshold was calibrated using benign validation samples
and was not selected using test labels.
"""

(OUTPUT / "anomaly_experiment_report.txt").write_text(
    report,
    encoding="utf-8"
)

print("\n" + "=" * 65)
print("ANOMALY EXPERIMENT COMPLETED")
print("=" * 65)

print("Output directory:", OUTPUT)
print("Metrics:", metrics_path)
print("Predictions:", predictions_path)
print("Report:", OUTPUT / "anomaly_experiment_report.txt")
