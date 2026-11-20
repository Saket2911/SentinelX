
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(r"F:\Projects\SentinelX")
DATA = ROOT / "Datasets"
MODEL_DIR = ROOT / "Models" / "lstm_world_model_v2"

CHECKPOINT = MODEL_DIR / "best_model.pt"

VAL_X = DATA / "sentinelx_validation_X_seq10_v1.npy"
VAL_Y = DATA / "sentinelx_validation_y_next_v1.npy"
VAL_META = DATA / "sentinelx_validation_sequence_metadata_v1.csv"

TEST_X = DATA / "sentinelx_test_X_seq10_v1.npy"
TEST_Y = DATA / "sentinelx_test_y_next_v1.npy"
TEST_META = DATA / "sentinelx_test_sequence_metadata_v1.csv"

REPORT = MODEL_DIR / "anomaly_score_evaluation.txt"

BATCH_SIZE = 32


class SentinelXLSTM(nn.Module):

    def __init__(self, input_features, hidden_size,
                 num_layers, dropout):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.output_layer = nn.Linear(
            hidden_size, input_features
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.output_layer(output[:, -1, :])


def calculate_anomaly_scores(model, X, Y, device):

    scores = np.empty(len(X), dtype=np.float64)

    model.eval()

    with torch.no_grad():

        for start in range(0, len(X), BATCH_SIZE):

            end = min(start + BATCH_SIZE, len(X))

            xb = torch.from_numpy(
                np.array(X[start:end], dtype=np.float32, copy=True)
            ).to(device)

            yb = torch.from_numpy(
                np.array(Y[start:end], dtype=np.float32, copy=True)
            ).to(device)

            predictions = model(xb)

            # Mean absolute prediction error per sequence
            batch_scores = torch.mean(
                torch.abs(predictions - yb),
                dim=1
            )

            scores[start:end] = batch_scores.cpu().numpy()

    return scores


def evaluate_threshold(scores, labels, threshold):

    # Bot = positive; Benign = negative.
    # Other labels are excluded from this binary evaluation.

    bot = labels == "Bot"
    benign = labels == "Benign"

    bot_flagged = scores[bot] > threshold
    benign_flagged = scores[benign] > threshold

    tp = int(bot_flagged.sum())
    fn = int((~bot_flagged).sum())
    fp = int(benign_flagged.sum())
    tn = int((~benign_flagged).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "threshold": threshold,
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "Bot detection rate": recall * 100,
        "Benign false positive rate": (1 - specificity) * 100,
        "Precision": precision * 100,
        "F1": f1
    }


print("=" * 75)
print("SENTINELX — WORLD MODEL ANOMALY SCORE EVALUATION")
print("=" * 75)

required = [
    CHECKPOINT,
    VAL_X, VAL_Y, VAL_META,
    TEST_X, TEST_Y, TEST_META
]

for path in required:
    if not path.exists():
        raise FileNotFoundError(path)

if REPORT.exists():
    raise FileExistsError(
        f"Report already exists: {REPORT}"
    )


# Load metadata and arrays
val_meta = pd.read_csv(VAL_META)
test_meta = pd.read_csv(TEST_META)

X_val = np.load(VAL_X, mmap_mode="r")
Y_val = np.load(VAL_Y, mmap_mode="r")

X_test = np.load(TEST_X, mmap_mode="r")
Y_test = np.load(TEST_Y, mmap_mode="r")

val_labels = val_meta["Target Label"].astype(str).to_numpy()
test_labels = test_meta["Target Label"].astype(str).to_numpy()

assert len(X_val) == len(Y_val) == len(val_labels)
assert len(X_test) == len(Y_test) == len(test_labels)

assert X_val.shape[1:] == (10, 212)
assert X_test.shape[1:] == (10, 212)


# Load exact trained checkpoint
checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

model = SentinelXLSTM(
    checkpoint["input_features"],
    checkpoint["hidden_size"],
    checkpoint["num_layers"],
    checkpoint["dropout"]
)

model.load_state_dict(checkpoint["model_state_dict"])

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()

print("\nCheckpoint loaded.")
print("Best epoch:", checkpoint["epoch"])
print("Device:", device)


# Calculate validation and test anomaly scores
print("\nCalculating validation anomaly scores...")

val_scores = calculate_anomaly_scores(
    model, X_val, Y_val, device
)

print("Calculating test anomaly scores...")

test_scores = calculate_anomaly_scores(
    model, X_test, Y_test, device
)


# Calibrate using validation Benign samples only
val_benign_scores = val_scores[val_labels == "Benign"]

if len(val_benign_scores) == 0:
    raise ValueError("No benign validation samples found.")

threshold_95 = np.percentile(val_benign_scores, 95)
threshold_99 = np.percentile(val_benign_scores, 99)

print("\n" + "=" * 75)
print("VALIDATION BENIGN THRESHOLD CALIBRATION")
print("=" * 75)

print("Benign calibration samples:", len(val_benign_scores))
print("Mean score:", np.mean(val_benign_scores))
print("Median score:", np.median(val_benign_scores))
print("95th percentile threshold:", threshold_95)
print("99th percentile threshold:", threshold_99)


# Test score distributions by label
print("\n" + "=" * 75)
print("TEST ANOMALY SCORE DISTRIBUTIONS")
print("=" * 75)

distribution_rows = []

for label in sorted(np.unique(test_labels)):

    group = test_scores[test_labels == label]

    row = {
        "Label": label,
        "Samples": len(group),
        "Mean": np.mean(group),
        "Median": np.median(group),
        "P95": np.percentile(group, 95),
        "P99": np.percentile(group, 99)
    }

    distribution_rows.append(row)

distribution_df = pd.DataFrame(distribution_rows)

print(distribution_df.to_string(
    index=False,
    float_format=lambda x: f"{x:.6f}"
))


# Evaluate fixed thresholds
print("\n" + "=" * 75)
print("TEST ANOMALY DETECTION RESULTS")
print("=" * 75)

results = []

for name, threshold in [
    ("95th percentile", threshold_95),
    ("99th percentile", threshold_99)
]:

    result = evaluate_threshold(
        test_scores,
        test_labels,
        threshold
    )

    result["Threshold type"] = name
    results.append(result)

results_df = pd.DataFrame(results)

print(results_df.to_string(
    index=False,
    float_format=lambda x: f"{x:.4f}"
))


# Save report
report_lines = [
    "SENTINELX — WORLD MODEL ANOMALY SCORE EVALUATION",
    "=" * 75,
    f"Checkpoint epoch: {checkpoint['epoch']}",
    f"Device: {device}",
    "",
    "Thresholds calibrated using validation Benign samples only:",
    f"95th percentile: {threshold_95:.10f}",
    f"99th percentile: {threshold_99:.10f}",
    "",
    "TEST SCORE DISTRIBUTIONS:",
    distribution_df.to_string(index=False),
    "",
    "TEST DETECTION RESULTS:",
    results_df.to_string(index=False),
    "",
    "Bot is treated as the positive class.",
    "Benign is treated as the negative class.",
    "NO_FLOWS and other labels are excluded from binary metrics.",
    "No model training or checkpoint modification performed."
]

REPORT.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print("\nReport saved:", REPORT)
print("Evaluation completed.")
