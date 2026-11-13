
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# SENTINELX V2 — FINAL TEST EVALUATION
# ============================================================

ROOT = Path(r"F:\Projects\SentinelX")
DATA_DIR = ROOT / "Datasets"
MODEL_DIR = ROOT / "Models" / "lstm_world_model_v2"

CHECKPOINT_PATH = MODEL_DIR / "best_model.pt"
CONFIG_PATH = MODEL_DIR / "config.json"

X_PATH = DATA_DIR / "sentinelx_test_X_seq10_v1.npy"
Y_PATH = DATA_DIR / "sentinelx_test_y_next_v1.npy"

REPORT_PATH = MODEL_DIR / "test_evaluation_report.txt"

BATCH_SIZE = 32


# ============================================================
# MODEL ARCHITECTURE — MATCHES V2 CHECKPOINT
# ============================================================

class SentinelXLSTM(nn.Module):

    def __init__(
        self,
        input_features,
        hidden_size,
        num_layers,
        dropout
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.output_layer = nn.Linear(
            hidden_size,
            input_features
        )

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        last_state = lstm_output[:, -1, :]
        return self.output_layer(last_state)


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(model, X, Y, device):

    total_absolute_error = 0.0
    total_squared_error = 0.0
    total_huber_loss = 0.0
    total_values = 0

    model.eval()

    with torch.no_grad():

        for start in range(0, len(X), BATCH_SIZE):

            end = min(start + BATCH_SIZE, len(X))

            x_batch = torch.from_numpy(
                np.array(X[start:end], dtype=np.float32, copy=True)
            ).to(device)

            y_batch = torch.from_numpy(
                np.array(Y[start:end], dtype=np.float32, copy=True)
            ).to(device)

            predictions = model(x_batch)

            errors = predictions - y_batch
            absolute_errors = torch.abs(errors)

            huber = torch.where(
                absolute_errors <= 1.0,
                0.5 * errors.square(),
                absolute_errors - 0.5
            )

            total_absolute_error += absolute_errors.sum().item()
            total_squared_error += errors.square().sum().item()
            total_huber_loss += huber.sum().item()

            total_values += errors.numel()

    return {
        "huber": total_huber_loss / total_values,
        "mae": total_absolute_error / total_values,
        "rmse": (total_squared_error / total_values) ** 0.5
    }


# ============================================================
# VALIDATION CHECKS
# ============================================================

print("=" * 70)
print("SENTINELX V2 — FINAL TEST EVALUATION")
print("=" * 70)

required_files = [
    CHECKPOINT_PATH,
    CONFIG_PATH,
    X_PATH,
    Y_PATH
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")

if REPORT_PATH.exists():
    raise FileExistsError(
        f"Report already exists. Rename it before running again:\n{REPORT_PATH}"
    )

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False
)

assert checkpoint["sequence_length"] == 10
assert checkpoint["input_features"] == 212
assert checkpoint["hidden_size"] == 128
assert checkpoint["num_layers"] == 1

print("\nCheckpoint configuration verified.")


# ============================================================
# LOAD TEST SEQUENCES
# ============================================================

print("\nLoading untouched test sequences...")

X_test = np.load(X_PATH, mmap_mode="r")
Y_test = np.load(Y_PATH, mmap_mode="r")

print("X test shape:", X_test.shape)
print("Y test shape:", Y_test.shape)

assert X_test.ndim == 3
assert Y_test.ndim == 2
assert X_test.shape[1] == 10
assert X_test.shape[2] == 212
assert Y_test.shape[1] == 212
assert X_test.shape[0] == Y_test.shape[0]

if not np.isfinite(X_test).all():
    raise ValueError("Non-finite values found in X_test.")

if not np.isfinite(Y_test).all():
    raise ValueError("Non-finite values found in Y_test.")

print("Test sequence checks passed.")
print("Test samples:", len(X_test))


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nEvaluation device:", device)

model = SentinelXLSTM(
    input_features=checkpoint["input_features"],
    hidden_size=checkpoint["hidden_size"],
    num_layers=checkpoint["num_layers"],
    dropout=checkpoint["dropout"]
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print("V2 checkpoint loaded successfully.")


# ============================================================
# LSTM TEST METRICS
# ============================================================

print("\nEvaluating LSTM V2...")

lstm_metrics = calculate_metrics(
    model,
    X_test,
    Y_test,
    device
)


# ============================================================
# PERSISTENCE BASELINE
# Last observed state predicts the next state
# ============================================================

print("Evaluating persistence baseline...")

persistence_metrics = {
    "huber": 0.0,
    "mae": 0.0,
    "rmse": 0.0
}

total_absolute_error = 0.0
total_squared_error = 0.0
total_huber_loss = 0.0
total_values = 0

for start in range(0, len(X_test), 1024):

    end = min(start + 1024, len(X_test))

    last_state = np.asarray(
        X_test[start:end, -1, :],
        dtype=np.float64
    )

    actual = np.asarray(
        Y_test[start:end],
        dtype=np.float64
    )

    errors = last_state - actual
    absolute_errors = np.abs(errors)

    huber = np.where(
        absolute_errors <= 1.0,
        0.5 * errors ** 2,
        absolute_errors - 0.5
    )

    total_absolute_error += absolute_errors.sum()
    total_squared_error += (errors ** 2).sum()
    total_huber_loss += huber.sum()

    total_values += errors.size

persistence_metrics = {
    "huber": total_huber_loss / total_values,
    "mae": total_absolute_error / total_values,
    "rmse": np.sqrt(total_squared_error / total_values)
}


# ============================================================
# COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print("\nLSTM V2:")
for metric, value in lstm_metrics.items():
    print(f"{metric.upper():8s}: {value:.8f}")

print("\nPERSISTENCE BASELINE:")
for metric, value in persistence_metrics.items():
    print(f"{metric.upper():8s}: {value:.8f}")

print("\nLSTM IMPROVEMENT OVER PERSISTENCE:")

improvements = {}

for metric in ["huber", "mae", "rmse"]:

    baseline = persistence_metrics[metric]
    lstm = lstm_metrics[metric]

    improvement = (
        (baseline - lstm) / baseline * 100
        if baseline != 0 else 0.0
    )

    improvements[metric] = improvement

    print(f"{metric.upper():8s}: {improvement:.2f}%")


# ============================================================
# SAVE REPORT
# ============================================================

report = [
    "SENTINELX V2 — FINAL TEST EVALUATION",
    "=" * 60,
    f"Test samples: {len(X_test)}",
    f"Evaluation device: {device}",
    f"Best checkpoint epoch: {checkpoint['epoch']}",
    "",
    "LSTM V2:",
]

for metric, value in lstm_metrics.items():
    report.append(f"{metric.upper()}: {value:.10f}")

report.extend(["", "PERSISTENCE BASELINE:"])

for metric, value in persistence_metrics.items():
    report.append(f"{metric.upper()}: {value:.10f}")

report.extend(["", "IMPROVEMENT OVER PERSISTENCE:"])

for metric, value in improvements.items():
    report.append(f"{metric.upper()}: {value:.2f}%")

report.extend([
    "",
    "Test partition evaluated only after model training.",
    "No model retraining or checkpoint modification performed."
])

REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

print("\n" + "=" * 70)
print("TEST EVALUATION COMPLETED")
print("=" * 70)
print("Report saved to:", REPORT_PATH)
print("The trained checkpoint was not modified.")
