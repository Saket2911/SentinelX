
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(r"F:\Projects\SentinelX")
DATA = ROOT / "Datasets"
MODEL_DIR = ROOT / "Models" / "lstm_world_model_v2"

CHECKPOINT = MODEL_DIR / "best_model.pt"
X_PATH = DATA / "sentinelx_test_X_seq10_v1.npy"
Y_PATH = DATA / "sentinelx_test_y_next_v1.npy"
META_PATH = DATA / "sentinelx_test_sequence_metadata_v1.csv"

REPORT_PATH = MODEL_DIR / "test_metrics_by_label.txt"

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


def new_totals():
    return {
        "absolute": 0.0,
        "squared": 0.0,
        "huber": 0.0,
        "values": 0
    }


def add_errors(total, prediction, actual):

    error = prediction - actual
    absolute = np.abs(error)

    huber = np.where(
        absolute <= 1.0,
        0.5 * error ** 2,
        absolute - 0.5
    )

    total["absolute"] += absolute.sum()
    total["squared"] += (error ** 2).sum()
    total["huber"] += huber.sum()
    total["values"] += error.size


def finish_metrics(total):

    n = total["values"]

    return {
        "Huber": total["huber"] / n,
        "MAE": total["absolute"] / n,
        "RMSE": np.sqrt(total["squared"] / n)
    }


print("=" * 70)
print("SENTINELX V2 — TEST METRICS BY LABEL")
print("=" * 70)

for path in [CHECKPOINT, X_PATH, Y_PATH, META_PATH]:
    if not path.exists():
        raise FileNotFoundError(path)

if REPORT_PATH.exists():
    raise FileExistsError(
        f"Report already exists: {REPORT_PATH}"
    )


# Load metadata and verify alignment
metadata = pd.read_csv(META_PATH)

X = np.load(X_PATH, mmap_mode="r")
Y = np.load(Y_PATH, mmap_mode="r")

if len(metadata) != len(X) or len(X) != len(Y):
    raise ValueError("Metadata and sequence counts do not match.")

if "Target Label" not in metadata.columns:
    raise ValueError("Target Label column is missing.")

labels = metadata["Target Label"].astype(str).to_numpy()

if not np.isfinite(X).all() or not np.isfinite(Y).all():
    raise ValueError("Non-finite sequence values detected.")

print("Test samples:", len(X))
print("Labels:", pd.Series(labels).value_counts().to_dict())


# Load exact V2 checkpoint
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

print("Checkpoint loaded. Device:", device)


# Evaluate each label independently
results = []

for label in sorted(np.unique(labels)):

    indices = np.flatnonzero(labels == label)

    lstm_total = new_totals()
    persistence_total = new_totals()

    print(f"\nEvaluating label: {label} ({len(indices)} samples)")

    with torch.no_grad():

        for start in range(0, len(indices), BATCH_SIZE):

            batch_indices = indices[start:start + BATCH_SIZE]

            xb = torch.from_numpy(
                np.array(X[batch_indices], dtype=np.float32, copy=True)
            ).to(device)

            yb = np.array(
                Y[batch_indices],
                dtype=np.float64,
                copy=True
            )

            prediction = model(xb).cpu().numpy().astype(np.float64)

            persistence = np.array(
                X[batch_indices, -1, :],
                dtype=np.float64,
                copy=True
            )

            add_errors(lstm_total, prediction, yb)
            add_errors(persistence_total, persistence, yb)

    lstm_metrics = finish_metrics(lstm_total)
    persistence_metrics = finish_metrics(persistence_total)

    row = {
        "Label": label,
        "Samples": len(indices)
    }

    for metric in ["Huber", "MAE", "RMSE"]:

        lstm_value = lstm_metrics[metric]
        baseline_value = persistence_metrics[metric]

        improvement = (
            (baseline_value - lstm_value)
            / baseline_value * 100
            if baseline_value != 0 else 0.0
        )

        row[f"LSTM {metric}"] = lstm_value
        row[f"Persistence {metric}"] = baseline_value
        row[f"{metric} Improvement %"] = improvement

    results.append(row)


# Display results
result_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("TEST PERFORMANCE BY LABEL")
print("=" * 70)

print(result_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


# Save report
report = [
    "SENTINELX V2 — TEST METRICS BY LABEL",
    "=" * 70,
    f"Total test samples: {len(X)}",
    f"Checkpoint epoch: {checkpoint['epoch']}",
    f"Evaluation device: {device}",
    "",
    result_df.to_string(index=False)
]

REPORT_PATH.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("\nReport saved:", REPORT_PATH)
print("Checkpoint and dataset were not modified.")

