from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

ROOT = Path(r"F:\Projects\SentinelX")
DATA_DIR = ROOT / "Datasets"
MODEL_DIR = ROOT / "Models" / "lstm_world_model_v1"

CHECKPOINT = MODEL_DIR / "best_model.pt"
REPORT = MODEL_DIR / "test_evaluation_report.txt"

BATCH_SIZE = 64
INPUT_FEATURES = 206
SEQUENCE_LENGTH = 10
HIDDEN_SIZE = 128

if not CHECKPOINT.exists():
    raise FileNotFoundError(CHECKPOINT)

if REPORT.exists():
    raise FileExistsError(
        f"Report already exists: {REPORT}. Not overwriting."
    )

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("SENTINELX - WORLD MODEL TEST EVALUATION")
print("=" * 65)
print("Device:", DEVICE)


# ---------------------------------------------------------
# 2. Load test arrays using memory mapping
# ---------------------------------------------------------

X = np.load(
    DATA_DIR / "sentinelx_test_X.npy",
    mmap_mode="r"
)

y = np.load(
    DATA_DIR / "sentinelx_test_y.npy",
    mmap_mode="r"
)

labels = np.load(
    DATA_DIR / "sentinelx_test_target_labels.npy",
    mmap_mode="r"
)

if len(X) != len(y) or len(y) != len(labels):
    raise ValueError("Test array lengths do not match.")

print("Test X:", X.shape)
print("Test y:", y.shape)
print("Test labels:", labels.shape)


# ---------------------------------------------------------
# 3. Reconstruct the exact trained architecture
# ---------------------------------------------------------

class SentinelXLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=INPUT_FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.2)

        self.output_layer = nn.Linear(
            HIDDEN_SIZE,
            INPUT_FEATURES
        )

    def forward(self, x):
        sequence_output, _ = self.lstm(x)
        final_state = sequence_output[:, -1, :]
        final_state = self.dropout(final_state)
        return self.output_layer(final_state)


model = SentinelXLSTM().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Loaded best checkpoint from epoch:", checkpoint["epoch"])


# ---------------------------------------------------------
# 4. Metric accumulation
# ---------------------------------------------------------

def metric_sums(prediction, actual):
    difference = prediction - actual
    absolute = np.abs(difference)

    huber = np.where(
        absolute <= 1.0,
        0.5 * difference ** 2,
        absolute - 0.5
    )

    return (
        float(huber.sum()),
        float(absolute.sum()),
        float((difference ** 2).sum()),
        int(difference.size)
    )


def add_metrics(store, key, prediction, actual):
    values = metric_sums(prediction, actual)

    for i, value in enumerate(values):
        store[key][i] += value


def format_metrics(values):
    huber, absolute, squared, count = values

    return {
        "huber": huber / count,
        "mae": absolute / count,
        "rmse": (squared / count) ** 0.5
    }


overall_model = np.zeros(4, dtype=np.float64)
overall_baseline = np.zeros(4, dtype=np.float64)

per_label_model = defaultdict(
    lambda: np.zeros(4, dtype=np.float64)
)

per_label_baseline = defaultdict(
    lambda: np.zeros(4, dtype=np.float64)
)


# ---------------------------------------------------------
# 5. Evaluate model and persistence baseline
# ---------------------------------------------------------

print("\nEvaluating test batches...")

with torch.no_grad():

    for start in range(0, len(X), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(X))

        xb_np = np.array(X[start:end], dtype=np.float32, copy=True)
        yb_np = np.array(y[start:end], dtype=np.float32, copy=True)

        xb = torch.from_numpy(xb_np).to(DEVICE)

        prediction = model(xb).cpu().numpy()

        # Baseline: predict the next state as the last input state.
        baseline_prediction = xb_np[:, -1, :]

        batch_labels = np.asarray(labels[start:end])

        for pred, base, actual, label in zip(
            prediction,
            baseline_prediction,
            yb_np,
            batch_labels
        ):
            add_metrics(
                {"result": overall_model},
                "result",
                pred,
                actual
            )

            add_metrics(
                {"result": overall_baseline},
                "result",
                base,
                actual
            )

            label = str(label)

            add_metrics(
                per_label_model,
                label,
                pred.reshape(1, -1),
                actual.reshape(1, -1)
            )

            add_metrics(
                per_label_baseline,
                label,
                base.reshape(1, -1),
                actual.reshape(1, -1)
            )

        if end % 2048 < BATCH_SIZE or end == len(X):
            print(f"Processed {end}/{len(X)} test sequences")


# ---------------------------------------------------------
# 6. Calculate final metrics
# ---------------------------------------------------------

model_metrics = format_metrics(overall_model)
baseline_metrics = format_metrics(overall_baseline)

print("\n" + "=" * 65)
print("OVERALL TEST RESULTS")
print("=" * 65)

print("\nLSTM WORLD MODEL")
for name, value in model_metrics.items():
    print(f"{name.upper()}: {value:.8f}")

print("\nPERSISTENCE BASELINE")
for name, value in baseline_metrics.items():
    print(f"{name.upper()}: {value:.8f}")


# Positive improvement means the LSTM has lower error.
huber_improvement = (
    (baseline_metrics["huber"] - model_metrics["huber"])
    / baseline_metrics["huber"] * 100
)

mae_improvement = (
    (baseline_metrics["mae"] - model_metrics["mae"])
    / baseline_metrics["mae"] * 100
)

rmse_improvement = (
    (baseline_metrics["rmse"] - model_metrics["rmse"])
    / baseline_metrics["rmse"] * 100
)

print("\nLSTM improvement over persistence:")
print(f"Huber loss: {huber_improvement:.2f}%")
print(f"MAE:        {mae_improvement:.2f}%")
print(f"RMSE:       {rmse_improvement:.2f}%")


# ---------------------------------------------------------
# 7. Per-label evaluation
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("PER-LABEL TEST RESULTS")
print("=" * 65)

label_report = []

for label in sorted(per_label_model):

    model_result = format_metrics(per_label_model[label])
    baseline_result = format_metrics(per_label_baseline[label])

    count = int(per_label_model[label][3] / INPUT_FEATURES)

    print(f"\nLabel: {label}")
    print("Sequences:", count)
    print(
        f"LSTM MAE: {model_result['mae']:.6f} | "
        f"Baseline MAE: {baseline_result['mae']:.6f}"
    )
    print(
        f"LSTM RMSE: {model_result['rmse']:.6f} | "
        f"Baseline RMSE: {baseline_result['rmse']:.6f}"
    )

    label_report.append(
        f"{label}: {count} sequences\n"
        f"  LSTM: {model_result}\n"
        f"  Persistence: {baseline_result}\n"
    )


# ---------------------------------------------------------
# 8. Save report
# ---------------------------------------------------------

report_text = f"""
SENTINELX - LSTM WORLD MODEL TEST EVALUATION

Checkpoint: {CHECKPOINT}
Checkpoint epoch: {checkpoint['epoch']}
Device: {DEVICE}

TEST DATA:
Sequences: {len(X)}
Sequence length: {SEQUENCE_LENGTH}
Features: {INPUT_FEATURES}

LSTM TEST METRICS:
{model_metrics}

PERSISTENCE BASELINE METRICS:
{baseline_metrics}

IMPROVEMENT OVER BASELINE:
Huber: {huber_improvement:.4f}%
MAE: {mae_improvement:.4f}%
RMSE: {rmse_improvement:.4f}%

PER-LABEL RESULTS:
{''.join(label_report)}

IMPORTANT:
Metrics are calculated in preprocessed feature space.
The persistence baseline predicts the next state as the
last observed input state.

This test evaluates next-state numerical prediction.
It does not establish attack detection accuracy.
"""

REPORT.write_text(report_text, encoding="utf-8")

print("\n" + "=" * 65)
print("TEST EVALUATION COMPLETED")
print("=" * 65)
print("Report saved:", REPORT)
