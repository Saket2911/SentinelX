from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(r"F:\Projects\SentinelX")
DATA = ROOT / "Datasets"
MODEL_DIR = ROOT / "Models" / "lstm_world_model_v1"

CHECKPOINT = MODEL_DIR / "best_model.pt"

OUTPUT_DIR = MODEL_DIR / "transition_forecast_analysis_v1"
OUTPUT_CSV = OUTPUT_DIR / "transition_forecast_metrics.csv"
OUTPUT_REPORT = OUTPUT_DIR / "transition_forecast_report.txt"

HORIZONS = [1, 2, 5, 10, 30]
SEQ_LEN = 10
FEATURES = 206
HIDDEN = 128
BATCH_SIZE = 64
TRANSITION_RADIUS = 5

if not CHECKPOINT.exists():
    raise FileNotFoundError(CHECKPOINT)

if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    raise FileExistsError(
        f"Output folder already contains files: {OUTPUT_DIR}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("SENTINELX - TRANSITION FORECAST ANALYSIS")
print("=" * 70)

print("Device:", DEVICE)
print("Horizons:", HORIZONS)


# ---------------------------------------------------------
# Load arrays
# ---------------------------------------------------------

X = np.load(DATA / "sentinelx_test_X.npy", mmap_mode="r")
y = np.load(DATA / "sentinelx_test_y.npy", mmap_mode="r")

labels = np.asarray(
    np.load(DATA / "sentinelx_test_target_labels.npy")
).astype(str)

timestamps = pd.to_datetime(
    np.load(DATA / "sentinelx_test_target_timestamps.npy"),
    errors="coerce"
)

segments = np.asarray(
    np.load(DATA / "sentinelx_test_target_segment_ids.npy")
).astype(str)

n = len(y)

if not (
    len(X) == len(y) == len(labels)
    == len(timestamps) == len(segments)
):
    raise ValueError("Array lengths do not match.")

if timestamps.isna().any():
    raise ValueError("Invalid timestamps detected.")

if X.shape[1:] != (SEQ_LEN, FEATURES):
    raise ValueError(f"Unexpected X shape: {X.shape}")


# ---------------------------------------------------------
# Identify label transitions
# ---------------------------------------------------------

transition = np.zeros(n, dtype=bool)

transition[1:] = (
    (labels[1:] != labels[:-1])
    |
    (segments[1:] != segments[:-1])
)

transition_indices = np.flatnonzero(transition)

near_transition = np.zeros(n, dtype=bool)

for index in transition_indices:
    start = max(0, index - TRANSITION_RADIUS)
    end = min(n, index + TRANSITION_RADIUS + 1)
    near_transition[start:end] = True

print("\nTest labels:", dict(zip(*np.unique(labels, return_counts=True))))
print("Label/segment transitions:", len(transition_indices))
print("States near transitions:", int(near_transition.sum()))


# ---------------------------------------------------------
# Model
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
        return self.output_layer(
            self.dropout(output[:, -1, :])
        )


model = SentinelXLSTM().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Loaded checkpoint epoch:", checkpoint["epoch"])


# ---------------------------------------------------------
# Continuity validation
# ---------------------------------------------------------

bad_edges = (
    (np.diff(timestamps.asi8) != 1_000_000_000)
    |
    (segments[1:] != segments[:-1])
)

prefix = np.concatenate([
    np.array([0], dtype=np.int64),
    np.cumsum(bad_edges, dtype=np.int64)
])

max_horizon = max(HORIZONS)
total_origins = n - max_horizon + 1

origins = np.arange(total_origins)

valid = {}

for h in HORIZONS:
    invalid = (
        prefix[origins + h - 1]
        - prefix[origins]
    )
    valid[h] = invalid == 0

print("Common origins:", total_origins)


# ---------------------------------------------------------
# Accumulators
# ---------------------------------------------------------

# Group by horizon, target label, and transition proximity.
acc = defaultdict(
    lambda: {
        "lstm_abs": 0.0,
        "lstm_sq": 0.0,
        "baseline_abs": 0.0,
        "baseline_sq": 0.0,
        "elements": 0,
        "sequences": 0
    }
)


def update(key, prediction, baseline, actual):

    item = acc[key]

    d1 = prediction - actual
    d2 = baseline - actual

    item["lstm_abs"] += np.abs(d1).sum()
    item["lstm_sq"] += np.square(d1).sum()

    item["baseline_abs"] += np.abs(d2).sum()
    item["baseline_sq"] += np.square(d2).sum()

    item["elements"] += actual.size
    item["sequences"] += len(actual)


# ---------------------------------------------------------
# Recursive forecasting
# ---------------------------------------------------------

print("\nStarting transition-aware forecasting...")

with torch.no_grad():

    for batch_start in range(0, total_origins, BATCH_SIZE):

        batch_end = min(
            batch_start + BATCH_SIZE,
            total_origins
        )

        starts = np.arange(batch_start, batch_end)

        batch_x = np.array(
            X[starts],
            dtype=np.float32,
            copy=True
        )

        baseline = batch_x[:, -1, :].copy()

        current = torch.from_numpy(batch_x).to(DEVICE)

        for h in range(1, max_horizon + 1):

            prediction_tensor = model(current)
            prediction = prediction_tensor.cpu().numpy()

            if h in HORIZONS:

                mask = valid[h][starts]

                if mask.any():

                    selected_starts = starts[mask]
                    target_indices = selected_starts + h - 1

                    actual = np.array(
                        y[target_indices],
                        dtype=np.float32,
                        copy=True
                    )

                    pred = prediction[mask]
                    base = baseline[mask]

                    target_labels = labels[target_indices]
                    target_transition = near_transition[target_indices]

                    for label in np.unique(target_labels):

                        label_mask = target_labels == label

                        for category, category_mask in [
                            ("Near_Transition", target_transition),
                            ("Away_From_Transition", ~target_transition)
                        ]:

                            selected = label_mask & category_mask

                            if not selected.any():
                                continue

                            key = (h, str(label), category)

                            update(
                                key,
                                pred[selected],
                                base[selected],
                                actual[selected]
                            )

            next_state = prediction_tensor.unsqueeze(1)

            current = torch.cat(
                [current[:, 1:, :], next_state],
                dim=1
            )

        if batch_end % 512 < BATCH_SIZE or batch_end == total_origins:
            print(f"Processed {batch_end}/{total_origins}")


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

rows = []

for (horizon, label, category), item in sorted(acc.items()):

    count = item["elements"]

    lstm_mae = item["lstm_abs"] / count
    lstm_rmse = np.sqrt(item["lstm_sq"] / count)

    base_mae = item["baseline_abs"] / count
    base_rmse = np.sqrt(item["baseline_sq"] / count)

    mae_improvement = (
        (base_mae - lstm_mae) / base_mae
    ) * 100

    rmse_improvement = (
        (base_rmse - lstm_rmse) / base_rmse
    ) * 100

    rows.append({
        "Horizon_Seconds": horizon,
        "Target_Label": label,
        "Transition_Category": category,
        "Sequences": item["sequences"],
        "LSTM_MAE": lstm_mae,
        "LSTM_RMSE": lstm_rmse,
        "Persistence_MAE": base_mae,
        "Persistence_RMSE": base_rmse,
        "MAE_Improvement_Percent": mae_improvement,
        "RMSE_Improvement_Percent": rmse_improvement
    })

df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError("No results generated.")

df.to_csv(OUTPUT_CSV, index=False)


# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

report = [
    "SENTINELX - TRANSITION FORECAST ANALYSIS",
    "=" * 70,
    "",
    f"Checkpoint epoch: {checkpoint['epoch']}",
    f"Device: {DEVICE}",
    f"Transition radius: {TRANSITION_RADIUS} states",
    f"Label/segment transitions: {len(transition_indices)}",
    f"Near-transition states: {int(near_transition.sum())}",
    "",
    "RESULTS",
    "-" * 70,
    df.to_string(index=False),
    "",
    "METHODOLOGY",
    "-" * 70,
    "A transition occurs when consecutive target labels change",
    "or when a temporal segment changes.",
    "",
    "Near-transition states are within five seconds before",
    "or after a detected transition.",
    "",
    "Predictions are grouped using the actual target label",
    "and the target state's transition proximity.",
    "",
    "The persistence baseline repeats the last observed state.",
    "",
    "LIMITATIONS",
    "-" * 70,
    "Label transitions are dataset annotations, not necessarily",
    "the exact onset of malicious activity.",
    "",
    "The test partition contains only the labels observed here.",
    "Forecasting error is not an attack probability.",
    "This experiment does not establish early-warning capability.",
    "",
    f"CSV: {OUTPUT_CSV}"
]

OUTPUT_REPORT.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("TRANSITION ANALYSIS COMPLETED")
print("=" * 70)

print(df.to_string(index=False))

print("\nCSV:", OUTPUT_CSV)
print("Report:", OUTPUT_REPORT)

print("\nCheckpoint and source arrays were not modified.")
print("No training or fine-tuning was performed.")
