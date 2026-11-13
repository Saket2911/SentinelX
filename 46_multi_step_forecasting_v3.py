from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# 1. Configuration
# ============================================================

ROOT = Path(".")
DATA_DIR = ROOT / "Datasets"
MODEL_DIR = ROOT / "Models" / "lstm_world_model_v3_gpu_run2"

CHECKPOINT = MODEL_DIR / "best_model.pt"

X_PATH = DATA_DIR / "sentinelx_test_X_seq10_v3.npy"
Y_PATH = DATA_DIR / "sentinelx_test_y_seq10_v3.npy"
METADATA_PATH = DATA_DIR / "sentinelx_test_metadata_seq10_v3.csv"

OUTPUT_DIR = MODEL_DIR / "multi_step_forecasting_v3_fixed"
OUTPUT_CSV = OUTPUT_DIR / "multi_step_metrics.csv"
OUTPUT_REPORT = OUTPUT_DIR / "multi_step_forecasting_report.txt"

HORIZONS = [1, 2, 5, 10, 30]
SEQUENCE_LENGTH = 10
INPUT_FEATURES = 209
HIDDEN_SIZE = 128
BATCH_SIZE = 128
CPU_THREADS = 4

if not CHECKPOINT.exists():
    raise FileNotFoundError(CHECKPOINT)

for path in [X_PATH, Y_PATH, METADATA_PATH]:
    if not path.exists():
        raise FileNotFoundError(path)

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Output directory already exists: {OUTPUT_DIR}"
    )

OUTPUT_DIR.mkdir(parents=True)

torch.set_num_threads(CPU_THREADS)
DEVICE = torch.device("cpu")

print("=" * 72)
print("SENTINELX V3 — RECURSIVE MULTI-STEP FORECASTING")
print("=" * 72)
print("Device:", DEVICE)
print("Horizons:", HORIZONS)


# ============================================================
# 2. Load V3 test data and metadata
# ============================================================

X = np.load(X_PATH, mmap_mode="r")
y = np.load(Y_PATH, mmap_mode="r")
metadata = pd.read_csv(METADATA_PATH)

if X.ndim != 3 or X.shape[1:] != (
    SEQUENCE_LENGTH, INPUT_FEATURES
):
    raise ValueError(f"Unexpected X shape: {X.shape}")

if y.shape != (len(X), INPUT_FEATURES):
    raise ValueError(f"Unexpected y shape: {y.shape}")

if len(metadata) != len(X):
    raise ValueError("Metadata and sequence counts do not match.")

required_columns = [
    "Target Timestamp",
    "Target Label",
    "Segment ID"
]

for column in required_columns:
    if column not in metadata.columns:
        raise ValueError(f"Missing metadata column: {column}")

timestamps = pd.to_datetime(
    metadata["Target Timestamp"],
    errors="coerce"
)

if timestamps.isna().any():
    raise ValueError("Invalid target timestamps.")

timestamp_ns = timestamps.to_numpy(dtype="datetime64[ns]").astype("int64")
segments = metadata["Segment ID"].to_numpy()
labels = metadata["Target Label"].astype(str).to_numpy()

n = len(X)
max_horizon = max(HORIZONS)

print("\nTest X:", X.shape)
print("Test y:", y.shape)
print("Metadata:", metadata.shape)
print("Labels:", dict(zip(*np.unique(labels, return_counts=True))))
print("Segments:", dict(zip(*np.unique(segments, return_counts=True))))


# ============================================================
# 3. Validate horizon continuity
# ============================================================

# An edge is invalid if adjacent target states:
# - are not exactly one second apart, or
# - belong to different segments.

bad_edges = (
    (np.diff(timestamp_ns) != 1_000_000_000)
    | (segments[1:] != segments[:-1])
)

bad_edge_prefix = np.concatenate([
    np.array([0], dtype=np.int64),
    np.cumsum(bad_edges, dtype=np.int64)
])

valid_origins = {}

print("\nValid forecast origins:")

for horizon in HORIZONS:
    possible = np.arange(0, n - horizon + 1)

    invalid_counts = (
        bad_edge_prefix[possible + horizon - 1]
        - bad_edge_prefix[possible]
    )

    valid_origins[horizon] = invalid_counts == 0

    common_valid = valid_origins[horizon][:n - max_horizon + 1]
    print(
        f"Horizon {horizon:2d}s: "
        f"{int(common_valid.sum()):,} / "
        f"{total_origins:,} common origins"
    )

total_origins = n - max_horizon + 1

if total_origins <= 0:
    raise ValueError("Not enough test sequences for forecasting.")


# ============================================================
# 4. Reconstruct the exact V3 LSTM
# ============================================================

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
        return self.output_layer(self.dropout(final_state))


model = SentinelXLSTM().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

if checkpoint.get("input_features") != INPUT_FEATURES:
    raise ValueError("Checkpoint feature count is not 209.")

if checkpoint.get("sequence_length") != SEQUENCE_LENGTH:
    raise ValueError("Checkpoint sequence length is not 10.")

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("\nCheckpoint epoch:", checkpoint["epoch"])
print("Model loaded successfully.")


# ============================================================
# 5. Metric accumulation
# ============================================================

# Each key stores summed losses and number of scalar elements.
stores = {}

def add_metrics(key, prediction, actual):
    difference = prediction - actual
    absolute = np.abs(difference)

    huber = np.where(
        absolute <= 1.0,
        0.5 * difference ** 2,
        absolute - 0.5
    )

    if key not in stores:
        stores[key] = {
            "huber": 0.0,
            "absolute": 0.0,
            "squared": 0.0,
            "elements": 0,
            "states": 0
        }

    store = stores[key]
    store["huber"] += huber.sum(dtype=np.float64)
    store["absolute"] += absolute.sum(dtype=np.float64)
    store["squared"] += np.square(
        difference
    ).sum(dtype=np.float64)
    store["elements"] += difference.size
    store["states"] += len(prediction)


def calculate_metrics(store):
    count = store["elements"]

    return {
        "Huber": store["huber"] / count,
        "MAE": store["absolute"] / count,
        "RMSE": np.sqrt(store["squared"] / count),
        "States": store["states"]
    }


# ============================================================
# 6. Recursive forecasting
# ============================================================

print("\nStarting recursive forecasting...")
start_time = time.time()

with torch.no_grad():

    for batch_start in range(0, total_origins, BATCH_SIZE):

        batch_end = min(
            batch_start + BATCH_SIZE,
            total_origins
        )

        starts = np.arange(batch_start, batch_end)

        # Copy this batch from the memory-mapped array.
        batch_np = np.array(
            X[starts],
            dtype=np.float32,
            copy=True
        )

        current = torch.from_numpy(batch_np).to(DEVICE)

        # Persistence repeats the last actually observed state.
        persistence_prediction = batch_np[:, -1, :].copy()

        # Generate predictions recursively.
        for horizon in range(1, max_horizon + 1):

            prediction_tensor = model(current)
            prediction = prediction_tensor.cpu().numpy()

            if horizon in HORIZONS:

                valid_mask = valid_origins[horizon][starts]

                if valid_mask.any():

                    selected_starts = starts[valid_mask]
                    target_indices = selected_starts + horizon - 1

                    actual = np.array(
                        y[target_indices],
                        dtype=np.float32,
                        copy=True
                    )

                    selected_prediction = prediction[valid_mask]
                    selected_persistence = (
                        persistence_prediction[valid_mask]
                    )

                    future_labels = labels[target_indices]

                    # Overall results.
                    add_metrics(
                        (horizon, "ALL", "LSTM"),
                        selected_prediction,
                        actual
                    )

                    add_metrics(
                        (horizon, "ALL", "Persistence"),
                        selected_persistence,
                        actual
                    )

                    # Results grouped by the actual future label.
                    for label in np.unique(future_labels):
                        mask = future_labels == label

                        add_metrics(
                            (horizon, str(label), "LSTM"),
                            selected_prediction[mask],
                            actual[mask]
                        )

                        add_metrics(
                            (horizon, str(label), "Persistence"),
                            selected_persistence[mask],
                            actual[mask]
                        )

            # Feed the model prediction into the next recursive step.
            current = torch.cat(
                [
                    current[:, 1:, :],
                    prediction_tensor.unsqueeze(1)
                ],
                dim=1
            )

        if (
            batch_start == 0
            or batch_end == total_origins
            or batch_end % 2048 < BATCH_SIZE
        ):
            print(
                f"Processed {batch_end:,}/{total_origins:,} origins"
            )


# ============================================================
# 7. Calculate and save results
# ============================================================

rows = []

for horizon in HORIZONS:

    labels_for_horizon = ["ALL"] + sorted(
        label
        for h, label, method in stores
        if h == horizon and label != "ALL" and method == "LSTM"
    )

    for label in labels_for_horizon:

        lstm_key = (horizon, label, "LSTM")
        persistence_key = (horizon, label, "Persistence")

        if lstm_key not in stores:
            continue

        lstm = calculate_metrics(stores[lstm_key])
        persistence = calculate_metrics(stores[persistence_key])

        rows.append({
            "Horizon_Seconds": horizon,
            "Target_Label": label,
            "Valid_Origins": lstm["States"],
            "LSTM_Huber": lstm["Huber"],
            "Persistence_Huber": persistence["Huber"],
            "LSTM_MAE": lstm["MAE"],
            "Persistence_MAE": persistence["MAE"],
            "MAE_Improvement_Percent": (
                (persistence["MAE"] - lstm["MAE"])
                / persistence["MAE"] * 100
                if persistence["MAE"] else 0.0
            ),
            "LSTM_RMSE": lstm["RMSE"],
            "Persistence_RMSE": persistence["RMSE"],
            "RMSE_Improvement_Percent": (
                (persistence["RMSE"] - lstm["RMSE"])
                / persistence["RMSE"] * 100
                if persistence["RMSE"] else 0.0
            )
        })

results = pd.DataFrame(rows)
results.to_csv(OUTPUT_CSV, index=False)

elapsed = time.time() - start_time

report = [
    "SENTINELX V3 — RECURSIVE MULTI-STEP FORECASTING",
    "=" * 72,
    "",
    f"Checkpoint: {CHECKPOINT}",
    f"Checkpoint epoch: {checkpoint['epoch']}",
    f"Device: {DEVICE}",
    f"CPU threads: {CPU_THREADS}",
    f"Training performed: No",
    "",
    f"Test sequences: {n}",
    f"Sequence length: {SEQUENCE_LENGTH}",
    f"Input/output features: {INPUT_FEATURES}",
    f"Forecast horizons: {HORIZONS}",
    f"Total forecast origins: {total_origins}",
    f"Runtime seconds: {elapsed:.2f}",
    "",
    "RESULTS",
    "-" * 72,
    results.to_string(index=False),
    "",
    "METHODOLOGY",
    "-" * 72,
    "The trained V3 LSTM predicts the next network state.",
    "Each predicted state is recursively fed into the next input.",
    "The persistence baseline repeats the last observed input state.",
    "A common cohort of forecast origins is used across horizons.",
"A common cohort of forecast origins is used across horizons.",
"Only same-segment, exactly one-second-spaced target sequences",
    "are included for each forecast horizon.",
    "Results are reported overall and by actual future target label.",
    "",
    "LIMITATIONS",
    "-" * 72,
    "All metrics are in preprocessed model feature space.",
    "Recursive errors may accumulate at longer horizons.",
    "This evaluates numerical future-state forecasting, not attack detection.",
    "The test partition was used in earlier evaluation and analysis;",
    "these results are not an untouched independent test.",
    "",
    f"CSV: {OUTPUT_CSV}",
    f"Runtime: {elapsed:.2f} seconds"
]

OUTPUT_REPORT.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("\n" + "=" * 72)
print("V3 MULTI-STEP FORECASTING COMPLETED")
print("=" * 72)
print(results.to_string(index=False))
print("\nCSV:", OUTPUT_CSV)
print("Report:", OUTPUT_REPORT)
print(f"Runtime: {elapsed:.2f} seconds")
