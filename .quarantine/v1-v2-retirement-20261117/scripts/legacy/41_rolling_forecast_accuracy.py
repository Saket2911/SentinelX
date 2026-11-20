from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =========================================================
# 1. CONFIGURATION
# =========================================================

ROOT = Path(r"F:\Projects\SentinelX")
DATA_DIR = ROOT / "Datasets"
MODEL_DIR = ROOT / "Models" / "lstm_world_model_v1"

CHECKPOINT = MODEL_DIR / "best_model.pt"

OUTPUT_DIR = MODEL_DIR / "rolling_forecast_analysis_v1"
OUTPUT_CSV = OUTPUT_DIR / "rolling_forecast_metrics.csv"
OUTPUT_REPORT = OUTPUT_DIR / "rolling_forecast_report.txt"

HORIZONS = [1, 2, 5, 10, 30]
SEQUENCE_LENGTH = 10
INPUT_FEATURES = 206
HIDDEN_SIZE = 128
BATCH_SIZE = 64
ROLLING_INTERVAL = "5min"

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
print("SENTINELX - ROLLING FORECAST ACCURACY ANALYSIS")
print("=" * 70)

print("Device:", DEVICE)
print("Horizons:", HORIZONS)
print("Rolling interval:", ROLLING_INTERVAL)


# =========================================================
# 2. LOAD TEST DATA
# =========================================================

X = np.load(
    DATA_DIR / "sentinelx_test_X.npy",
    mmap_mode="r"
)

y = np.load(
    DATA_DIR / "sentinelx_test_y.npy",
    mmap_mode="r"
)

timestamps = np.load(
    DATA_DIR / "sentinelx_test_target_timestamps.npy",
    mmap_mode="r"
)

segment_ids = np.load(
    DATA_DIR / "sentinelx_test_target_segment_ids.npy",
    mmap_mode="r"
)

n = len(y)

if not (
    len(X) == len(y)
    == len(timestamps)
    == len(segment_ids)
):
    raise ValueError("Test array lengths do not match.")

if X.shape[1:] != (SEQUENCE_LENGTH, INPUT_FEATURES):
    raise ValueError(f"Unexpected X shape: {X.shape}")

if y.shape[1] != INPUT_FEATURES:
    raise ValueError(f"Unexpected y shape: {y.shape}")

timestamps = pd.to_datetime(
    timestamps,
    errors="coerce"
)

if timestamps.isna().any():
    raise ValueError("Invalid timestamps detected.")

segment_ids = np.asarray(segment_ids).astype(str)

print("\nTest X:", X.shape)
print("Test y:", y.shape)
print("Test timestamp range:")
print(timestamps.min(), "to", timestamps.max())


# =========================================================
# 3. CHECK TEMPORAL CONTINUITY
# =========================================================

timestamp_ns = timestamps.asi8

bad_edges = (
    (np.diff(timestamp_ns) != 1_000_000_000)
    |
    (segment_ids[1:] != segment_ids[:-1])
)

bad_edge_prefix = np.concatenate([
    np.array([0], dtype=np.int64),
    np.cumsum(bad_edges, dtype=np.int64)
])

max_horizon = max(HORIZONS)

total_origins = n - max_horizon + 1

if total_origins <= 0:
    raise ValueError("Insufficient test sequences.")

starts_all = np.arange(total_origins)

valid_origins = {}

for horizon in HORIZONS:

    invalid_edges = (
        bad_edge_prefix[
            starts_all + horizon - 1
        ]
        -
        bad_edge_prefix[starts_all]
    )

    valid_origins[horizon] = invalid_edges == 0

    print(
        f"Horizon {horizon}: "
        f"{valid_origins[horizon].sum()} valid origins"
    )


# =========================================================
# 4. RECONSTRUCT THE LSTM
# =========================================================

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

        output, _ = self.lstm(x)

        final_state = output[:, -1, :]

        final_state = self.dropout(final_state)

        return self.output_layer(final_state)


model = SentinelXLSTM().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nLoaded checkpoint epoch:", checkpoint["epoch"])


# =========================================================
# 5. METRIC CALCULATION
# =========================================================

def calculate_sums(prediction, actual):

    difference = prediction - actual

    absolute = np.abs(difference)

    huber = np.where(
        absolute <= 1.0,
        0.5 * difference ** 2,
        absolute - 0.5
    )

    return np.array([
        huber.sum(),
        absolute.sum(),
        (difference ** 2).sum(),
        difference.size
    ], dtype=np.float64)


def calculate_metrics(values):

    huber_sum, absolute_sum, squared_sum, count = values

    return {
        "Huber": huber_sum / count,
        "MAE": absolute_sum / count,
        "RMSE": np.sqrt(squared_sum / count)
    }


# Each key represents one horizon and one five-minute interval.
results = defaultdict(
    lambda: {
        "lstm": np.zeros(4, dtype=np.float64),
        "persistence": np.zeros(4, dtype=np.float64),
        "sequences": 0
    }
)


# =========================================================
# 6. RECURSIVE FORECASTING AND ROLLING AGGREGATION
# =========================================================

print("\nStarting rolling forecasting analysis...")

with torch.no_grad():

    for batch_start in range(
        0,
        total_origins,
        BATCH_SIZE
    ):

        batch_end = min(
            batch_start + BATCH_SIZE,
            total_origins
        )

        starts = np.arange(batch_start, batch_end)

        xb = np.array(
            X[starts],
            dtype=np.float32,
            copy=True
        )

        last_observed = xb[:, -1, :].copy()

        current_sequence = torch.from_numpy(
            xb
        ).to(DEVICE)

        for horizon in range(1, max_horizon + 1):

            prediction_tensor = model(current_sequence)

            prediction = prediction_tensor.cpu().numpy()

            if horizon in HORIZONS:

                valid_mask = valid_origins[horizon][starts]

                if valid_mask.any():

                    selected_starts = starts[valid_mask]

                    target_indices = (
                        selected_starts + horizon - 1
                    )

                    actual = np.array(
                        y[target_indices],
                        dtype=np.float32,
                        copy=True
                    )

                    selected_prediction = prediction[valid_mask]

                    selected_baseline = last_observed[valid_mask]

                    target_times = timestamps[
                        target_indices
                    ]

                    time_bins = target_times.floor(
                        ROLLING_INTERVAL
                    )

                    for time_bin in time_bins.unique():

                        bin_mask = (
                            time_bins == time_bin
                        )

                        key = (
                            horizon,
                            str(time_bin)
                        )

                        entry = results[key]

                        entry["lstm"] += calculate_sums(
                            selected_prediction[bin_mask],
                            actual[bin_mask]
                        )

                        entry["persistence"] += calculate_sums(
                            selected_baseline[bin_mask],
                            actual[bin_mask]
                        )

                        entry["sequences"] += int(
                            bin_mask.sum()
                        )

            # Feed the model's prediction into its next input.
            next_state = prediction_tensor.unsqueeze(1)

            current_sequence = torch.cat(
                [
                    current_sequence[:, 1:, :],
                    next_state
                ],
                dim=1
            )

        if (
            batch_end % 512 < BATCH_SIZE
            or batch_end == total_origins
        ):
            print(
                f"Processed {batch_end}/{total_origins} origins"
            )


# =========================================================
# 7. BUILD ROLLING METRICS TABLE
# =========================================================

rows = []

for (horizon, time_bin), values in sorted(results.items()):

    lstm_metrics = calculate_metrics(
        values["lstm"]
    )

    baseline_metrics = calculate_metrics(
        values["persistence"]
    )

    mae_improvement = (
        (
            baseline_metrics["MAE"]
            - lstm_metrics["MAE"]
        )
        / baseline_metrics["MAE"]
    ) * 100

    rmse_improvement = (
        (
            baseline_metrics["RMSE"]
            - lstm_metrics["RMSE"]
        )
        / baseline_metrics["RMSE"]
    ) * 100

    rows.append({

        "Horizon_Seconds": horizon,
        "Time_Window": time_bin,
        "Sequences": values["sequences"],

        "LSTM_Huber": lstm_metrics["Huber"],
        "LSTM_MAE": lstm_metrics["MAE"],
        "LSTM_RMSE": lstm_metrics["RMSE"],

        "Persistence_Huber": baseline_metrics["Huber"],
        "Persistence_MAE": baseline_metrics["MAE"],
        "Persistence_RMSE": baseline_metrics["RMSE"],

        "MAE_Improvement_Percent": mae_improvement,
        "RMSE_Improvement_Percent": rmse_improvement
    })


results_df = pd.DataFrame(rows)

if results_df.empty:
    raise RuntimeError("No rolling metrics were generated.")

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# =========================================================
# 8. GENERATE REPORT
# =========================================================

report_lines = [

    "SENTINELX - ROLLING FORECAST ACCURACY ANALYSIS",
    "=" * 70,

    "",
    f"Checkpoint epoch: {checkpoint['epoch']}",
    f"Device: {DEVICE}",
    f"Forecast horizons: {HORIZONS}",
    f"Rolling interval: {ROLLING_INTERVAL}",

    "",
    f"Test sequences: {n}",
    f"Common forecast origins: {total_origins}",
    f"Test time range: {timestamps.min()} to {timestamps.max()}",

    "",
    "ROLLING METRICS",
    "-" * 70,

    results_df.to_string(index=False),

    "",
    "METHODOLOGY",
    "-" * 70,

    "The LSTM recursively predicts future network states.",
    "Predictions are fed back into the next input sequence.",

    "Persistence repeats the last observed network state.",

    "Forecasts are grouped by the actual target timestamp",
    "into consecutive five-minute time intervals.",

    "Only forecasts with continuous one-second timestamps",
    "inside the same temporal segment are evaluated.",

    "",
    "INTERPRETATION LIMITATIONS",
    "-" * 70,

    "Metrics measure numerical forecasting accuracy",
    "in preprocessed feature space.",

    "Lower MAE and RMSE indicate more accurate forecasts.",

    "Positive improvement means LSTM outperformed persistence.",
    "Negative improvement means persistence performed better.",

    "This experiment does not establish attack detection",
    "or early-warning capability.",

    "The test partition covers one date and one temporal segment.",
    "The test data has also been used in earlier experiments.",

    "",
    f"CSV: {OUTPUT_CSV}"
]

OUTPUT_REPORT.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)


# =========================================================
# 9. DISPLAY SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("ROLLING FORECAST ANALYSIS SUMMARY")
print("=" * 70)

for horizon in HORIZONS:

    subset = results_df[
        results_df["Horizon_Seconds"] == horizon
    ]

    print(f"\nHORIZON: {horizon} seconds")
    print("Five-minute intervals:", len(subset))

    print(
        "Average LSTM MAE:",
        round(subset["LSTM_MAE"].mean(), 6)
    )

    print(
        "Average Persistence MAE:",
        round(subset["Persistence_MAE"].mean(), 6)
    )

    print(
        "Average MAE improvement:",
        round(
            subset["MAE_Improvement_Percent"].mean(),
            2
        ),
        "%"
    )

    print(
        "Intervals where LSTM wins:",
        int(
            (
                subset["MAE_Improvement_Percent"] > 0
            ).sum()
        ),
        "/",
        len(subset)
    )


print("\n" + "=" * 70)
print("EXPERIMENT COMPLETED")
print("=" * 70)

print("CSV:", OUTPUT_CSV)
print("Report:", OUTPUT_REPORT)

print("\nOriginal checkpoint and test arrays were not modified.")
print("No training or fine-tuning was performed.")
