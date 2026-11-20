from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

ROOT = Path(r"F:\Projects\SentinelX")
DATA_DIR = ROOT / "Datasets"
MODEL_DIR = ROOT / "Models" / "lstm_world_model_v1"

CHECKPOINT = MODEL_DIR / "best_model.pt"

OUTPUT_DIR = MODEL_DIR / "multi_step_forecasting_v1"
OUTPUT_CSV = OUTPUT_DIR / "multi_step_metrics.csv"
OUTPUT_REPORT = OUTPUT_DIR / "multi_step_forecasting_report.txt"

BATCH_SIZE = 64
HORIZONS = [1, 2, 5, 10, 30]

INPUT_FEATURES = 206
SEQUENCE_LENGTH = 10
HIDDEN_SIZE = 128

if not CHECKPOINT.exists():
    raise FileNotFoundError(CHECKPOINT)

if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    raise FileExistsError(
        f"Output directory is not empty: {OUTPUT_DIR}. "
        "Not overwriting existing results."
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("SENTINELX - RECURSIVE MULTI-STEP FORECASTING")
print("=" * 70)

print("Device:", DEVICE)
print("Forecast horizons:", HORIZONS)


# ---------------------------------------------------------
# 2. Load test arrays
# ---------------------------------------------------------

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

if X.shape[0] != n:
    raise ValueError("X and y lengths do not match.")

if len(timestamps) != n or len(segment_ids) != n:
    raise ValueError("Timestamp/segment metadata lengths do not match.")

if X.ndim != 3 or X.shape[1] != SEQUENCE_LENGTH:
    raise ValueError(f"Unexpected X shape: {X.shape}")

if X.shape[2] != INPUT_FEATURES or y.shape[1] != INPUT_FEATURES:
    raise ValueError("Unexpected feature count.")

print("\nTest X:", X.shape)
print("Test y:", y.shape)
print("Timestamps:", timestamps.shape)
print("Segment IDs:", segment_ids.shape)


# ---------------------------------------------------------
# 3. Validate chronological continuity
# ---------------------------------------------------------

parsed_timestamps = pd.to_datetime(
    timestamps,
    errors="coerce"
)

if parsed_timestamps.isna().any():
    raise ValueError("Invalid timestamps detected.")

timestamp_ns = parsed_timestamps.asi8

segment_ids = np.asarray(segment_ids).astype(str)

# An edge is invalid if:
# 1. The next timestamp is not exactly one second later.
# 2. The segment ID changes.

timestamp_gaps = np.diff(timestamp_ns) != 1_000_000_000
segment_changes = segment_ids[1:] != segment_ids[:-1]

bad_edges = timestamp_gaps | segment_changes

# Prefix sums allow efficient validation of every horizon.
bad_edge_prefix = np.concatenate([
    np.array([0], dtype=np.int64),
    np.cumsum(bad_edges, dtype=np.int64)
])

valid_origins = {}

for horizon in HORIZONS:

    possible_origins = np.arange(
        0,
        n - horizon + 1
    )

    invalid_edge_counts = (
        bad_edge_prefix[possible_origins + horizon - 1]
        - bad_edge_prefix[possible_origins]
    )

    valid_origins[horizon] = (
        invalid_edge_counts == 0
    )

    print(
        f"Horizon {horizon:2d}: "
        f"{int(valid_origins[horizon].sum())} valid origins "
        f"out of {len(possible_origins)} possible"
    )


# ---------------------------------------------------------
# 4. Reconstruct exact trained LSTM architecture
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

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nLoaded checkpoint epoch:", checkpoint["epoch"])


# ---------------------------------------------------------
# 5. Metric accumulation
# ---------------------------------------------------------

def metric_sums(prediction, actual):

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
        "RMSE": np.sqrt(squared_sum / count),
        "Elements": int(count)
    }


model_sums = {
    h: np.zeros(4, dtype=np.float64)
    for h in HORIZONS
}

baseline_sums = {
    h: np.zeros(4, dtype=np.float64)
    for h in HORIZONS
}

valid_counts = {
    h: 0
    for h in HORIZONS
}


# ---------------------------------------------------------
# 6. Recursive multi-step forecasting
# ---------------------------------------------------------

print("\nStarting recursive forecasting...")

max_horizon = max(HORIZONS)
total_origins = n - max_horizon + 1

if total_origins <= 0:
    raise ValueError("Not enough test sequences for requested horizons.")

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

        # Original historical input sequences.
        xb_np = np.array(
            X[starts],
            dtype=np.float32,
            copy=True
        )

        # Persistence baseline: last observed state.
        last_observed = xb_np[:, -1, :].copy()

        current_sequence = torch.from_numpy(
            xb_np
        ).to(DEVICE)

        # Generate future states recursively.
        for horizon in range(1, max_horizon + 1):

            prediction_tensor = model(current_sequence)

            prediction = prediction_tensor.cpu().numpy()

            # Evaluate only requested horizons.
            if horizon in HORIZONS:

                valid_mask = valid_origins[horizon][starts]

                if valid_mask.any():

                    selected_starts = starts[valid_mask]

                    actual_indices = (
                        selected_starts + horizon - 1
                    )

                    actual = np.array(
                        y[actual_indices],
                        dtype=np.float32,
                        copy=True
                    )

                    selected_prediction = prediction[valid_mask]

                    selected_baseline = last_observed[valid_mask]

                    model_sums[horizon] += metric_sums(
                        selected_prediction,
                        actual
                    )

                    baseline_sums[horizon] += metric_sums(
                        selected_baseline,
                        actual
                    )

                    valid_counts[horizon] += int(valid_mask.sum())

            # Feed the model's own prediction into its next input.
            next_state = prediction_tensor.unsqueeze(1)

            current_sequence = torch.cat(
                [
                    current_sequence[:, 1:, :],
                    next_state
                ],
                dim=1
            )

        if batch_end % 512 < BATCH_SIZE or batch_end == total_origins:
            print(
                f"Processed {batch_end}/{total_origins} "
                "forecast origins"
            )


# ---------------------------------------------------------
# 7. Calculate results
# ---------------------------------------------------------

results = []

print("\n" + "=" * 70)
print("MULTI-STEP FORECASTING RESULTS")
print("=" * 70)

for horizon in HORIZONS:

    if valid_counts[horizon] == 0:
        raise ValueError(
            f"No valid origins for horizon {horizon}."
        )

    model_metrics = calculate_metrics(
        model_sums[horizon]
    )

    baseline_metrics = calculate_metrics(
        baseline_sums[horizon]
    )

    mae_improvement = (
        (
            baseline_metrics["MAE"]
            - model_metrics["MAE"]
        )
        / baseline_metrics["MAE"]
    ) * 100

    rmse_improvement = (
        (
            baseline_metrics["RMSE"]
            - model_metrics["RMSE"]
        )
        / baseline_metrics["RMSE"]
    ) * 100

    print(f"\nFORECAST HORIZON: {horizon} second(s)")
    print("Valid forecast origins:", valid_counts[horizon])

    print(
        f"LSTM MAE: {model_metrics['MAE']:.8f} | "
        f"Persistence MAE: {baseline_metrics['MAE']:.8f}"
    )

    print(
        f"LSTM RMSE: {model_metrics['RMSE']:.8f} | "
        f"Persistence RMSE: {baseline_metrics['RMSE']:.8f}"
    )

    print(
        f"MAE improvement: {mae_improvement:.2f}%"
    )

    print(
        f"RMSE improvement: {rmse_improvement:.2f}%"
    )

    results.append({
        "Horizon_Seconds": horizon,
        "Valid_Origins": valid_counts[horizon],

        "LSTM_Huber": model_metrics["Huber"],
        "LSTM_MAE": model_metrics["MAE"],
        "LSTM_RMSE": model_metrics["RMSE"],

        "Persistence_Huber": baseline_metrics["Huber"],
        "Persistence_MAE": baseline_metrics["MAE"],
        "Persistence_RMSE": baseline_metrics["RMSE"],

        "MAE_Improvement_Percent": mae_improvement,
        "RMSE_Improvement_Percent": rmse_improvement
    })


# ---------------------------------------------------------
# 8. Save CSV and report
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)

report_lines = [
    "SENTINELX - RECURSIVE MULTI-STEP FORECASTING",
    "=" * 70,
    "",
    f"Checkpoint: {CHECKPOINT}",
    f"Checkpoint epoch: {checkpoint['epoch']}",
    f"Device: {DEVICE}",
    "",
    f"Test sequences: {n}",
    f"Input sequence length: {SEQUENCE_LENGTH}",
    f"Input features: {INPUT_FEATURES}",
    "",
    "FORECAST RESULTS",
    "-" * 70,
    results_df.to_string(index=False),
    "",
    "METHODOLOGY",
    "-" * 70,
    "The model predicts one future network state at a time.",
    "Each predicted state is fed back into the input sequence",
    "to generate the next prediction.",
    "",
    "The persistence baseline repeats the last observed state",
    "for every forecast horizon.",
    "",
    "Only forecasts whose target timestamps are consecutive",
    "one-second intervals and remain in the same segment are used.",
    "",
    "METRIC INTERPRETATION",
    "-" * 70,
    "MAE and RMSE measure numerical forecasting error.",
    "Positive improvement means the LSTM has lower error",
    "than the persistence baseline.",
    "",
    "IMPORTANT LIMITATIONS",
    "-" * 70,
    "Metrics are calculated in preprocessed feature space.",
    "Recursive prediction errors can accumulate over time.",
    "This experiment evaluates numerical future-state forecasting.",
    "It does not establish attack detection or early-warning accuracy.",
    "",
    "The test partition has been used in previous exploratory",
    "experiments, so these results are not an untouched evaluation.",
    "",
    f"CSV output: {OUTPUT_CSV}"
]

report_text = "\n".join(report_lines)

OUTPUT_REPORT.write_text(
    report_text,
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETED")
print("=" * 70)

print("CSV:", OUTPUT_CSV)
print("Report:", OUTPUT_REPORT)

print("\nOriginal checkpoint and test arrays were not modified.")
print("No training or fine-tuning was performed.")


