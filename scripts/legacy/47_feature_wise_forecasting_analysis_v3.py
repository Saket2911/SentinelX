from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(".")
DATA_DIR = ROOT / "Datasets"
MODEL_DIR = ROOT / "Models" / "lstm_world_model_v3_gpu_run2"

CHECKPOINT_PATH = MODEL_DIR / "best_model.pt"

X_PATH = DATA_DIR / "sentinelx_test_X_seq10_v3.npy"
Y_PATH = DATA_DIR / "sentinelx_test_y_seq10_v3.npy"
METADATA_PATH = DATA_DIR / "sentinelx_test_metadata_seq10_v3.csv"

REFERENCE_METRICS_PATH = (
    MODEL_DIR
    / "multi_step_forecasting_v3_fixed"
    / "multi_step_metrics.csv"
)

OUTPUT_DIR = (
    MODEL_DIR
    / "multi_step_feature_analysis_v1"
)

OUTPUT_FEATURE_CSV = OUTPUT_DIR / "feature_wise_metrics.csv"
OUTPUT_AGGREGATE_CSV = OUTPUT_DIR / "aggregate_verification.csv"
OUTPUT_REPORT = OUTPUT_DIR / "feature_analysis_report.txt"

HORIZONS = [1, 2, 5, 10, 30]

SEQUENCE_LENGTH = 10
INPUT_FEATURES = 209
HIDDEN_SIZE = 128

BATCH_SIZE = 128
CPU_THREADS = 4

DEVICE = torch.device("cpu")


# ============================================================
# 2. CHECK INPUT FILES
# ============================================================

required_files = [
    CHECKPOINT_PATH,
    X_PATH,
    Y_PATH,
    METADATA_PATH,
    REFERENCE_METRICS_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(path)

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Output directory already exists: {OUTPUT_DIR}"
    )

OUTPUT_DIR.mkdir(parents=True)

torch.set_num_threads(CPU_THREADS)

print("=" * 75)
print("SENTINELX V3 — FEATURE-WISE RECURSIVE FORECASTING ANALYSIS")
print("=" * 75)

print("Device:", DEVICE)
print("Horizons:", HORIZONS)


# ============================================================
# 3. LOAD CHECKPOINT AND AUTHORITATIVE FEATURE ORDER
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

feature_names = checkpoint["feature_names"]

if len(feature_names) != INPUT_FEATURES:
    raise ValueError(
        f"Expected {INPUT_FEATURES} checkpoint features, "
        f"found {len(feature_names)}."
    )

if len(set(feature_names)) != INPUT_FEATURES:
    raise ValueError("Duplicate feature names in checkpoint.")

if checkpoint["input_features"] != INPUT_FEATURES:
    raise ValueError("Checkpoint input feature count mismatch.")

if checkpoint["sequence_length"] != SEQUENCE_LENGTH:
    raise ValueError("Checkpoint sequence length mismatch.")

if checkpoint["hidden_size"] != HIDDEN_SIZE:
    raise ValueError("Checkpoint hidden size mismatch.")

print("\nCheckpoint epoch:", checkpoint["epoch"])
print("Feature count:", len(feature_names))
print("Feature order source: checkpoint['feature_names']")


# ============================================================
# 4. LOAD TEST SEQUENCES AND METADATA
# ============================================================

X = np.load(X_PATH, mmap_mode="r")
y = np.load(Y_PATH, mmap_mode="r")

metadata = pd.read_csv(METADATA_PATH)

if X.shape[1:] != (SEQUENCE_LENGTH, INPUT_FEATURES):
    raise ValueError(f"Unexpected X shape: {X.shape}")

if y.shape != (len(X), INPUT_FEATURES):
    raise ValueError(f"Unexpected y shape: {y.shape}")

if len(metadata) != len(X):
    raise ValueError("Metadata and sequence counts do not match.")

required_columns = [
    "Target Timestamp",
    "Target Label",
    "Segment ID",
]

for column in required_columns:
    if column not in metadata.columns:
        raise ValueError(f"Missing metadata column: {column}")

timestamps = pd.to_datetime(
    metadata["Target Timestamp"],
    errors="coerce"
)

if timestamps.isna().any():
    raise ValueError("Invalid timestamps found.")

timestamp_ns = (
    timestamps
    .to_numpy(dtype="datetime64[ns]")
    .astype("int64")
)

segments = metadata["Segment ID"].to_numpy()
labels = metadata["Target Label"].astype(str).to_numpy()

n = len(X)
max_horizon = max(HORIZONS)

total_origins = n - max_horizon + 1

if total_origins <= 0:
    raise ValueError("Insufficient test sequences.")

print("\nTest X:", X.shape)
print("Test y:", y.shape)
print("Metadata:", metadata.shape)
print("Common forecast origins:", total_origins)

print(
    "Labels:",
    dict(zip(*np.unique(labels, return_counts=True)))
)


# ============================================================
# 5. VALIDATE TEMPORAL CONTINUITY
# ============================================================

# An edge is invalid if adjacent target states:
# - are not exactly one second apart, OR
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

    possible = np.arange(total_origins)

    invalid_counts = (
        bad_edge_prefix[possible + horizon - 1]
        - bad_edge_prefix[possible]
    )

    valid_origins[horizon] = invalid_counts == 0

    print(
        f"Horizon {horizon:2d}s: "
        f"{int(valid_origins[horizon].sum()):,} / "
        f"{total_origins:,}"
    )


# ============================================================
# 6. RECONSTRUCT EXACT V3 MODEL
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

        return self.output_layer(
            self.dropout(final_state)
        )


model = SentinelXLSTM().to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nModel loaded successfully.")


# ============================================================
# 7. FEATURE-WISE METRIC ACCUMULATION
# ============================================================

stores = {}


def add_feature_metrics(key, prediction, actual):

    difference = prediction - actual

    absolute = np.abs(difference)
    squared = np.square(difference)

    huber = np.where(
        absolute <= 1.0,
        0.5 * squared,
        absolute - 0.5
    )

    if key not in stores:

        stores[key] = {
            "absolute": np.zeros(INPUT_FEATURES, dtype=np.float64),
            "squared": np.zeros(INPUT_FEATURES, dtype=np.float64),
            "huber": np.zeros(INPUT_FEATURES, dtype=np.float64),
            "states": 0,
        }

    store = stores[key]

    store["absolute"] += absolute.sum(
        axis=0,
        dtype=np.float64
    )

    store["squared"] += squared.sum(
        axis=0,
        dtype=np.float64
    )

    store["huber"] += huber.sum(
        axis=0,
        dtype=np.float64
    )

    store["states"] += len(prediction)


def aggregate_metrics(store):

    total_elements = store["states"] * INPUT_FEATURES

    return {
        "Huber": store["huber"].sum() / total_elements,
        "MAE": store["absolute"].sum() / total_elements,
        "RMSE": np.sqrt(
            store["squared"].sum() / total_elements
        ),
        "States": store["states"],
    }


# ============================================================
# 8. RECURSIVE MULTI-STEP FORECASTING
# ============================================================

print("\nStarting recursive feature-wise forecasting...")

start_time = time.time()

with torch.no_grad():

    for batch_start in range(0, total_origins, BATCH_SIZE):

        batch_end = min(
            batch_start + BATCH_SIZE,
            total_origins
        )

        starts = np.arange(batch_start, batch_end)

        batch_np = np.array(
            X[starts],
            dtype=np.float32,
            copy=True
        )

        current = torch.from_numpy(batch_np).to(DEVICE)

        # Persistence repeats the last observed state.
        persistence_prediction = batch_np[:, -1, :].copy()

        for horizon in range(1, max_horizon + 1):

            prediction_tensor = model(current)

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

                    selected_persistence = (
                        persistence_prediction[valid_mask]
                    )

                    future_labels = labels[target_indices]

                    # Overall group.
                    add_feature_metrics(
                        (horizon, "ALL", "LSTM"),
                        selected_prediction,
                        actual
                    )

                    add_feature_metrics(
                        (horizon, "ALL", "Persistence"),
                        selected_persistence,
                        actual
                    )

                    # Actual future-label groups.
                    for label in np.unique(future_labels):

                        label_mask = future_labels == label

                        add_feature_metrics(
                            (horizon, str(label), "LSTM"),
                            selected_prediction[label_mask],
                            actual[label_mask]
                        )

                        add_feature_metrics(
                            (horizon, str(label), "Persistence"),
                            selected_persistence[label_mask],
                            actual[label_mask]
                        )

            # Recursive feedback: predicted state becomes next input.
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


elapsed = time.time() - start_time

print(f"\nForecasting completed in {elapsed:.2f} seconds.")


# ============================================================
# 9. BUILD FEATURE-WISE RESULTS
# ============================================================

feature_rows = []

for horizon in HORIZONS:

    available_labels = sorted(
        label
        for h, label, method in stores
        if h == horizon
        and label != "ALL"
        and method == "LSTM"
    )

    for label in ["ALL"] + available_labels:

        lstm_key = (horizon, label, "LSTM")
        persistence_key = (horizon, label, "Persistence")

        if lstm_key not in stores:
            continue

        lstm = stores[lstm_key]
        persistence = stores[persistence_key]

        states = lstm["states"]

        for index, feature in enumerate(feature_names):

            lstm_mae = lstm["absolute"][index] / states

            lstm_rmse = np.sqrt(
                lstm["squared"][index] / states
            )

            persistence_mae = (
                persistence["absolute"][index] / states
            )

            persistence_rmse = np.sqrt(
                persistence["squared"][index] / states
            )

            lstm_huber = lstm["huber"][index] / states

            persistence_huber = (
                persistence["huber"][index] / states
            )

            mae_improvement = (
                (persistence_mae - lstm_mae)
                / persistence_mae * 100
                if persistence_mae > 0 else np.nan
            )

            rmse_improvement = (
                (persistence_rmse - lstm_rmse)
                / persistence_rmse * 100
                if persistence_rmse > 0 else np.nan
            )

            feature_rows.append({

                "Horizon_Seconds": horizon,
                "Target_Label": label,
                "Valid_Origins": states,

                "Feature_Index": index,
                "Feature": feature,

                "LSTM_Huber": lstm_huber,
                "Persistence_Huber": persistence_huber,

                "LSTM_MAE": lstm_mae,
                "Persistence_MAE": persistence_mae,
                "MAE_Improvement_Percent": mae_improvement,

                "LSTM_RMSE": lstm_rmse,
                "Persistence_RMSE": persistence_rmse,
                "RMSE_Improvement_Percent": rmse_improvement,
            })


feature_results = pd.DataFrame(feature_rows)

feature_results.to_csv(
    OUTPUT_FEATURE_CSV,
    index=False
)

print("\nSaved feature-wise metrics:")
print(OUTPUT_FEATURE_CSV)

print("Feature-wise rows:", len(feature_results))


# ============================================================
# 10. VERIFY AGGREGATE RESULTS AGAINST ORIGINAL EXPERIMENT
# ============================================================

reference = pd.read_csv(REFERENCE_METRICS_PATH)

verification_rows = []

for horizon in HORIZONS:

    labels_for_horizon = sorted(
        label
        for h, label, method in stores
        if h == horizon
        and label != "ALL"
        and method == "LSTM"
    )

    for label in ["ALL"] + labels_for_horizon:

        lstm = aggregate_metrics(
            stores[(horizon, label, "LSTM")]
        )

        persistence = aggregate_metrics(
            stores[(horizon, label, "Persistence")]
        )

        matching = reference[
            (reference["Horizon_Seconds"] == horizon)
            & (reference["Target_Label"].astype(str) == label)
        ]

        if len(matching) != 1:
            raise ValueError(
                f"Expected one reference row for "
                f"{horizon}s / {label}; found {len(matching)}."
            )

        ref = matching.iloc[0]

        comparisons = {
            "LSTM_Huber": (
                lstm["Huber"],
                ref["LSTM_Huber"]
            ),
            "Persistence_Huber": (
                persistence["Huber"],
                ref["Persistence_Huber"]
            ),
            "LSTM_MAE": (
                lstm["MAE"],
                ref["LSTM_MAE"]
            ),
            "Persistence_MAE": (
                persistence["MAE"],
                ref["Persistence_MAE"]
            ),
            "LSTM_RMSE": (
                lstm["RMSE"],
                ref["LSTM_RMSE"]
            ),
            "Persistence_RMSE": (
                persistence["RMSE"],
                ref["Persistence_RMSE"]
            ),
        }

        checks = {}

        for metric, (new_value, old_value) in comparisons.items():

            checks[metric] = bool(
                np.isclose(
                    new_value,
                    old_value,
                    rtol=2e-5,
                    atol=2e-6
                )
            )

        count_match = (
            lstm["States"] == int(ref["Valid_Origins"])
        )

        passed = all(checks.values()) and count_match

        verification_rows.append({

            "Horizon_Seconds": horizon,
            "Target_Label": label,

            "New_Valid_Origins": lstm["States"],
            "Reference_Valid_Origins": int(ref["Valid_Origins"]),

            "LSTM_MAE_New": lstm["MAE"],
            "LSTM_MAE_Reference": ref["LSTM_MAE"],

            "LSTM_RMSE_New": lstm["RMSE"],
            "LSTM_RMSE_Reference": ref["LSTM_RMSE"],

            "Persistence_MAE_New": persistence["MAE"],
            "Persistence_MAE_Reference": ref["Persistence_MAE"],

            "Persistence_RMSE_New": persistence["RMSE"],
            "Persistence_RMSE_Reference": ref["Persistence_RMSE"],

            "All_Metrics_Match": all(checks.values()),
            "Count_Matches": count_match,
            "Verification": "PASS" if passed else "FAIL",
        })


verification = pd.DataFrame(verification_rows)

verification.to_csv(
    OUTPUT_AGGREGATE_CSV,
    index=False
)

print("\n" + "=" * 75)
print("AGGREGATE VERIFICATION")
print("=" * 75)

print(
    verification[
        [
            "Horizon_Seconds",
            "Target_Label",
            "Verification"
        ]
    ].to_string(index=False)
)

all_passed = (
    verification["Verification"] == "PASS"
).all()

print("\nAll aggregate checks passed:", all_passed)


# ============================================================
# 11. FEATURE-WISE SUMMARY
# ============================================================

overall = feature_results[
    feature_results["Target_Label"] == "ALL"
].copy()

summary_lines = [

    "SENTINELX V3 — FEATURE-WISE RECURSIVE FORECASTING",
    "=" * 75,

    f"Checkpoint epoch: {checkpoint['epoch']}",
    f"Features: {INPUT_FEATURES}",
    f"Sequence length: {SEQUENCE_LENGTH}",
    f"Common forecast origins: {total_origins}",
    f"Device: {DEVICE}",
    f"Forecasting seconds: {elapsed:.2f}",

    "",
    "AGGREGATE VERIFICATION",
    "-" * 75,
    f"All checks passed: {all_passed}",

    "",
    "FEATURE-WISE RESULTS",
    "-" * 75,
]

for horizon in HORIZONS:

    subset = overall[
        overall["Horizon_Seconds"] == horizon
    ].copy()

    better = (
        subset["MAE_Improvement_Percent"] > 0
    ).sum()

    worse = (
        subset["MAE_Improvement_Percent"] < 0
    ).sum()

    equal = (
        subset["MAE_Improvement_Percent"] == 0
    ).sum()

    best = subset.sort_values(
        "MAE_Improvement_Percent",
        ascending=False
    ).head(5)

    worst = subset.sort_values(
        "MAE_Improvement_Percent",
        ascending=True
    ).head(5)

    summary_lines.extend([

        "",
        f"HORIZON: {horizon} seconds",
        f"Features improved over persistence: {better}/{INPUT_FEATURES}",
        f"Features worse than persistence: {worse}/{INPUT_FEATURES}",
        f"Features equal to persistence: {equal}/{INPUT_FEATURES}",

        "",
        "TOP 5 FEATURES BY MAE IMPROVEMENT:",
    ])

    for _, row in best.iterrows():

        summary_lines.append(
            f"  {row['Feature']}: "
            f"{row['MAE_Improvement_Percent']:.3f}%"
        )

    summary_lines.append(
        "\nBOTTOM 5 FEATURES BY MAE IMPROVEMENT:"
    )

    for _, row in worst.iterrows():

        summary_lines.append(
            f"  {row['Feature']}: "
            f"{row['MAE_Improvement_Percent']:.3f}%"
        )


summary_lines.extend([

    "",
    "=" * 75,
    "INTERPRETATION NOTES",
    "-" * 75,

    "Positive improvement means LSTM MAE is lower than persistence MAE.",
    "Negative improvement means persistence MAE is lower than LSTM MAE.",
    "Feature-wise errors are measured in preprocessed model feature space.",
    "They are not raw network units.",
    "Recursive predictions are fed into subsequent forecast windows.",
    "This evaluates numerical future-state prediction, not attack detection.",
    "Test data was previously used for exploratory evaluation.",
    "These results are not an untouched independent test evaluation.",

    "",
    f"Aggregate verification: {'PASS' if all_passed else 'FAIL'}",
])

OUTPUT_REPORT.write_text(
    "\n".join(summary_lines),
    encoding="utf-8"
)


# ============================================================
# 12. FINAL STATUS
# ============================================================

print("\n" + "=" * 75)
print("ANALYSIS COMPLETE")
print("=" * 75)

print("Feature-wise CSV:", OUTPUT_FEATURE_CSV)
print("Verification CSV:", OUTPUT_AGGREGATE_CSV)
print("Report:", OUTPUT_REPORT)

print(
    "Overall feature rows:",
    len(overall)
)

if not all_passed:

    print("\nWARNING: Aggregate verification failed.")
    print("Do not interpret the feature-wise results yet.")

    raise RuntimeError(
        "New analysis did not reproduce the reference metrics."
    )

print("\nSUCCESS: Aggregate results reproduced.")
print("Feature-wise results are ready for inspection.")
