import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("/mnt/f/Projects/SentinelX")
DATA = ROOT / "Datasets"
MODEL_DIR = ROOT / "Models/lstm_world_model_v3_gpu_run2"
CHECKPOINT_PATH = MODEL_DIR / "best_model.pt"

OUTPUT_DIR = MODEL_DIR / "shap_explainability_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_X_PATH = DATA / "sentinelx_train_X_seq10_v3.npy"
TEST_X_PATH = DATA / "sentinelx_test_X_seq10_v3.npy"

BACKGROUND_SIZE = 32
EXPLAIN_SIZE = 8
NSAMPLES = 50
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_num_threads(2)

DEVICE = torch.device("cpu")


# ============================================================
# LOAD CHECKPOINT AND RECONSTRUCT EXACT MODEL
# ============================================================

print("=" * 70)
print("SENTINELX V3 — SHAP EXPLAINABILITY EXPERIMENT")
print("=" * 70)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

INPUT_FEATURES = checkpoint["input_features"]
HIDDEN_SIZE = checkpoint["hidden_size"]
NUM_LAYERS = checkpoint["num_layers"]
DROPOUT = checkpoint["dropout"]
SEQUENCE_LENGTH = checkpoint["sequence_length"]

FEATURE_NAMES = checkpoint["feature_names"]

class SentinelXLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=INPUT_FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True
        )

        self.dropout = nn.Dropout(DROPOUT)

        self.output_layer = nn.Linear(
            HIDDEN_SIZE,
            INPUT_FEATURES
        )

    def forward(self, x):
        sequence_output, _ = self.lstm(x)
        last_state = sequence_output[:, -1, :]
        last_state = self.dropout(last_state)
        return self.output_layer(last_state)


model = SentinelXLSTM().to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Checkpoint epoch:", checkpoint["epoch"])
print("Sequence length:", SEQUENCE_LENGTH)
print("Input features:", INPUT_FEATURES)
print("Hidden size:", HIDDEN_SIZE)
print("Model device:", DEVICE)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading sequence arrays...")

X_train = np.load(TRAIN_X_PATH, mmap_mode="r")
X_test = np.load(TEST_X_PATH, mmap_mode="r")

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

assert X_train.shape[1:] == (SEQUENCE_LENGTH, INPUT_FEATURES)
assert X_test.shape[1:] == (SEQUENCE_LENGTH, INPUT_FEATURES)
assert len(FEATURE_NAMES) == INPUT_FEATURES


# ============================================================
# SELECT BACKGROUND AND EXPLANATION SAMPLES
# ============================================================

rng = np.random.default_rng(SEED)

background_indices = rng.choice(
    len(X_train),
    size=BACKGROUND_SIZE,
    replace=False
)

explain_indices = rng.choice(
    len(X_test),
    size=EXPLAIN_SIZE,
    replace=False
)

background = np.array(
    X_train[background_indices],
    dtype=np.float32,
    copy=True
)

explain_X = np.array(
    X_test[explain_indices],
    dtype=np.float32,
    copy=True
)

assert np.isfinite(background).all()
assert np.isfinite(explain_X).all()

background_tensor = torch.from_numpy(background).to(DEVICE)
explain_tensor = torch.from_numpy(explain_X).to(DEVICE)

print("\nBackground samples:", len(background))
print("Explanation samples:", len(explain_X))


# ============================================================
# SELECT NEXT-STATE OUTPUT FEATURES
# ============================================================

requested_outputs = [
    "Flow Count",
    "Flow IAT Mean__mean",
    "Flow IAT Min__mean",
    "TotLen Fwd Pkts__mean",
    "ACK Flag Cnt__mean",
]

output_indices = []

for name in requested_outputs:
    if name not in FEATURE_NAMES:
        raise ValueError(f"Missing output feature: {name}")
    output_indices.append(FEATURE_NAMES.index(name))

print("\nSelected predicted output features:")
for name, index in zip(requested_outputs, output_indices):
    print(f"  {index}: {name}")


# ============================================================
# WRAPPER: EXPLAIN ONLY SELECTED MODEL OUTPUTS
# ============================================================

class SelectedOutputModel(nn.Module):
    def __init__(self, base_model, indices):
        super().__init__()
        self.base_model = base_model
        self.indices = indices

    def forward(self, x):
        predictions = self.base_model(x)
        return predictions[:, self.indices]


explain_model = SelectedOutputModel(
    model,
    output_indices
).to(DEVICE)

explain_model.eval()


# ============================================================
# SHAP GRADIENT EXPLAINER
# ============================================================

print("\nCreating SHAP GradientExplainer...")

explainer = shap.GradientExplainer(
    explain_model,
    background_tensor
)

print("Calculating SHAP values...")
print("This may take some time on CPU.")

shap_values = explainer.shap_values(
    explain_tensor,
    nsamples=NSAMPLES
)

# SHAP versions may return a list (one array per output)
# or a single array with output dimension last.
if isinstance(shap_values, list):
    shap_array = np.stack(shap_values, axis=-1)
else:
    shap_array = np.asarray(shap_values)

print("\nRaw SHAP output shape:", shap_array.shape)

expected_shape = (
    EXPLAIN_SIZE,
    SEQUENCE_LENGTH,
    INPUT_FEATURES,
    len(output_indices)
)

if shap_array.shape != expected_shape:
    raise RuntimeError(
        f"Unexpected SHAP shape: {shap_array.shape}; "
        f"expected {expected_shape}"
    )

if not np.isfinite(shap_array).all():
    raise RuntimeError("SHAP values contain non-finite values.")


# ============================================================
# AGGREGATE FEATURE IMPORTANCE
# ============================================================

# Average absolute contribution across samples and time steps.
# Each output feature gets its own input-feature ranking.

summary_rows = []

for output_position, output_name in enumerate(requested_outputs):
    values = shap_array[:, :, :, output_position]

    mean_abs = np.mean(np.abs(values), axis=(0, 1))
    mean_signed = np.mean(values, axis=(0, 1))

    for feature_index, feature_name in enumerate(FEATURE_NAMES):
        summary_rows.append({
            "Predicted Output Feature": output_name,
            "Input Feature": feature_name,
            "Mean Absolute SHAP": float(mean_abs[feature_index]),
            "Mean Signed SHAP": float(mean_signed[feature_index]),
        })

summary_df = pd.DataFrame(summary_rows)

summary_df["Feature Rank"] = (
    summary_df.groupby("Predicted Output Feature")[
        "Mean Absolute SHAP"
    ]
    .rank(method="first", ascending=False)
    .astype(int)
)

summary_df = summary_df.sort_values(
    ["Predicted Output Feature", "Feature Rank"]
)

summary_path = OUTPUT_DIR / "shap_feature_importance.csv"
summary_df.to_csv(summary_path, index=False)


# ============================================================
# TEMPORAL IMPORTANCE
# ============================================================

temporal_rows = []

for output_position, output_name in enumerate(requested_outputs):
    values = shap_array[:, :, :, output_position]

    for time_index in range(SEQUENCE_LENGTH):
        temporal_rows.append({
            "Predicted Output Feature": output_name,
            "Input Time Position": time_index,
            "Mean Absolute SHAP": float(
                np.mean(np.abs(values[:, time_index, :]))
            )
        })

temporal_df = pd.DataFrame(temporal_rows)

temporal_path = OUTPUT_DIR / "shap_temporal_importance.csv"
temporal_df.to_csv(temporal_path, index=False)


# ============================================================
# SAVE RAW SHAP VALUES AND METADATA
# ============================================================

np.savez_compressed(
    OUTPUT_DIR / "shap_values.npz",
    shap_values=shap_array,
    background_indices=background_indices,
    explain_indices=explain_indices,
    output_indices=np.array(output_indices),
    feature_names=np.array(FEATURE_NAMES),
    output_names=np.array(requested_outputs),
)

metadata = {
    "experiment": "SentinelX LSTM SHAP V1",
    "method": "SHAP GradientExplainer",
    "checkpoint": str(CHECKPOINT_PATH),
    "checkpoint_epoch": int(checkpoint["epoch"]),
    "sequence_length": int(SEQUENCE_LENGTH),
    "input_features": int(INPUT_FEATURES),
    "background_size": int(BACKGROUND_SIZE),
    "explanation_samples": int(EXPLAIN_SIZE),
    "nsamples": int(NSAMPLES),
    "output_features": requested_outputs,
    "background_source": "Random training sequences",
    "explanation_source": "Held-out test sequences",
    "device": str(DEVICE),
    "shap_version": shap.__version__,
    "interpretation": (
        "Feature contributions to selected next-state predictions. "
        "SHAP values are not attack probabilities or causal evidence."
    ),
    "shap_shape": list(shap_array.shape),
}

with open(OUTPUT_DIR / "shap_experiment_summary.json", "w") as f:
    json.dump(metadata, f, indent=4)


# ============================================================
# DISPLAY TOP RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TOP SHAP INPUT FEATURES")
print("=" * 70)

for output_name in requested_outputs:
    print(f"\nPredicted output: {output_name}")

    top = summary_df[
        summary_df["Predicted Output Feature"] == output_name
    ].head(10)

    print(
        top[
            [
                "Feature Rank",
                "Input Feature",
                "Mean Absolute SHAP",
                "Mean Signed SHAP",
            ]
        ].to_string(index=False)
    )

print("\n" + "=" * 70)
print("TEMPORAL IMPORTANCE")
print("=" * 70)

print(
    temporal_df.to_string(index=False)
)

print("\n" + "=" * 70)
print("SHAP EXPERIMENT COMPLETED")
print("=" * 70)
print("Output directory:", OUTPUT_DIR)
print("Feature importance:", summary_path)
print("Temporal importance:", temporal_path)
print("Raw SHAP values:", OUTPUT_DIR / "shap_values.npz")
print("Metadata:", OUTPUT_DIR / "shap_experiment_summary.json")
print("SHAP shape:", shap_array.shape)
