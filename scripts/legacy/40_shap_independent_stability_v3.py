import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn


ROOT = Path("/mnt/f/Projects/SentinelX")
DATA = ROOT / "Datasets"
MODEL_DIR = ROOT / "Models/lstm_world_model_v3_gpu_run2"
OUT = MODEL_DIR / "shap_explainability_v3"
OUT.mkdir(parents=True, exist_ok=True)

CHECKPOINT = MODEL_DIR / "best_model.pt"

BACKGROUND_SIZE = 128
SAMPLES_PER_LABEL = 32
NSAMPLES = 100
SEED = 123

DEVICE = torch.device("cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(2)


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT, map_location=DEVICE, weights_only=False
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
        self.output_layer = nn.Linear(HIDDEN_SIZE, INPUT_FEATURES)

    def forward(self, x):
        sequence_output, _ = self.lstm(x)
        last_state = sequence_output[:, -1, :]
        return self.output_layer(self.dropout(last_state))


class SelectedOutputModel(nn.Module):
    def __init__(self, base_model, indices):
        super().__init__()
        self.base_model = base_model
        self.indices = indices

    def forward(self, x):
        return self.base_model(x)[:, self.indices]


model = SentinelXLSTM().to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# ------------------------------------------------------------
# LOAD ARRAYS AND METADATA
# ------------------------------------------------------------

X_train = np.load(
    DATA / "sentinelx_train_X_seq10_v3.npy",
    mmap_mode="r"
)
X_test = np.load(
    DATA / "sentinelx_test_X_seq10_v3.npy",
    mmap_mode="r"
)

train_meta = pd.read_csv(
    DATA / "sentinelx_train_metadata_seq10_v3.csv"
)
test_meta = pd.read_csv(
    DATA / "sentinelx_test_metadata_seq10_v3.csv"
)

assert len(X_train) == len(train_meta)
assert len(X_test) == len(test_meta)
assert X_train.shape[1:] == (SEQUENCE_LENGTH, INPUT_FEATURES)
assert X_test.shape[1:] == (SEQUENCE_LENGTH, INPUT_FEATURES)


# ------------------------------------------------------------
# SELECT BENIGN TRAINING BACKGROUND
# ------------------------------------------------------------

rng = np.random.default_rng(SEED)

benign_train_indices = np.flatnonzero(
    train_meta["Target Label"].to_numpy() == "Benign"
)

if len(benign_train_indices) < BACKGROUND_SIZE:
    raise RuntimeError("Not enough benign training sequences.")

background_indices = rng.choice(
    benign_train_indices,
    size=BACKGROUND_SIZE,
    replace=False
)

background = np.array(
    X_train[background_indices],
    dtype=np.float32,
    copy=True
)


# ------------------------------------------------------------
# SELECT HELD-OUT BENIGN AND BOT EXPLANATION SAMPLES
# ------------------------------------------------------------

test_labels = test_meta["Target Label"].to_numpy()

benign_test_indices = np.flatnonzero(test_labels == "Benign")
bot_test_indices = np.flatnonzero(test_labels == "Bot")

if min(len(benign_test_indices), len(bot_test_indices)) < SAMPLES_PER_LABEL:
    raise RuntimeError("Insufficient held-out samples for comparison.")

selected_benign = rng.choice(
    benign_test_indices,
    size=SAMPLES_PER_LABEL,
    replace=False
)

selected_bot = rng.choice(
    bot_test_indices,
    size=SAMPLES_PER_LABEL,
    replace=False
)

explain_indices = np.concatenate([selected_benign, selected_bot])
explain_labels = np.array(
    ["Benign"] * SAMPLES_PER_LABEL +
    ["Bot"] * SAMPLES_PER_LABEL
)

explain_X = np.array(
    X_test[explain_indices],
    dtype=np.float32,
    copy=True
)

if not np.isfinite(background).all():
    raise RuntimeError("Non-finite background values.")
if not np.isfinite(explain_X).all():
    raise RuntimeError("Non-finite explanation values.")

print("Checkpoint epoch:", checkpoint["epoch"])
print("Training benign background:", len(background))
print("Held-out benign explanations:", len(selected_benign))
print("Held-out Bot explanations:", len(selected_bot))
print("Input shape:", explain_X.shape)


# ------------------------------------------------------------
# SELECT FIVE OUTPUT FEATURES
# ------------------------------------------------------------

requested_outputs = [
    "Flow Count",
    "Flow IAT Mean__mean",
    "Flow IAT Min__mean",
    "TotLen Fwd Pkts__mean",
    "ACK Flag Cnt__mean",
]

output_indices = [
    FEATURE_NAMES.index(name)
    for name in requested_outputs
]

explain_model = SelectedOutputModel(
    model, output_indices
).to(DEVICE)
explain_model.eval()

background_tensor = torch.from_numpy(background)
explain_tensor = torch.from_numpy(explain_X)


# ------------------------------------------------------------
# RUN SHAP
# ------------------------------------------------------------

print("\nCreating GradientExplainer...")
explainer = shap.GradientExplainer(
    explain_model,
    background_tensor
)

print("Calculating SHAP values...")
shap_values = explainer.shap_values(
    explain_tensor,
    nsamples=NSAMPLES
)

if isinstance(shap_values, list):
    shap_array = np.stack(shap_values, axis=-1)
else:
    shap_array = np.asarray(shap_values)

expected_shape = (
    len(explain_indices),
    SEQUENCE_LENGTH,
    INPUT_FEATURES,
    len(requested_outputs)
)

print("SHAP shape:", shap_array.shape)

if shap_array.shape != expected_shape:
    raise RuntimeError(
        f"Unexpected SHAP shape {shap_array.shape}; "
        f"expected {expected_shape}"
    )

if not np.isfinite(shap_array).all():
    raise RuntimeError("SHAP contains non-finite values.")


# ------------------------------------------------------------
# FEATURE IMPORTANCE BY LABEL AND OUTPUT
# ------------------------------------------------------------

feature_rows = []
temporal_rows = []

for label in ["Benign", "Bot"]:
    sample_mask = explain_labels == label
    label_values = shap_array[sample_mask]

    for output_position, output_name in enumerate(requested_outputs):
        values = label_values[:, :, :, output_position]

        mean_abs = np.mean(np.abs(values), axis=(0, 1))
        mean_signed = np.mean(values, axis=(0, 1))

        for feature_index, feature_name in enumerate(FEATURE_NAMES):
            feature_rows.append({
                "Target Label": label,
                "Predicted Output Feature": output_name,
                "Input Feature": feature_name,
                "Mean Absolute SHAP": float(mean_abs[feature_index]),
                "Mean Signed SHAP": float(mean_signed[feature_index]),
            })

        for time_index in range(SEQUENCE_LENGTH):
            temporal_rows.append({
                "Target Label": label,
                "Predicted Output Feature": output_name,
                "Input Time Position": time_index,
                "Mean Absolute SHAP": float(
                    np.mean(np.abs(values[:, time_index, :]))
                )
            })


feature_df = pd.DataFrame(feature_rows)

feature_df["Feature Rank"] = (
    feature_df.groupby(
        ["Target Label", "Predicted Output Feature"]
    )["Mean Absolute SHAP"]
    .rank(method="first", ascending=False)
    .astype(int)
)

feature_df = feature_df.sort_values(
    ["Target Label", "Predicted Output Feature", "Feature Rank"]
)

temporal_df = pd.DataFrame(temporal_rows)

feature_df.to_csv(
    OUT / "shap_feature_importance_by_label.csv",
    index=False
)

temporal_df.to_csv(
    OUT / "shap_temporal_importance_by_label.csv",
    index=False
)


# ------------------------------------------------------------
# BENIGN VS BOT ATTRIBUTION DIFFERENCES
# ------------------------------------------------------------

comparison = feature_df.pivot_table(
    index=["Predicted Output Feature", "Input Feature"],
    columns="Target Label",
    values="Mean Absolute SHAP"
).reset_index()

comparison["Bot_minus_Benign_SHAP"] = (
    comparison["Bot"] - comparison["Benign"]
)

comparison["Absolute_Difference"] = (
    comparison["Bot_minus_Benign_SHAP"].abs()
)

comparison.to_csv(
    OUT / "shap_benign_bot_comparison.csv",
    index=False
)


# ------------------------------------------------------------
# SAVE RAW VALUES AND EXPERIMENT METADATA
# ------------------------------------------------------------

np.savez_compressed(
    OUT / "shap_values.npz",
    shap_values=shap_array,
    background_indices=background_indices,
    explain_indices=explain_indices,
    explain_labels=explain_labels,
    output_indices=np.array(output_indices),
    feature_names=np.array(FEATURE_NAMES),
    output_names=np.array(requested_outputs)
)

metadata = {
    "experiment": "SentinelX SHAP Benign Reference V2",
    "method": "SHAP GradientExplainer",
    "checkpoint_epoch": int(checkpoint["epoch"]),
    "sequence_length": int(SEQUENCE_LENGTH),
    "input_features": int(INPUT_FEATURES),
    "background_source": "Benign training sequences",
    "background_size": int(BACKGROUND_SIZE),
    "explanation_source": "Held-out test sequences",
    "benign_explanation_samples": int(SAMPLES_PER_LABEL),
    "bot_explanation_samples": int(SAMPLES_PER_LABEL),
    "nsamples": int(NSAMPLES),
    "output_features": requested_outputs,
    "device": str(DEVICE),
    "shap_version": shap.__version__,
    "shap_shape": list(shap_array.shape),
    "interpretation": (
        "Predictive feature attributions relative to benign "
        "training background. Not attack probabilities, "
        "causal evidence, or a standalone detector."
    )
}

with open(OUT / "summary.json", "w") as f:
    json.dump(metadata, f, indent=4)


# ------------------------------------------------------------
# DISPLAY TOP FEATURES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SHAP V2 — TOP FEATURES BY LABEL")
print("=" * 70)

for label in ["Benign", "Bot"]:
    for output_name in requested_outputs:
        subset = feature_df[
            (feature_df["Target Label"] == label) &
            (feature_df["Predicted Output Feature"] == output_name)
        ].head(10)

        print(f"\n{label} | Predicted output: {output_name}")
        print(subset[
            ["Feature Rank", "Input Feature", "Mean Absolute SHAP"]
        ].to_string(index=False))


print("\n" + "=" * 70)
print("SHAP V2 COMPLETED")
print("=" * 70)
print("Output directory:", OUT)
print("SHAP shape:", shap_array.shape)
print("Feature rankings:", OUT / "shap_feature_importance_by_label.csv")
print("Temporal rankings:", OUT / "shap_temporal_importance_by_label.csv")
print("Label comparison:", OUT / "shap_benign_bot_comparison.csv")
print("Raw values:", OUT / "shap_values.npz")
print("Metadata:", OUT / "summary.json")
