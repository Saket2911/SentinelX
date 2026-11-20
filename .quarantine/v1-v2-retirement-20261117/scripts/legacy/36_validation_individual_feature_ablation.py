from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(r"F:\Projects\SentinelX")
DATA = ROOT / "Datasets"
MODEL_DIR = ROOT / "Models" / "lstm_world_model_v1"

CHECKPOINT = MODEL_DIR / "best_model.pt"
X_PATH = DATA / "sentinelx_validation_X.npy"
Y_PATH = DATA / "sentinelx_validation_y.npy"

FEATURE_NAMES_PATH = DATA / "sentinelx_sequence_feature_names.txt"

OUTPUT = MODEL_DIR / "individual_feature_ablation_validation_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentinelXLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=206,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.2)
        self.output_layer = nn.Linear(128, 206)

    def forward(self, x):
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]
        last_output = self.dropout(last_output)
        return self.output_layer(last_output)


def base_feature(name):

    for suffix in ("__mean", "__std", "__max"):
        if name.endswith(suffix):
            return name[:-len(suffix)]

    return name


def classify_feature(name):

    name = base_feature(name).lower()

    if name.startswith((
        "flow duration", "flow iat", "fwd iat",
        "bwd iat", "active ", "idle "
    )):
        return "Timing Behaviour"

    if name.startswith((
        "tot fwd pkts", "tot bwd pkts",
        "subflow fwd pkts", "subflow bwd pkts",
        "fwd act data pkts", "flow count", "has flows"
    )):
        return "Packet Volume"

    if name.startswith((
        "totlen fwd", "totlen bwd",
        "subflow fwd byts", "subflow bwd byts",
        "fwd pkt len", "bwd pkt len",
        "pkt len", "pkt size avg",
        "fwd seg size avg", "bwd seg size avg",
        "fwd seg size min"
    )):
        return "Packet Size and Bytes"

    if name.startswith((
        "down/up ratio", "fwd header len",
        "bwd header len", "fwd pkts/s",
        "bwd pkts/s", "flow byts/s",
        "flow pkts/s"
    )):
        return "Directional Behaviour and Rates"

    return "Protocol and Control"


def calculate_metrics(predictions, targets, feature_names, groups):

    errors = predictions - targets

    mae = np.abs(errors).mean()
    rmse = np.sqrt(np.square(errors).mean())

    # Huber loss with delta=1, matching training.
    abs_errors = np.abs(errors)
    huber = np.where(
        abs_errors <= 1.0,
        0.5 * np.square(errors),
        abs_errors - 0.5
    ).mean()

    result = {
        "Overall_MAE": float(mae),
        "Overall_RMSE": float(rmse),
        "Overall_Huber": float(huber)
    }

    for group, indices in groups.items():

        group_errors = np.abs(errors[:, indices])

        result[f"{group}_MAE"] = float(
            group_errors.mean()
        )

    return result


def evaluate(model, X, y, feature_names, groups, masked_indices=None):

    predictions = []

    for start in range(0, len(X), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(X))

        # Copy this batch so the original memmap is never modified.
        batch = np.array(X[start:end], dtype=np.float32, copy=True)

        if masked_indices is not None:
            batch[:, :, masked_indices] = 0.0

        batch_tensor = torch.from_numpy(batch).to(DEVICE)

        with torch.no_grad():
            output = model(batch_tensor)

        predictions.append(
            output.cpu().numpy()
        )

    predictions = np.concatenate(predictions, axis=0)

    targets = np.asarray(y, dtype=np.float32)

    metrics = calculate_metrics(
        predictions, targets, feature_names, groups
    )

    return metrics


def main():

    print("=" * 70)
    print("SENTINELX - VALIDATION INDIVIDUAL FEATURE ABLATION")
    print("=" * 70)

    for path in [CHECKPOINT, X_PATH, Y_PATH, FEATURE_NAMES_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    print(f"\nDevice: {DEVICE}")

    # Load feature order.
    feature_names = [
        line.strip()
        for line in FEATURE_NAMES_PATH.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]

    if len(feature_names) != 206:
        raise ValueError(
            f"Expected 206 feature names, got {len(feature_names)}"
        )

    # Group mean, std and max representations by original base feature.
    groups = {}

    for index, name in enumerate(feature_names):
        group = base_feature(name)
        groups.setdefault(group, []).append(index)

    print(f"\nIndividual base features: {len(groups)}")

    for group, indices in groups.items():
        print(f"{group}: {len(indices)} model features")

    if sum(len(v) for v in groups.values()) != 206:
        raise ValueError("Not all model features were assigned")

    # Load validation sequences and targets.
    X = np.load(X_PATH, mmap_mode="r")
    y = np.load(Y_PATH, mmap_mode="r")

    if X.ndim != 3 or X.shape[1:] != (10, 206):
        raise ValueError(f"Unexpected X shape: {X.shape}")

    if y.shape != (len(X), 206):
        raise ValueError(f"Unexpected y shape: {y.shape}")

    print(f"\nValidation X shape: {X.shape}")
    print(f"Validation y shape: {y.shape}")

    # Load the original trained model.
    model = SentinelXLSTM().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint: {CHECKPOINT}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    results = []

    # Baseline: no features masked.
    print("\nEvaluating original baseline...")

    baseline = evaluate(
        model, X, y, feature_names, groups
    )

    results.append({
        "Experiment": "Baseline",
        "Masked_Group": "None",
        "Masked_Feature_Count": 0,
        **baseline
    })

    print(f"Baseline MAE: {baseline['Overall_MAE']:.6f}")
    print(f"Baseline RMSE: {baseline['Overall_RMSE']:.6f}")

    # Mask one group at a time in input sequences only.
    for group, indices in groups.items():

        print(f"\nMasking individual feature: {group}")

        metrics = evaluate(
            model,
            X,
            y,
            feature_names,
            groups,
            masked_indices=indices
        )

        mae_change = (
            metrics["Overall_MAE"] - baseline["Overall_MAE"]
        )

        percentage_change = (
            mae_change / baseline["Overall_MAE"]
        ) * 100

        results.append({
            "Experiment": "Ablation",
            "Masked_Group": group,
            "Masked_Feature_Count": len(indices),
            **metrics
        })

        print(f"MAE: {metrics['Overall_MAE']:.6f}")
        print(f"MAE change: {mae_change:+.6f}")
        print(f"MAE percentage change: {percentage_change:+.2f}%")

    result_df = pd.DataFrame(results)

    # Calculate changes relative to baseline.
    for metric in [
        "Overall_MAE",
        "Overall_RMSE",
        "Overall_Huber"
    ]:

        baseline_value = baseline[metric]

        result_df[f"{metric}_Change_Percent"] = (
            (result_df[metric] - baseline_value)
            / baseline_value
        ) * 100

    result_path = OUTPUT / "ablation_results.csv"

    result_df.to_csv(result_path, index=False)

    print("\n" + "=" * 70)
    print("FINAL ABLATION RESULTS")
    print("=" * 70)

    print(
        result_df[[
            "Experiment",
            "Masked_Group",
            "Masked_Feature_Count",
            "Overall_MAE",
            "Overall_MAE_Change_Percent",
            "Overall_RMSE",
            "Overall_Huber"
        ]].to_string(index=False)
    )

    print("\nResults saved to:")
    print(result_path)

    print("\nOriginal model checkpoint and validation arrays were not modified.")
    print("No training or fine-tuning was performed.")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()



