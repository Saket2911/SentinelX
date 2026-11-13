import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

ROOT = Path(".")
DATA = ROOT / "Datasets"
MODEL_DIR = ROOT / "Models/lstm_world_model_v3_gpu_run2"
OUTPUT = MODEL_DIR / "historical_replay_v1"

CHECKPOINT = MODEL_DIR / "best_model.pt"

class SentinelXLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=209,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.2)
        self.output_layer = nn.Linear(128, 209)

    def forward(self, x):
        output, _ = self.lstm(x)
        last_state = output[:, -1, :]
        return self.output_layer(self.dropout(last_state))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    print("Device:", device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    model = SentinelXLSTM().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    feature_names = checkpoint["feature_names"]
    if len(feature_names) != 209:
        raise ValueError("Checkpoint feature mapping is not 209 features.")

    X_path = DATA / "sentinelx_test_X_seq10_v3.npy"
    y_path = DATA / "sentinelx_test_y_seq10_v3.npy"
    metadata_path = DATA / "sentinelx_test_metadata_seq10_v3.csv"

    X = np.load(X_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    metadata = pd.read_csv(metadata_path)

    if X.shape != (len(y), 10, 209):
        raise ValueError(f"Unexpected X shape: {X.shape}")

    if y.shape != (len(metadata), 209):
        raise ValueError(f"Unexpected y shape: {y.shape}")

    print("Test sequences:", len(X))
    print("Input shape:", X.shape)
    print("Target shape:", y.shape)

    predictions = np.empty((len(X), 209), dtype=np.float32)

    batch_size = 256

    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))

            batch = torch.from_numpy(
                np.asarray(X[start:end], dtype=np.float32)
            ).to(device)

            predictions[start:end] = (
                model(batch).cpu().numpy()
            )

            if start % 5000 == 0:
                print(f"Predicted {end:,}/{len(X):,}")

    actual = np.asarray(y, dtype=np.float32)

    # Residual is calculated in the same transformed model space
    # used during training and evaluation.
    absolute_errors = np.abs(actual - predictions)
    sample_mae = absolute_errors.mean(axis=1)

    replay = metadata.copy()
    replay.insert(0, "Replay Step", np.arange(1, len(X) + 1))
    replay["Prediction MAE (model space)"] = sample_mae

    # Include a few interpretable feature comparisons when available.
    preferred_features = [
        "Flow Count",
        "Flow IAT Mean__mean",
        "Flow IAT Min__mean",
        "TotLen Fwd Pkts__mean",
        "ACK Flag Cnt__mean"
    ]

    selected = [
        name for name in preferred_features
        if name in feature_names
    ]

    print("\nSelected replay features:")
    for name in selected:
        print(" -", name)

    for name in selected:
        index = feature_names.index(name)
        replay[f"Actual | {name}"] = actual[:, index]
        replay[f"Predicted | {name}"] = predictions[:, index]
        replay[f"Absolute Error | {name}"] = absolute_errors[:, index]

    csv_path = OUTPUT / "test_replay.csv"
    replay.to_csv(csv_path, index=False)

    np.savez_compressed(
        OUTPUT / "replay_arrays.npz",
        predictions=predictions,
        actual=actual,
        feature_names=np.asarray(feature_names)
    )

    manifest = {
        "engine": "SentinelX Historical Replay V1",
        "source": "CIC-IDS-2018 held-out test sequences",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "sequence_length": 10,
        "prediction_horizon": "one next state",
        "input_features": 209,
        "number_of_steps": len(X),
        "prediction_method": "rolling one-step-ahead",
        "predictions_fed_back_as_inputs": False,
        "prediction_comparison_space": "preprocessed model feature space",
        "mean_test_mae": float(sample_mae.mean()),
        "selected_features": selected,
        "csv": str(csv_path),
        "arrays": str(OUTPUT / "replay_arrays.npz")
    }

    with open(OUTPUT / "replay_manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)

    print("\n" + "=" * 60)
    print("HISTORICAL REPLAY GENERATION COMPLETE")
    print("=" * 60)
    print("Replay steps:", len(replay))
    print("Mean prediction MAE:", round(float(sample_mae.mean()), 6))
    print("CSV:", csv_path)
    print("Arrays:", OUTPUT / "replay_arrays.npz")
    print("Manifest:", OUTPUT / "replay_manifest.json")


if __name__ == "__main__":
    main()

