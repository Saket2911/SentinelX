from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(r"F:\Projects\SentinelX")
DATA = ROOT / "Datasets"
CKPT_PATH = ROOT / "Models" / "lstm_world_model_v2" / "best_model.pt"
BATCH_SIZE = 32


class SentinelXLSTM(nn.Module):
    def __init__(self, input_features, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.output_layer = nn.Linear(hidden_size, input_features)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.output_layer(output[:, -1, :])


def calculate_metrics(pred, target):
    error = pred - target
    abs_error = np.abs(error)

    mae = abs_error.mean()
    rmse = np.sqrt(np.mean(error ** 2))

    huber = np.where(
        abs_error <= 1.0,
        0.5 * error ** 2,
        abs_error - 0.5
    ).mean()

    return mae, rmse, huber


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    model = SentinelXLSTM(
        ckpt["input_features"],
        ckpt["hidden_size"],
        ckpt["num_layers"],
        ckpt["dropout"]
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    for split in ["validation", "test"]:
        X = np.load(DATA / f"sentinelx_{split}_X_seq10_v1.npy", mmap_mode="r")
        Y = np.load(DATA / f"sentinelx_{split}_y_next_v1.npy", mmap_mode="r")
        meta = pd.read_csv(
            DATA / f"sentinelx_{split}_sequence_metadata_v1.csv"
        )

        labels = meta["Target Label"].astype(str).to_numpy()

        lstm_predictions = np.empty(Y.shape, dtype=np.float32)

        with torch.no_grad():
            for start in range(0, len(X), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(X))

                xb = torch.from_numpy(
                    np.array(X[start:end], dtype=np.float32, copy=True)
                ).to(device)

                lstm_predictions[start:end] = (
                    model(xb).cpu().numpy()
                )

        target = np.array(Y, dtype=np.float32, copy=True)
        persistence_predictions = np.array(
            X[:, -1, :], dtype=np.float32
        )

        print("\n" + "=" * 80)
        print(f"{split.upper()} — PER-LABEL PREDICTION PERFORMANCE")
        print("=" * 80)

        rows = []

        for label in sorted(np.unique(labels)):
            mask = labels == label

            lstm_metrics = calculate_metrics(
                lstm_predictions[mask], target[mask]
            )
            persistence_metrics = calculate_metrics(
                persistence_predictions[mask], target[mask]
            )

            rows.append({
                "Label": label,
                "Samples": int(mask.sum()),
                "LSTM MAE": lstm_metrics[0],
                "Persistence MAE": persistence_metrics[0],
                "MAE improvement %": (
                    (persistence_metrics[0] - lstm_metrics[0])
                    / persistence_metrics[0] * 100
                ),
                "LSTM RMSE": lstm_metrics[1],
                "Persistence RMSE": persistence_metrics[1],
                "LSTM Huber": lstm_metrics[2],
                "Persistence Huber": persistence_metrics[2],
            })

        df = pd.DataFrame(rows)
        print(df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}"
        ))


if __name__ == "__main__":
    main()
