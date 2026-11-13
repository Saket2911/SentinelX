from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(r"F:\Projects\SentinelX")
DATA = ROOT / "Datasets"
MODEL_DIR = ROOT / "Models" / "lstm_world_model_v2"
CHECKPOINT = MODEL_DIR / "best_model.pt"

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


def metrics(pred, target):
    error = pred - target
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    abs_error = np.abs(error)
    huber = np.where(
        abs_error <= 1.0,
        0.5 * error ** 2,
        abs_error - 0.5
    ).mean()

    return mae, rmse, huber


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

    model = SentinelXLSTM(
        ckpt["input_features"],
        ckpt["hidden_size"],
        ckpt["num_layers"],
        ckpt["dropout"]
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print("SentinelX V2 checkpoint epoch:", ckpt["epoch"])
    print("Device:", device)

    for split in ["validation", "test"]:
        X = np.load(DATA / f"sentinelx_{split}_X_seq10_v1.npy", mmap_mode="r")
        Y = np.load(DATA / f"sentinelx_{split}_y_next_v1.npy", mmap_mode="r")

        lstm_predictions = np.empty(Y.shape, dtype=np.float32)

        with torch.no_grad():
            for start in range(0, len(X), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(X))

                xb = torch.from_numpy(
                    np.array(X[start:end], dtype=np.float32, copy=True)
                ).to(device)

                pred = model(xb)
                lstm_predictions[start:end] = pred.cpu().numpy()

        # Persistence predicts the next state using the last observed state.
        persistence_predictions = np.array(X[:, -1, :], dtype=np.float32)

        target = np.array(Y, dtype=np.float32, copy=True)

        lstm_mae, lstm_rmse, lstm_huber = metrics(lstm_predictions, target)
        pers_mae, pers_rmse, pers_huber = metrics(persistence_predictions, target)

        print("\n" + "=" * 70)
        print(split.upper(), "RESULTS")
        print("=" * 70)
        print(f"{'Metric':<12}{'LSTM':>14}{'Persistence':>16}{'LSTM improvement':>20}")

        for name, lstm_value, pers_value in [
            ("MAE", lstm_mae, pers_mae),
            ("RMSE", lstm_rmse, pers_rmse),
            ("Huber", lstm_huber, pers_huber),
        ]:
            improvement = (pers_value - lstm_value) / pers_value * 100
            print(
                f"{name:<12}{lstm_value:>14.6f}"
                f"{pers_value:>16.6f}{improvement:>19.2f}%"
            )


if __name__ == "__main__":
    main()
