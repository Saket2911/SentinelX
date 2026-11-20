from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

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


def main():
    X = np.load(DATA / "sentinelx_validation_X_seq10_v1.npy", mmap_mode="r")
    Y = np.load(DATA / "sentinelx_validation_y_next_v1.npy", mmap_mode="r")
    meta = pd.read_csv(DATA / "sentinelx_validation_sequence_metadata_v1.csv")

    labels = meta["Target Label"].astype(str).to_numpy()

    benign_mask = labels == "Benign"
    attack_mask = labels == "Infilteration"
    selected = benign_mask | attack_mask

    print("Benign samples:", int(benign_mask.sum()))
    print("Infiltration samples:", int(attack_mask.sum()))

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

    scores = np.empty(len(X), dtype=np.float64)

    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(X))

            xb = torch.from_numpy(
                np.array(X[start:end], dtype=np.float32, copy=True)
            ).to(device)

            yb = torch.from_numpy(
                np.array(Y[start:end], dtype=np.float32, copy=True)
            ).to(device)

            pred = model(xb)
            scores[start:end] = torch.mean(
                torch.abs(pred - yb), dim=1
            ).cpu().numpy()

    benign_scores = scores[benign_mask]
    attack_scores = scores[attack_mask]

    print("\nVALIDATION SCORE COMPARISON")
    print("--------------------------")
    print(f"Benign mean:       {benign_scores.mean():.6f}")
    print(f"Benign median:     {np.median(benign_scores):.6f}")
    print(f"Infiltration mean: {attack_scores.mean():.6f}")
    print(f"Infiltration median: {np.median(attack_scores):.6f}")

    y_true = (labels[selected] == "Infilteration").astype(int)
    y_score = scores[selected]

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    prevalence = y_true.mean()

    print(f"\nROC-AUC: {auc:.6f}")
    print(f"Average Precision: {ap:.6f}")
    print(f"Infiltration prevalence baseline AP: {prevalence:.6f}")

    if auc > 0.5:
        print("Higher prediction errors tend to rank Infiltration higher.")
    else:
        print("Prediction errors do not rank Infiltration higher than Benign.")


if __name__ == "__main__":
    main()
