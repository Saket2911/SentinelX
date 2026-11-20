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
REPORT = MODEL_DIR / "anomaly_score_diagnostic_v1.txt"

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


def calculate_scores(model, X, Y, device):
    scores = np.empty(len(X), dtype=np.float64)

    model.eval()

    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(X))

            xb = torch.from_numpy(
                np.array(X[start:end], dtype=np.float32, copy=True)
            ).to(device)

            yb = torch.from_numpy(
                np.array(Y[start:end], dtype=np.float32, copy=True)
            ).to(device)

            predictions = model(xb)

            batch_scores = torch.mean(
                torch.abs(predictions - yb),
                dim=1
            )

            scores[start:end] = batch_scores.cpu().numpy()

    return scores


def evaluate_partition(name, scores, labels):
    benign_mask = labels == "Benign"
    bot_mask = labels == "Bot"

    print(f"\n{'=' * 70}")
    print(f"{name} DIAGNOSTICS")
    print("=" * 70)

    print("Label counts:")
    print(pd.Series(labels).value_counts().to_string())

    if not benign_mask.any() or not bot_mask.any():
        print("Cannot calculate ROC-AUC/AP: Benign or Bot samples missing.")
        return

    benign_scores = scores[benign_mask]
    bot_scores = scores[bot_mask]

    print("\nScore statistics:")
    print(f"Benign samples: {len(benign_scores)}")
    print(f"Bot samples:    {len(bot_scores)}")

    print(f"Benign mean:   {benign_scores.mean():.6f}")
    print(f"Benign median: {np.median(benign_scores):.6f}")
    print(f"Benign P95:    {np.percentile(benign_scores, 95):.6f}")

    print(f"\nBot mean:      {bot_scores.mean():.6f}")
    print(f"Bot median:    {np.median(bot_scores):.6f}")
    print(f"Bot P95:       {np.percentile(bot_scores, 95):.6f}")

    print(
        "\nBot samples with score greater than Benign median: "
        f"{100 * np.mean(bot_scores > np.median(benign_scores)):.2f}%"
    )

    # Binary comparison only: Bot=1, Benign=0.
    # NO_FLOWS and other labels are excluded.
    mask = benign_mask | bot_mask

    binary_labels = (labels[mask] == "Bot").astype(np.int32)
    binary_scores = scores[mask]

    auc = roc_auc_score(binary_labels, binary_scores)
    ap = average_precision_score(binary_labels, binary_scores)

    print(f"\nROC-AUC:          {auc:.6f}")
    print(f"Average Precision:{ap:.6f}")

    print("\nROC-AUC interpretation:")
    if auc > 0.7:
        print("Scores show useful Bot-vs-Benign ranking.")
    elif auc >= 0.55:
        print("Scores show weak ranking ability.")
    elif auc >= 0.45:
        print("Scores are close to random ranking.")
    else:
        print("Scores may rank Benign above Bot; investigate score direction.")


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    if REPORT.exists():
        raise FileExistsError(
            f"Diagnostic report already exists: {REPORT}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    model = SentinelXLSTM(
        checkpoint["input_features"],
        checkpoint["hidden_size"],
        checkpoint["num_layers"],
        checkpoint["dropout"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("=" * 70)
    print("SENTINELX V2 ANOMALY SCORE DIAGNOSTICS")
    print("=" * 70)
    print("Checkpoint:", CHECKPOINT)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Features:", checkpoint["input_features"])
    print("Device:", device)

    partitions = [
        (
            "VALIDATION",
            DATA / "sentinelx_validation_X_seq10_v1.npy",
            DATA / "sentinelx_validation_y_next_v1.npy",
            DATA / "sentinelx_validation_sequence_metadata_v1.csv"
        ),
        (
            "TEST",
            DATA / "sentinelx_test_X_seq10_v1.npy",
            DATA / "sentinelx_test_y_next_v1.npy",
            DATA / "sentinelx_test_sequence_metadata_v1.csv"
        )
    ]

    report_lines = []

    for name, x_path, y_path, meta_path in partitions:
        X = np.load(x_path, mmap_mode="r")
        Y = np.load(y_path, mmap_mode="r")
        metadata = pd.read_csv(meta_path)

        labels = metadata["Target Label"].astype(str).to_numpy()

        assert len(X) == len(Y) == len(labels)
        assert X.shape[1:] == (10, 212)
        assert Y.shape[1:] == (212,)

        print(f"\nCalculating {name.lower()} scores...")
        scores = calculate_scores(model, X, Y, device)

        evaluate_partition(name, scores, labels)

        benign = scores[labels == "Benign"]
        bot = scores[labels == "Bot"]

        report_lines.extend([
            f"\n{name}",
            f"Samples total: {len(scores)}",
            f"Benign count: {len(benign)}",
            f"Bot count: {len(bot)}",
            f"Benign mean: {benign.mean():.8f}",
            f"Benign median: {np.median(benign):.8f}",
            f"Bot mean: {bot.mean():.8f}",
            f"Bot median: {np.median(bot):.8f}",
        ])

        binary_mask = (labels == "Benign") | (labels == "Bot")
        binary_labels = (labels[binary_mask] == "Bot").astype(np.int32)

        report_lines.append(
            f"ROC-AUC: {roc_auc_score(binary_labels, scores[binary_mask]):.8f}"
        )

        report_lines.append(
            f"Average Precision: "
            f"{average_precision_score(binary_labels, scores[binary_mask]):.8f}"
        )

    REPORT.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    print("\nDiagnostic report saved to:")
    print(REPORT)


if __name__ == "__main__":
    main()
