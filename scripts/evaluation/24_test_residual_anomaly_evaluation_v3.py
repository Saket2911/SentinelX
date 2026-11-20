import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("/mnt/f/Projects/SentinelX")

DATA_DIR = ROOT / "Datasets"
MODEL_DIR = ROOT / "Models" / "lstm_world_model_v3_gpu_run2"

CHECKPOINT_PATH = MODEL_DIR / "best_model.pt"

X_PATH = DATA_DIR / "sentinelx_test_X_seq10_v3.npy"
Y_PATH = DATA_DIR / "sentinelx_test_y_seq10_v3.npy"
METADATA_PATH = DATA_DIR / "sentinelx_test_metadata_seq10_v3.csv"

OUTPUT_DIR = MODEL_DIR / "test_anomaly_evaluation"

# Frozen from benign validation data. DO NOT recalibrate on test.
ANOMALY_THRESHOLD = 1.88139677

BATCH_SIZE = 4096

torch.set_num_threads(4)


# ============================================================
# MODEL — SAME V3 ARCHITECTURE
# ============================================================

class SentinelXLSTM(nn.Module):

    def __init__(self, input_features, hidden_size, num_layers, dropout):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

        self.output_layer = nn.Linear(
            hidden_size,
            input_features
        )

    def forward(self, x):
        sequence_output, _ = self.lstm(x)
        last_state = sequence_output[:, -1, :]
        last_state = self.dropout(last_state)
        return self.output_layer(last_state)


# ============================================================
# METRICS
# ============================================================

def calculate_detection_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)

    tp = int(np.sum(actual & predicted))
    tn = int(np.sum(~actual & ~predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    accuracy = (
        (tp + tn) / (tp + tn + fp + fn)
        if tp + tn + fp + fn else 0.0
    )

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "f1_score": f1,
        "accuracy": accuracy
    }


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 70)
    print("SENTINELX V3 — HELD-OUT TEST ANOMALY EVALUATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    print("\nLoading trained V3 checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

    model = SentinelXLSTM(
        input_features=checkpoint["input_features"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("Best epoch:", checkpoint["epoch"])
    print("Input features:", checkpoint["input_features"])
    print("Checkpoint loaded.")

    # --------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------

    print("\nLoading held-out test data...")

    X = np.load(X_PATH, mmap_mode="r")
    y = np.load(Y_PATH, mmap_mode="r")
    metadata = pd.read_csv(METADATA_PATH)

    assert X.shape[0] == y.shape[0] == len(metadata)
    assert X.shape[1] == checkpoint["sequence_length"]
    assert X.shape[2] == checkpoint["input_features"]
    assert y.shape[1] == checkpoint["input_features"]

    print("Test X:", X.shape)
    print("Test y:", y.shape)
    print("Metadata:", metadata.shape)

    labels = metadata["Target Label"].astype(str)

    print("\nTest label distribution:")
    print(labels.value_counts())

    # --------------------------------------------------------
    # GENERATE PREDICTIONS AND RESIDUAL SCORES
    # --------------------------------------------------------

    print("\nGenerating test predictions...")

    sample_count = X.shape[0]

    anomaly_scores = np.empty(
        sample_count,
        dtype=np.float32
    )

    with torch.inference_mode():

        for start in range(0, sample_count, BATCH_SIZE):

            end = min(start + BATCH_SIZE, sample_count)

            x_batch = torch.from_numpy(
                np.array(X[start:end], dtype=np.float32)
            ).to(device)

            y_batch = torch.from_numpy(
                np.array(y[start:end], dtype=np.float32)
            ).to(device)

            predicted = model(x_batch)

            residual = torch.abs(predicted - y_batch)

            batch_scores = residual.mean(dim=1)

            anomaly_scores[start:end] = (
                batch_scores.cpu().numpy()
            )

            if start == 0 or end == sample_count:
                print(f"Processed {end:,}/{sample_count:,}")

    if not np.isfinite(anomaly_scores).all():
        raise ValueError("Non-finite anomaly scores detected.")

    # --------------------------------------------------------
    # APPLY FROZEN VALIDATION THRESHOLD
    # --------------------------------------------------------

    print("\nApplying frozen validation threshold:")
    print(f"Threshold: {ANOMALY_THRESHOLD:.8f}")

    results = metadata.copy()
    results["Anomaly Score"] = anomaly_scores
    results["Anomaly"] = anomaly_scores > ANOMALY_THRESHOLD

    results.to_csv(
        OUTPUT_DIR / "test_anomaly_scores.csv",
        index=False
    )

    # --------------------------------------------------------
    # SCORE DISTRIBUTIONS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST ANOMALY SCORE DISTRIBUTION")
    print("=" * 70)

    distribution = []

    for label in ["Benign", "Bot", "NO_FLOWS"]:

        mask = labels == label

        if mask.sum() == 0:
            continue

        scores = anomaly_scores[mask]

        distribution.append({
            "Label": label,
            "Samples": int(mask.sum()),
            "Mean": float(np.mean(scores)),
            "Median": float(np.median(scores)),
            "Std": float(np.std(scores)),
            "P90": float(np.percentile(scores, 90)),
            "P95": float(np.percentile(scores, 95)),
            "P99": float(np.percentile(scores, 99)),
            "Above_Threshold": int(
                np.sum(scores > ANOMALY_THRESHOLD)
            ),
            "Above_Threshold_Percent": float(
                np.mean(scores > ANOMALY_THRESHOLD) * 100
            )
        })

    distribution_df = pd.DataFrame(distribution)

    print(distribution_df.to_string(index=False))

    distribution_df.to_csv(
        OUTPUT_DIR / "test_score_distribution.csv",
        index=False
    )

    # --------------------------------------------------------
    # BENIGN VS BOT DETECTION METRICS
    # --------------------------------------------------------

    benign_mask = labels == "Benign"
    bot_mask = labels == "Bot"
    no_flows_mask = labels == "NO_FLOWS"

    if benign_mask.sum() == 0 or bot_mask.sum() == 0:
        raise ValueError("Benign or Bot test samples are missing.")

    # NO_FLOWS is excluded from binary detection metrics.
    evaluation_mask = benign_mask | bot_mask

    actual_attack = bot_mask[evaluation_mask]

    predicted_anomaly = (
        anomaly_scores[evaluation_mask] > ANOMALY_THRESHOLD
    )

    metrics = calculate_detection_metrics(
        actual_attack,
        predicted_anomaly
    )

    print("\n" + "=" * 70)
    print("BENIGN VS BOT — HELD-OUT TEST RESULTS")
    print("=" * 70)

    print("\nFrozen threshold:", f"{ANOMALY_THRESHOLD:.8f}")

    for name, value in metrics.items():

        if isinstance(value, float):
            print(f"{name:25s}: {value:.6f}")
        else:
            print(f"{name:25s}: {value}")

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    summary = {
        "experiment": "V3 held-out test residual anomaly evaluation",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint": str(CHECKPOINT_PATH),
        "device": str(device),
        "test_samples": int(sample_count),
        "threshold": ANOMALY_THRESHOLD,
        "threshold_source": "Frozen 99th percentile of benign validation scores",
        "threshold_recalibrated_on_test": False,
        "score_definition": "Mean absolute residual across 209 features",
        "detection_metrics_benign_vs_bot": metrics,
        "label_distribution": {
            str(k): int(v)
            for k, v in labels.value_counts().items()
        },
        "no_flows_in_binary_metrics": False,
        "test_used_for_model_training": False,
        "evaluation_seconds": time.time() - start_time
    }

    with open(
        OUTPUT_DIR / "test_anomaly_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=4)

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("HELD-OUT TEST EVALUATION COMPLETED")
    print("=" * 70)

    print("\nSaved outputs:")
    print(OUTPUT_DIR / "test_anomaly_scores.csv")
    print(OUTPUT_DIR / "test_score_distribution.csv")
    print(OUTPUT_DIR / "test_anomaly_summary.json")

    print("\nNO_FLOWS excluded from binary detection metrics.")
    print("Validation threshold was kept unchanged.")
    print("Test data was not used for training or threshold tuning.")

    print("=" * 70)


if __name__ == "__main__":
    main()
