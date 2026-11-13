
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

X_PATH = DATA_DIR / "sentinelx_validation_X_seq10_v3.npy"
Y_PATH = DATA_DIR / "sentinelx_validation_y_seq10_v3.npy"
METADATA_PATH = DATA_DIR / "sentinelx_validation_metadata_seq10_v3.csv"

OUTPUT_DIR = MODEL_DIR / "validation_anomaly_analysis"

BATCH_SIZE = 4096
BENIGN_THRESHOLD_PERCENTILE = 99.0

torch.set_num_threads(4)


# ============================================================
# MODEL — SAME ARCHITECTURE AS TRAINING
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
    print("SENTINELX V3 — VALIDATION RESIDUAL ANOMALY ANALYSIS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    print("\nLoading trained V3 checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

    input_features = checkpoint["input_features"]

    model = SentinelXLSTM(
        input_features=input_features,
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    print("Best epoch:", checkpoint["epoch"])
    print("Input features:", input_features)
    print("Checkpoint loaded.")

    # --------------------------------------------------------
    # LOAD VALIDATION DATA
    # --------------------------------------------------------

    print("\nLoading validation data...")

    X = np.load(X_PATH, mmap_mode="r")
    y = np.load(Y_PATH, mmap_mode="r")

    metadata = pd.read_csv(METADATA_PATH)

    assert X.shape[0] == y.shape[0] == len(metadata)
    assert X.shape[1] == checkpoint["sequence_length"]
    assert X.shape[2] == input_features
    assert y.shape[1] == input_features

    print("Validation X:", X.shape)
    print("Validation y:", y.shape)
    print("Metadata:", metadata.shape)

    print("\nLabel distribution:")
    print(metadata["Target Label"].value_counts())

    # --------------------------------------------------------
    # GENERATE PREDICTIONS
    # --------------------------------------------------------

    print("\nGenerating validation predictions...")

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

            # Absolute residual for every feature.
            residual = torch.abs(predicted - y_batch)

            # One score per network state.
            batch_scores = residual.mean(dim=1)

            anomaly_scores[start:end] = (
                batch_scores.cpu().numpy()
            )

            if start == 0 or end == sample_count:
                print(
                    f"Processed {end:,}/{sample_count:,}"
                )

    if not np.isfinite(anomaly_scores).all():
        raise ValueError("Non-finite anomaly scores detected.")

    # --------------------------------------------------------
    # BUILD RESULT TABLE
    # --------------------------------------------------------

    results = metadata.copy()

    results["Anomaly Score"] = anomaly_scores

    results.to_csv(
        OUTPUT_DIR / "validation_anomaly_scores.csv",
        index=False
    )

    labels = results["Target Label"].astype(str)

    benign_mask = labels == "Benign"
    attack_mask = labels == "Infilteration"
    no_flows_mask = labels == "NO_FLOWS"

    if benign_mask.sum() == 0:
        raise ValueError("No Benign validation samples found.")

    if attack_mask.sum() == 0:
        raise ValueError("No Infilteration samples found.")

    # --------------------------------------------------------
    # CALIBRATE THRESHOLD USING BENIGN VALIDATION SCORES
    # --------------------------------------------------------

    print("\nCalibrating benign threshold...")

    benign_scores = anomaly_scores[benign_mask]

    threshold = float(
        np.percentile(
            benign_scores,
            BENIGN_THRESHOLD_PERCENTILE
        )
    )

    print(
        f"\nBenign {BENIGN_THRESHOLD_PERCENTILE}th percentile:"
    )

    print(f"Anomaly threshold: {threshold:.8f}")

    # Strictly greater than threshold = anomaly.
    results["Anomaly"] = anomaly_scores > threshold

    # --------------------------------------------------------
    # SCORE DISTRIBUTIONS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ANOMALY SCORE DISTRIBUTION")
    print("=" * 70)

    distribution = []

    for label in ["Benign", "Infilteration", "NO_FLOWS"]:

        mask = labels == label

        if mask.sum() == 0:
            continue

        scores = anomaly_scores[mask]

        row = {
            "Label": label,
            "Samples": int(mask.sum()),
            "Mean": float(np.mean(scores)),
            "Median": float(np.median(scores)),
            "Std": float(np.std(scores)),
            "P90": float(np.percentile(scores, 90)),
            "P95": float(np.percentile(scores, 95)),
            "P99": float(np.percentile(scores, 99)),
            "Above_Threshold": int(
                np.sum(scores > threshold)
            ),
            "Above_Threshold_Percent": float(
                np.mean(scores > threshold) * 100
            )
        }

        distribution.append(row)

    distribution_df = pd.DataFrame(distribution)

    print(distribution_df.to_string(index=False))

    distribution_df.to_csv(
        OUTPUT_DIR / "anomaly_score_distribution.csv",
        index=False
    )

    # --------------------------------------------------------
    # BENIGN VS INFILTRATION
    # --------------------------------------------------------

    # NO_FLOWS is excluded from binary detection metrics.
    evaluation_mask = benign_mask | attack_mask

    actual_attack = attack_mask[evaluation_mask]

    predicted_anomaly = (
        anomaly_scores[evaluation_mask] > threshold
    )

    metrics = calculate_detection_metrics(
        actual_attack,
        predicted_anomaly
    )

    print("\n" + "=" * 70)
    print("BENIGN VS INFILTRATION ANALYSIS")
    print("=" * 70)

    print("\nThreshold:", f"{threshold:.8f}")

    for name, value in metrics.items():

        if isinstance(value, float):
            print(f"{name:25s}: {value:.6f}")
        else:
            print(f"{name:25s}: {value}")

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    summary = {

        "experiment": "V3 residual-based anomaly scoring",

        "checkpoint_epoch": int(checkpoint["epoch"]),

        "checkpoint": str(CHECKPOINT_PATH),

        "device": str(device),

        "validation_samples": int(sample_count),

        "threshold_method": "Benign validation percentile",

        "threshold_percentile": BENIGN_THRESHOLD_PERCENTILE,

        "anomaly_threshold": threshold,

        "score_definition": "Mean absolute residual across 209 features",

        "detection_metrics_benign_vs_infilteration": metrics,

        "label_distribution": {
            str(k): int(v)
            for k, v in labels.value_counts().items()
        },

        "no_flows_in_binary_metrics": False,

        "test_data_used": False,

        "validation_used_for_threshold_calibration": True,

        "validation_used_for_exploratory_detection_analysis": True,

        "evaluation_seconds": time.time() - start_time
    }

    with open(
        OUTPUT_DIR / "anomaly_analysis_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(summary, f, indent=4)

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESIDUAL ANOMALY ANALYSIS COMPLETED")
    print("=" * 70)

    print("\nSaved outputs:")

    print(OUTPUT_DIR / "validation_anomaly_scores.csv")
    print(OUTPUT_DIR / "anomaly_score_distribution.csv")
    print(OUTPUT_DIR / "anomaly_analysis_summary.json")

    print("\nNO_FLOWS was excluded from binary detection metrics.")

    print(
        "\nNote: threshold calibration and detection analysis "
        "use the same validation dataset."
    )

    print("Held-out test data was not used.")

    print("=" * 70)


if __name__ == "__main__":
    main()
