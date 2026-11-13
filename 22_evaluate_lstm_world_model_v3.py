
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(r"F:\Projects\SentinelX")

DATA_DIR = ROOT / "Datasets"

MODEL_DIR = ROOT / "Models" / "lstm_world_model_v3_gpu_run2"

CHECKPOINT_PATH = MODEL_DIR / "best_model.pt"

X_PATH = DATA_DIR / "sentinelx_test_X_seq10_v3.npy"
Y_PATH = DATA_DIR / "sentinelx_test_y_seq10_v3.npy"
METADATA_PATH = DATA_DIR / "sentinelx_test_metadata_seq10_v3.csv"

OUTPUT_DIR = MODEL_DIR / "test_evaluation"

BATCH_SIZE = 4096
HUBER_DELTA = 1.0

torch.set_num_threads(4)


# ============================================================
# MODEL DEFINITION — EXACT TRAINING ARCHITECTURE
# ============================================================

class SentinelXLSTM(nn.Module):

    def __init__(
        self,
        input_features,
        hidden_size,
        num_layers,
        dropout
    ):
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

def calculate_metrics(predictions, targets):

    error = predictions - targets

    absolute_error = np.abs(error)

    mse = np.mean(error ** 2)

    mae = np.mean(absolute_error)

    rmse = np.sqrt(mse)

    # Huber loss with delta = 1.0
    huber = np.where(
        absolute_error <= HUBER_DELTA,
        0.5 * error ** 2,
        HUBER_DELTA * (
            absolute_error - 0.5 * HUBER_DELTA
        )
    )

    huber_loss = np.mean(huber)

    return {
        "huber_loss": float(huber_loss),
        "mae": float(mae),
        "rmse": float(rmse)
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    start_time = time.time()

    print("=" * 70)
    print("SENTINELX V3 — FINAL HELD-OUT TEST EVALUATION")
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

    print("\nLoading best checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

    input_features = checkpoint["input_features"]
    hidden_size = checkpoint["hidden_size"]
    num_layers = checkpoint["num_layers"]
    dropout = checkpoint["dropout"]

    feature_names = checkpoint["feature_names"]

    print("Best epoch:", checkpoint["epoch"])
    print("Input features:", input_features)
    print("Hidden size:", hidden_size)
    print("Number of layers:", num_layers)
    print("Validation Huber:", checkpoint["validation_huber_loss"])

    # --------------------------------------------------------
    # LOAD TEST ARRAYS
    # --------------------------------------------------------

    print("\nLoading held-out test arrays...")

    X = np.load(X_PATH, mmap_mode="r")
    y = np.load(Y_PATH, mmap_mode="r")

    metadata = pd.read_csv(METADATA_PATH)

    print("Test X:", X.shape)
    print("Test y:", y.shape)
    print("Metadata:", metadata.shape)

    # --------------------------------------------------------
    # VALIDATION CHECKS
    # --------------------------------------------------------

    assert X.ndim == 3
    assert y.ndim == 2

    assert X.shape[0] == y.shape[0]
    assert X.shape[0] == len(metadata)

    assert X.shape[1] == checkpoint["sequence_length"]
    assert X.shape[2] == input_features
    assert y.shape[1] == input_features

    assert len(feature_names) == input_features

    assert "Target Label" in metadata.columns
    assert "Segment ID" in metadata.columns

    print("\nAll shape and metadata checks passed.")

    print("\nTest label distribution:")
    print(metadata["Target Label"].value_counts(dropna=False))

    # --------------------------------------------------------
    # RECONSTRUCT MODEL
    # --------------------------------------------------------

    print("\nReconstructing SentinelXLSTM V3...")

    model = SentinelXLSTM(
        input_features=input_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print("Checkpoint loaded successfully.")

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    print("\nGenerating test predictions...")

    sample_count = X.shape[0]

    predictions = np.empty(
        (sample_count, input_features),
        dtype=np.float32
    )

    baseline_predictions = np.empty_like(predictions)

    with torch.inference_mode():

        for start in range(0, sample_count, BATCH_SIZE):

            end = min(start + BATCH_SIZE, sample_count)

            x_batch = torch.from_numpy(
                np.array(X[start:end], dtype=np.float32)
            ).to(device)

            # LSTM prediction
            predicted = model(x_batch)

            # Persistence baseline:
            # predict next state = last observed state
            baseline = x_batch[:, -1, :]

            predictions[start:end] = (
                predicted.cpu().numpy()
            )

            baseline_predictions[start:end] = (
                baseline.cpu().numpy()
            )

            if start == 0 or end == sample_count:
                print(
                    f"Processed {end:,}/{sample_count:,} samples"
                )

    # --------------------------------------------------------
    # CHECK NUMERICAL VALIDITY
    # --------------------------------------------------------

    targets = np.asarray(y, dtype=np.float32)

    if not np.isfinite(predictions).all():
        raise ValueError("Non-finite values found in predictions.")

    if not np.isfinite(targets).all():
        raise ValueError("Non-finite values found in targets.")

    if not np.isfinite(baseline_predictions).all():
        raise ValueError("Non-finite values found in baseline.")

    # --------------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------------

    print("\nCalculating overall metrics...")

    lstm_metrics = calculate_metrics(
        predictions,
        targets
    )

    baseline_metrics = calculate_metrics(
        baseline_predictions,
        targets
    )

    print("\n" + "=" * 70)
    print("OVERALL TEST RESULTS")
    print("=" * 70)

    print("\nLSTM WORLD MODEL:")

    for name, value in lstm_metrics.items():
        print(f"{name.upper():15s}: {value:.8f}")

    print("\nPERSISTENCE BASELINE:")

    for name, value in baseline_metrics.items():
        print(f"{name.upper():15s}: {value:.8f}")

    # Positive means LSTM improves over baseline.
    improvements = {}

    for metric in lstm_metrics:

        baseline_value = baseline_metrics[metric]
        lstm_value = lstm_metrics[metric]

        improvement = (
            (baseline_value - lstm_value)
            / baseline_value * 100
            if baseline_value != 0
            else 0.0
        )

        improvements[metric] = float(improvement)

    print("\nLSTM IMPROVEMENT OVER BASELINE:")

    for name, value in improvements.items():
        print(f"{name.upper():15s}: {value:+.4f}%")

    # --------------------------------------------------------
    # FEATURE-WISE ERRORS
    # --------------------------------------------------------

    print("\nCalculating feature-wise errors...")

    lstm_error = predictions - targets

    baseline_error = baseline_predictions - targets

    feature_results = pd.DataFrame({

        "Feature": feature_names,

        "LSTM_MAE": np.mean(
            np.abs(lstm_error),
            axis=0
        ),

        "LSTM_RMSE": np.sqrt(
            np.mean(lstm_error ** 2, axis=0)
        ),

        "Baseline_MAE": np.mean(
            np.abs(baseline_error),
            axis=0
        ),

        "Baseline_RMSE": np.sqrt(
            np.mean(baseline_error ** 2, axis=0)
        )
    })

    feature_results["MAE_Improvement_Percent"] = (
        (
            feature_results["Baseline_MAE"]
            - feature_results["LSTM_MAE"]
        )
        / feature_results["Baseline_MAE"].replace(0, np.nan)
    ) * 100

    feature_results = feature_results.sort_values(
        "LSTM_MAE",
        ascending=False
    )

    feature_results.to_csv(
        OUTPUT_DIR / "feature_wise_errors.csv",
        index=False
    )

    print("\nTop 15 features with highest LSTM MAE:")

    print(
        feature_results[
            ["Feature", "LSTM_MAE", "Baseline_MAE"]
        ].head(15).to_string(index=False)
    )

    # --------------------------------------------------------
    # LABEL-WISE PERFORMANCE
    # --------------------------------------------------------

    print("\nCalculating label-wise performance...")

    label_results = []

    labels = metadata["Target Label"].fillna(
        "UNKNOWN"
    ).astype(str).to_numpy()

    for label in pd.unique(labels):

        mask = labels == label

        label_predictions = predictions[mask]
        label_targets = targets[mask]
        label_baseline = baseline_predictions[mask]

        model_metrics = calculate_metrics(
            label_predictions,
            label_targets
        )

        baseline_label_metrics = calculate_metrics(
            label_baseline,
            label_targets
        )

        label_results.append({

            "Label": label,

            "Samples": int(mask.sum()),

            "LSTM_Huber": model_metrics["huber_loss"],
            "LSTM_MAE": model_metrics["mae"],
            "LSTM_RMSE": model_metrics["rmse"],

            "Baseline_Huber": baseline_label_metrics["huber_loss"],
            "Baseline_MAE": baseline_label_metrics["mae"],
            "Baseline_RMSE": baseline_label_metrics["rmse"]
        })

    label_results = pd.DataFrame(label_results)

    label_results.to_csv(
        OUTPUT_DIR / "label_wise_performance.csv",
        index=False
    )

    print("\nLABEL-WISE RESULTS:")

    print(label_results.to_string(index=False))

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    summary = {

        "model": "SentinelXLSTM V3",

        "checkpoint": str(CHECKPOINT_PATH),

        "best_epoch": checkpoint["epoch"],

        "device": str(device),

        "test_samples": int(sample_count),

        "input_features": int(input_features),

        "sequence_length": int(X.shape[1]),

        "lstm_metrics": lstm_metrics,

        "persistence_baseline_metrics": baseline_metrics,

        "improvement_over_baseline_percent": improvements,

        "test_data_used_for_training": False,

        "test_data_used_for_model_selection": False,

        "evaluation_seconds": time.time() - start_time
    }

    with open(
        OUTPUT_DIR / "test_evaluation_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(summary, f, indent=4)

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)

    print(f"Test samples: {sample_count:,}")
    print(f"Evaluation time: {elapsed:.2f} seconds")

    print("\nSaved files:")

    print(OUTPUT_DIR / "test_evaluation_summary.json")
    print(OUTPUT_DIR / "feature_wise_errors.csv")
    print(OUTPUT_DIR / "label_wise_performance.csv")

    print("\nTest data was not used for training or model selection.")

    print("=" * 70)


if __name__ == "__main__":
    main()
