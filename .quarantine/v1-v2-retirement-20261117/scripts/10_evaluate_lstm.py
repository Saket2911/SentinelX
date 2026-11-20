
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentinelx.data.loader import load_dataset
from sentinelx.data.splitter import split_dataset
from sentinelx.features.sequences import (
    build_network_states,
    make_sequences,
)
from sentinelx.models.lstm_world_model import LSTMWorldModel


DATASET_PATH = (
    PROJECT_ROOT
    / "Datasets"
    / "Wednesday-14-02-2018_prepared.csv"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "lstm_world_model.pt"
)

PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "preprocessors"
    / "state_preprocessor.joblib"
)

REPORT_DIR = PROJECT_ROOT / "reports"


def calculate_metrics(actual, predicted):
    errors = actual - predicted

    return {
        "MSE": float(np.mean(errors ** 2)),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "MAE": float(np.mean(np.abs(errors))),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU unavailable. Evaluation stopped."
        )

    device = torch.device("cuda")

    print("Evaluation device:", device)
    print("GPU:", torch.cuda.get_device_name(0))

    # 1. Load data and reproduce the original chronological split.
    print("\n1. Loading dataset...")
    df = load_dataset(DATASET_PATH)

    _, _, test_df = split_dataset(df)

    # 2. Rebuild test states using the original pipeline.
    print("2. Building test network states...")
    test_states = build_network_states(test_df)

    # Reuse the saved training-fitted preprocessor.
    print("3. Loading saved state preprocessor...")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    test_states = preprocessor.transform(test_states)

    # 3. Create test sequences.
    print("4. Creating test sequences...")
    X_test, y_test, timestamps, labels, feature_columns = (
        make_sequences(test_states, sequence_length=10)
    )

    if len(X_test) == 0:
        raise ValueError("No test sequences were created.")

    if not np.isfinite(X_test).all():
        raise ValueError("Test inputs contain non-finite values.")

    if not np.isfinite(y_test).all():
        raise ValueError("Test targets contain non-finite values.")

    print("Test sequences:", X_test.shape)
    print("Target states:", y_test.shape)

    # 4. Load the saved best checkpoint.
    print("5. Loading best LSTM checkpoint...")
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    if checkpoint["input_size"] != X_test.shape[2]:
        raise ValueError("Model and test feature counts differ.")

    model = LSTMWorldModel(
        input_size=X_test.shape[2],
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 5. Predict in small batches to conserve GPU memory.
    print("6. Predicting future network states...")
    batch_size = 32
    predictions = []

    with torch.no_grad():
        for start in range(0, len(X_test), batch_size):
            X_batch = torch.as_tensor(
                X_test[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )

            batch_predictions = model(X_batch)
            predictions.append(batch_predictions.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)

    # Baseline: the next state equals the last observed state.
    baseline_predictions = X_test[:, -1, :]

    # 6. Overall test metrics.
    lstm_metrics = calculate_metrics(y_test, predictions)
    baseline_metrics = calculate_metrics(
        y_test, baseline_predictions
    )

    print("\n===== HELD-OUT TEST RESULTS =====")
    print("LSTM:", lstm_metrics)
    print("Persistence baseline:", baseline_metrics)

    improvement = (
        (baseline_metrics["MSE"] - lstm_metrics["MSE"])
        / baseline_metrics["MSE"]
        * 100
        if baseline_metrics["MSE"] > 0
        else float("nan")
    )

    print(f"LSTM MSE improvement over baseline: {improvement:.2f}%")

    # 7. Per-label results for descriptive analysis only.
    labels = np.asarray(labels)
    label_rows = []

    for label in sorted(set(labels)):
        mask = labels == label

        lstm_label_metrics = calculate_metrics(
            y_test[mask], predictions[mask]
        )
        baseline_label_metrics = calculate_metrics(
            y_test[mask], baseline_predictions[mask]
        )

        label_rows.append({
            "target_label": str(label),
            "samples": int(mask.sum()),
            "lstm_mse": lstm_label_metrics["MSE"],
            "lstm_mae": lstm_label_metrics["MAE"],
            "baseline_mse": baseline_label_metrics["MSE"],
            "baseline_mae": baseline_label_metrics["MAE"],
        })

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    overall_report = pd.DataFrame([
        {"model": "LSTM World Model", **lstm_metrics},
        {"model": "Persistence baseline", **baseline_metrics},
    ])

    overall_report["lstm_mse_improvement_percent"] = [
        improvement, improvement
    ]

    overall_path = REPORT_DIR / "lstm_test_evaluation.csv"
    label_path = REPORT_DIR / "lstm_test_by_label.csv"

    overall_report.to_csv(overall_path, index=False)
    pd.DataFrame(label_rows).to_csv(label_path, index=False)

    print("\nReports saved:")
    print(overall_path)
    print(label_path)

    print(
        "\nEvaluation complete. The model was not retrained, "
        "and the test set was not used for model selection."
    )


if __name__ == "__main__":
    main()
