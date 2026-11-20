
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
    PROJECT_ROOT / "artifacts" / "models"
    / "lstm_world_model.pt"
)
PREPROCESSOR_PATH = (
    PROJECT_ROOT / "artifacts" / "preprocessors"
    / "state_preprocessor.joblib"
)
REPORT_DIR = PROJECT_ROOT / "reports"


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Diagnostic device:", device)

    # Reproduce the original chronological test pipeline.
    df = load_dataset(DATASET_PATH)
    _, _, test_df = split_dataset(df)

    test_states = build_network_states(test_df)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    test_states = preprocessor.transform(test_states)

    X, y, timestamps, labels, feature_columns = make_sequences(
        test_states, sequence_length=10
    )

    if len(X) == 0:
        raise ValueError("No test sequences were created.")

    if not (
        np.isfinite(X).all()
        and np.isfinite(y).all()
    ):
        raise ValueError("Non-finite test values detected.")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model = LSTMWorldModel(
        input_size=X.shape[2],
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Predict in small batches.
    predictions = []
    with torch.no_grad():
        for start in range(0, len(X), 32):
            batch = torch.as_tensor(
                X[start:start + 32],
                dtype=torch.float32,
                device=device,
            )
            predictions.append(
                model(batch).cpu().numpy()
            )

    pred = np.concatenate(predictions, axis=0)
    baseline = X[:, -1, :]

    if not np.isfinite(pred).all():
        raise ValueError("Model predictions contain non-finite values.")

    # Check feature alignment before producing reports.
    if len(feature_columns) != y.shape[1]:
        raise ValueError(
            f"Feature-name count {len(feature_columns)} "
            f"does not match target width {y.shape[1]}."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Per-feature error analysis.
    error = y - pred
    baseline_error = y - baseline

    feature_report = pd.DataFrame({
        "feature": list(feature_columns),
        "lstm_mse": np.mean(error ** 2, axis=0),
        "lstm_mae": np.mean(np.abs(error), axis=0),
        "baseline_mse": np.mean(baseline_error ** 2, axis=0),
        "baseline_mae": np.mean(
            np.abs(baseline_error), axis=0
        ),
        "target_mean": np.mean(y, axis=0),
        "target_std": np.std(y, axis=0),
        "target_min": np.min(y, axis=0),
        "target_max": np.max(y, axis=0),
        "target_abs_max": np.max(np.abs(y), axis=0),
        "prediction_abs_max": np.max(np.abs(pred), axis=0),
        "target_abs_gt_5_fraction": np.mean(
            np.abs(y) > 5, axis=0
        ),
        "target_abs_gt_10_fraction": np.mean(
            np.abs(y) > 10, axis=0
        ),
    })

    feature_report = feature_report.sort_values(
        "lstm_mse", ascending=False
    )

    feature_path = REPORT_DIR / "lstm_feature_diagnostics.csv"
    feature_report.to_csv(feature_path, index=False)

    # 2. Find test samples with the largest average errors.
    sample_mse = np.mean(error ** 2, axis=1)
    sample_mae = np.mean(np.abs(error), axis=1)
    largest_feature_index = np.argmax(error ** 2, axis=1)

    sample_report = pd.DataFrame({
        "timestamp": [str(t) for t in timestamps],
        "target_label": [str(label) for label in labels],
        "sample_mse": sample_mse,
        "sample_mae": sample_mae,
        "max_absolute_error": np.max(np.abs(error), axis=1),
        "largest_error_feature": [
            feature_columns[i] for i in largest_feature_index
        ],
        "target_abs_max": np.max(np.abs(y), axis=1),
        "prediction_abs_max": np.max(np.abs(pred), axis=1),
    })

    sample_report = sample_report.sort_values(
        "sample_mse", ascending=False
    )

    sample_path = REPORT_DIR / "lstm_worst_test_samples.csv"
    sample_report.to_csv(sample_path, index=False)

    # 3. Recalculate results separately for each target label.
    label_rows = []
    labels_array = np.asarray(labels)

    for label in sorted(set(labels_array)):
        mask = labels_array == label

        label_rows.append({
            "target_label": str(label),
            "samples": int(mask.sum()),
            "lstm_mse": float(np.mean(error[mask] ** 2)),
            "lstm_mae": float(np.mean(np.abs(error[mask]))),
            "baseline_mse": float(
                np.mean(baseline_error[mask] ** 2)
            ),
            "baseline_mae": float(
                np.mean(np.abs(baseline_error[mask]))
            ),
        })

    label_path = REPORT_DIR / "lstm_diagnostic_by_label.csv"
    pd.DataFrame(label_rows).to_csv(label_path, index=False)

    # 4. Print concise findings.
    print("\n===== TOP 15 FEATURES BY LSTM MSE =====")
    print(
        feature_report.head(15)[[
            "feature",
            "lstm_mse",
            "baseline_mse",
            "target_abs_max",
            "prediction_abs_max",
            "target_abs_gt_10_fraction",
        ]].to_string(index=False)
    )

    print("\n===== TOP 20 WORST TEST SAMPLES =====")
    print(sample_report.head(20).to_string(index=False))

    print("\n===== RESULTS BY LABEL =====")
    print(pd.DataFrame(label_rows).to_string(index=False))

    print("\nReports saved:")
    print(feature_path)
    print(sample_path)
    print(label_path)


if __name__ == "__main__":
    main()
