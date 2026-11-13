import numpy as np
import pandas as pd
import torch

from sentinelx.evaluation.metrics import (
    calculate_metrics,
    calculate_improvements,
)


def evaluate_one_step(
    model,
    X,
    y,
    metadata,
    feature_names,
    batch_size=1024,
    huber_delta=1.0,
):
    """
    Evaluate the SentinelX world model on one-step predictions.

    The persistence baseline predicts the next state by
    repeating the last observed state in each input sequence.

    Returns:
        summary: Overall metrics and metadata.
        feature_results: Feature-wise metrics.
        label_results: Label-wise metrics.
        predictions: Model predictions.
        baseline_predictions: Persistence predictions.
    """

    # Validate batch size.
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, (int, np.integer))
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer.")

    # Validate Huber delta before inference.
    try:
        huber_delta = float(huber_delta)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            "huber_delta must be a finite positive number."
        )

    if not np.isfinite(huber_delta) or huber_delta <= 0:
        raise ValueError(
            "huber_delta must be a finite positive number."
        )

    # Validate input arrays.
    try:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("X and y must contain numeric values.")

    if X.ndim != 3 or y.ndim != 2:
        raise ValueError("X must be 3D and y must be 2D.")

    sample_count, sequence_length, input_features = X.shape

    if sample_count == 0:
        raise ValueError("Evaluation dataset is empty.")

    if sequence_length < 1 or input_features < 1:
        raise ValueError(
            "X must have positive sequence and feature dimensions."
        )

    if y.shape != (sample_count, input_features):
        raise ValueError("X and y shapes do not match.")

    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite values.")

    if not np.isfinite(y).all():
        raise ValueError("y contains non-finite values.")

    # Validate metadata.
    if not isinstance(metadata, pd.DataFrame):
        raise ValueError("metadata must be a pandas DataFrame.")

    if len(metadata) != sample_count:
        raise ValueError("Metadata and sequence counts do not match.")

    if not {"Target Label", "Segment ID"}.issubset(metadata.columns):
        raise ValueError("Required metadata columns are missing.")

    # Validate feature names.
    if len(feature_names) != input_features:
        raise ValueError(
            "Feature-name count does not match model input."
        )

    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Feature names must be unique.")

    # Validate model compatibility.
    if model.input_size != input_features:
        raise ValueError("Model input size does not match X.")

    device = next(model.parameters()).device
    model.eval()

    predictions = np.empty(
        (sample_count, input_features),
        dtype=np.float32,
    )

    baseline_predictions = np.empty_like(predictions)

    with torch.inference_mode():
        for start in range(0, sample_count, batch_size):
            end = min(start + batch_size, sample_count)

            x_batch = torch.from_numpy(
                np.array(X[start:end], dtype=np.float32, copy=True)
            ).to(device)

            predicted = model(x_batch)

            predictions[start:end] = predicted.cpu().numpy()

            baseline_predictions[start:end] = (
                x_batch[:, -1, :].cpu().numpy()
            )

    targets = y

    if not np.isfinite(predictions).all():
        raise ValueError("Model predictions contain non-finite values.")

    if not np.isfinite(baseline_predictions).all():
        raise ValueError("Baseline predictions contain non-finite values.")

    if not np.isfinite(targets).all():
        raise ValueError("Targets contain non-finite values.")

    model_metrics = calculate_metrics(
        predictions,
        targets,
        huber_delta,
    )

    baseline_metrics = calculate_metrics(
        baseline_predictions,
        targets,
        huber_delta,
    )

    improvements = calculate_improvements(
        model_metrics,
        baseline_metrics,
    )

    summary = {
        "samples": int(sample_count),
        "sequence_length": int(sequence_length),
        "input_features": int(input_features),
        "device": str(device),
        "huber_delta": float(huber_delta),
        "model_metrics": model_metrics,
        "persistence_metrics": baseline_metrics,
        "improvement_over_baseline_percent": improvements,
    }

    # Feature-wise metrics.
    model_error = predictions.astype(np.float64) - targets
    baseline_error = (
        baseline_predictions.astype(np.float64) - targets
    )

    model_mae = np.mean(np.abs(model_error), axis=0)
    model_rmse = np.sqrt(np.mean(model_error ** 2, axis=0))

    baseline_mae = np.mean(np.abs(baseline_error), axis=0)
    baseline_rmse = np.sqrt(
        np.mean(baseline_error ** 2, axis=0)
    )

    feature_results = pd.DataFrame({
        "Feature": feature_names,
        "LSTM_MAE": model_mae,
        "LSTM_RMSE": model_rmse,
        "Baseline_MAE": baseline_mae,
        "Baseline_RMSE": baseline_rmse,
    })

    feature_results["MAE_Improvement_Percent"] = (
        (baseline_mae - model_mae)
        / np.where(baseline_mae == 0, np.nan, baseline_mae)
    ) * 100

    feature_results = feature_results.sort_values(
        "LSTM_MAE",
        ascending=False,
    ).reset_index(drop=True)

    # Label-wise metrics.
    labels = (
        metadata["Target Label"]
        .fillna("UNKNOWN")
        .astype(str)
        .to_numpy()
    )

    label_rows = []

    for label in pd.unique(labels):
        mask = labels == label

        model_label_metrics = calculate_metrics(
            predictions[mask],
            targets[mask],
            huber_delta,
        )

        baseline_label_metrics = calculate_metrics(
            baseline_predictions[mask],
            targets[mask],
            huber_delta,
        )

        label_rows.append({
            "Label": label,
            "Samples": int(mask.sum()),
            "LSTM_Huber": model_label_metrics["huber_loss"],
            "LSTM_MAE": model_label_metrics["mae"],
            "LSTM_RMSE": model_label_metrics["rmse"],
            "Baseline_Huber": baseline_label_metrics["huber_loss"],
            "Baseline_MAE": baseline_label_metrics["mae"],
            "Baseline_RMSE": baseline_label_metrics["rmse"],
        })

    label_results = pd.DataFrame(label_rows)

    return {
        "summary": summary,
        "feature_results": feature_results,
        "label_results": label_results,
        "predictions": predictions,
        "baseline_predictions": baseline_predictions,
    }
