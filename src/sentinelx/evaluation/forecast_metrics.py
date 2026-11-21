"""Forecast evaluation primitives for the SentinelX world model.

Every function operates on NumPy arrays in *preprocessed model feature space*
unless the caller explicitly passes raw values. Nothing here tunes a model: this
module only measures forecasts, baselines and residual calibration.

Design rules:
  * shapes are validated, never broadcast silently;
  * non-finite values are rejected rather than dropped;
  * persistence (repeat the last state) is always available as the baseline;
  * detection metrics are computed from an explicit threshold and labelled as a
    descriptive operating point, not as a validated detector.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np


BENIGN_LABELS = frozenset({"benign", "normal", "no flows"})


def _as_2d(array: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional, received shape {values.shape}.")
    if values.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{name} must be numeric.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values.")
    return values.astype(np.float64, copy=False)


def _paired(actual: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    actual = _as_2d(actual, "actual")
    predicted = _as_2d(predicted, "predicted")
    if actual.shape != predicted.shape:
        raise ValueError(
            f"actual and predicted must have identical shapes, received {actual.shape} and {predicted.shape}."
        )
    return actual, predicted


def huber_loss(residuals: np.ndarray, delta: float = 1.0) -> float:
    """Mean Huber loss, matching ``torch.nn.HuberLoss`` with mean reduction."""
    if delta <= 0:
        raise ValueError("delta must be positive.")
    absolute = np.abs(residuals)
    quadratic = np.minimum(absolute, delta)
    linear = absolute - quadratic
    return float(np.mean(0.5 * quadratic**2 + delta * linear))


def error_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    delta: float = 1.0,
) -> dict[str, float]:
    """Aggregate one-step forecast error."""
    actual, predicted = _paired(actual, predicted)
    residuals = actual - predicted
    absolute = np.abs(residuals)
    mse = float(np.mean(residuals**2))
    return {
        "mae": float(absolute.mean()),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "huber": huber_loss(residuals, delta=delta),
        "max_abs_error": float(absolute.max()),
        "p95_abs_error": float(np.percentile(absolute, 95)),
        "mean_signed_error": float(residuals.mean()),
        "samples": float(actual.shape[0]),
        "features": float(actual.shape[1]),
    }


def per_feature_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    feature_names: Sequence[str] | None = None,
    delta: float = 1.0,
    top: int | None = None,
) -> list[dict[str, float | str]]:
    """Per-feature MAE/RMSE/Huber, ordered by MAE descending."""
    actual, predicted = _paired(actual, predicted)
    if feature_names is not None and len(feature_names) != actual.shape[1]:
        raise ValueError(
            f"feature_names has {len(feature_names)} entries for {actual.shape[1]} features."
        )
    names = list(feature_names) if feature_names is not None else [
        f"feature_{index}" for index in range(actual.shape[1])
    ]
    rows: list[dict[str, float | str]] = []
    for index, name in enumerate(names):
        residual = actual[:, index] - predicted[:, index]
        mse = float(np.mean(residual**2))
        rows.append(
            {
                "feature": name,
                "mae": float(np.abs(residual).mean()),
                "rmse": float(np.sqrt(mse)),
                "huber": huber_loss(residual, delta=delta),
                "mean_signed_error": float(residual.mean()),
            }
        )
    rows.sort(key=lambda row: float(row["mae"]), reverse=True)
    return rows[:top] if top else rows


def error_quantiles(
    residuals: np.ndarray,
    probabilities: Iterable[float] = (0.5, 0.9, 0.95, 0.99, 0.999),
) -> dict[str, float]:
    """Quantiles of per-observation mean absolute residual."""
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim == 2:
        values = np.abs(values).mean(axis=1)
    if values.ndim != 1:
        raise ValueError("residuals must be one- or two-dimensional.")
    if values.size == 0:
        raise ValueError("residuals cannot be empty.")
    if not np.isfinite(values).all():
        raise ValueError("residuals contain non-finite values.")
    values = np.abs(values)
    return {
        f"{probability:g}": float(np.percentile(values, probability * 100))
        for probability in probabilities
    }


def per_label_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    labels: Sequence[str],
    delta: float = 1.0,
) -> list[dict[str, object]]:
    """Forecast quality per dataset reference label."""
    actual, predicted = _paired(actual, predicted)
    if len(labels) != actual.shape[0]:
        raise ValueError("labels length must match the number of rows.")
    rows: list[dict[str, object]] = []
    for label in sorted({str(value) for value in labels}):
        mask = np.asarray([str(value) == label for value in labels])
        if not mask.any():
            continue
        rows.append(
            {
                "label": label,
                "samples": int(mask.sum()),
                **error_metrics(actual[mask], predicted[mask], delta=delta),
            }
        )
    return rows


def persistence_forecast(sequences: np.ndarray) -> np.ndarray:
    """Baseline: the next state equals the most recent observed state."""
    values = np.asarray(sequences, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(
            "sequences must have shape (samples, sequence_length, features)."
        )
    if not np.isfinite(values).all():
        raise ValueError("sequences contain non-finite values.")
    return values[:, -1, :].copy()


def multi_step_recursive(
    predict_next: Callable[[np.ndarray], np.ndarray],
    sequence: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Recursive multi-step forecast, re-feeding predictions.

    Returns shape ``(horizon, features)``. The caller owns the model; this helper
    exists so multi-step evaluation is implemented identically everywhere.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    window = np.asarray(sequence, dtype=np.float64)
    if window.ndim != 2:
        raise ValueError("sequence must have shape (sequence_length, features).")
    if not np.isfinite(window).all():
        raise ValueError("sequence contains non-finite values.")
    forecasts: list[np.ndarray] = []
    for _ in range(horizon):
        prediction = np.asarray(predict_next(window), dtype=np.float64)
        if prediction.shape != (window.shape[1],):
            raise ValueError(
                f"predict_next must return ({window.shape[1]},), received {prediction.shape}."
            )
        if not np.isfinite(prediction).all():
            raise ValueError("predict_next returned non-finite values.")
        forecasts.append(prediction)
        window = np.vstack([window[1:], prediction])
    return np.vstack(forecasts)


def multi_step_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    horizons: Sequence[int],
    delta: float = 1.0,
) -> list[dict[str, float | int]]:
    """Error at each horizon for aligned forecasts.

    ``actual`` and ``predicted`` have shape ``(origins, max_horizon, features)``.
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have identical shapes.")
    if actual.ndim != 3:
        raise ValueError("expected shape (origins, horizons, features).")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("arrays contain non-finite values.")
    rows: list[dict[str, float | int]] = []
    for horizon in horizons:
        if horizon < 1 or horizon > actual.shape[1]:
            raise ValueError(f"horizon {horizon} is outside the available range.")
        metrics = error_metrics(actual[:, horizon - 1, :], predicted[:, horizon - 1, :], delta=delta)
        rows.append({"horizon": int(horizon), "origins": int(actual.shape[0]), **metrics})
    return rows


def calibrate_threshold(
    benign_scores: np.ndarray,
    percentile: float = 99.0,
) -> dict[str, float]:
    """Threshold from benign residual scores only (e.g. validation benign p99)."""
    values = np.asarray(benign_scores, dtype=np.float64)
    if values.ndim == 2:
        values = np.abs(values).mean(axis=1)
    if values.size == 0:
        raise ValueError("benign_scores cannot be empty.")
    if not np.isfinite(values).all():
        raise ValueError("benign_scores contain non-finite values.")
    if not 0 < percentile < 100:
        raise ValueError("percentile must be in (0, 100).")
    return {
        "threshold": float(np.percentile(values, percentile)),
        "percentile": float(percentile),
        "benign_samples": float(values.size),
        "benign_median": float(np.median(values)),
    }


def detection_metrics(
    scores: np.ndarray,
    labels: Sequence[str],
    threshold: float,
    exclude_labels: Iterable[str] = ("NO_FLOWS",),
    benign_labels: Iterable[str] = BENIGN_LABELS,
) -> dict[str, float | int]:
    """Descriptive operating point of a residual threshold.

    ``labels`` are dataset reference annotations. These counts are reported for
    transparency and must never be presented as a validated detector.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim == 2:
        values = np.abs(values).mean(axis=1)
    if values.ndim != 1:
        raise ValueError("scores must be one- or two-dimensional.")
    if len(values) != len(labels):
        raise ValueError("scores and labels must have the same length.")
    if not np.isfinite(values).all():
        raise ValueError("scores contain non-finite values.")
    if threshold < 0:
        raise ValueError("threshold cannot be negative.")

    excluded = {str(label) for label in exclude_labels}
    benign = {str(label).lower() for label in benign_labels}
    keep = np.asarray([str(label) not in excluded for label in labels])
    if not keep.any():
        raise ValueError("all observations were excluded.")
    scores_kept = values[keep]
    attack = np.asarray(
        [str(label).lower() not in benign for label, kept in zip(labels, keep) if kept],
        dtype=bool,
    )
    flagged = scores_kept >= threshold

    tp = int(np.sum(flagged & attack))
    fp = int(np.sum(flagged & ~attack))
    tn = int(np.sum(~flagged & ~attack))
    fn = int(np.sum(~flagged & attack))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": 1 - specificity,
        "f1_score": f1,
        "accuracy": accuracy,
        "threshold": float(threshold),
        "scored_observations": int(keep.sum()),
        "excluded_observations": int((~keep).sum()),
        "interpretation": (
            "Descriptive operating point at a frozen residual threshold. "
            "Not a validated intrusion-detection performance claim."
        ),
    }
