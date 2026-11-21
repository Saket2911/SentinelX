"""Controlled training experiments with guardrails.

Design rules:
  * a new experiment always writes to its own directory and never overwrites a
    preserved checkpoint;
  * validation metrics are compared with the persistence baseline at the same
    time, so an "improvement" cannot come from an easier metric;
  * the test partition is never read here — promotion decisions use validation
    data only;
  * every report records seed, hardware, versions, config and duration.

Torch is imported lazily so the guardrail logic stays testable without it.
"""

from __future__ import annotations

import json
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from ..evaluation.forecast_metrics import error_metrics, persistence_forecast
from ..models.variants import VARIANT_DESCRIPTIONS, VARIANT_NAMES, build_world_model


PROMOTION_MIN_MAE_GAIN = 0.01  # 1 % relative MAE improvement over the control
PROMOTION_MIN_RMSE_GAIN = 0.01


@dataclass
class ExperimentConfig:
    name: str
    variant: str = "baseline_v3"
    input_size: int = 209
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    sequence_length: int = 10
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    warmup_fraction: float = 0.05
    scheduler: str = "cosine"
    huber_delta: float = 1.0
    patience: int = 6
    seed: int = 42
    device: str = "auto"

    def validate(self) -> None:
        if self.variant not in VARIANT_NAMES:
            raise ValueError(f"variant must be one of {VARIANT_NAMES}")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive.")
        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must be in (0, 1).")
        if self.grad_clip < 0:
            raise ValueError("grad_clip cannot be negative.")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1).")
        if self.scheduler not in {"cosine", "none"}:
            raise ValueError("scheduler must be 'cosine' or 'none'.")
        if self.patience < 1:
            raise ValueError("patience must be at least 1.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return requested


def _hardware() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = None
    return info


def _shapes(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
    for name, array in (("X_train", X_train), ("X_val", X_val)):
        if array.ndim != 3:
            raise ValueError(f"{name} must be (samples, sequence_length, features).")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values.")
    for name, array in (("y_train", y_train), ("y_val", y_val)):
        if array.ndim != 2:
            raise ValueError(f"{name} must be (samples, features).")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values.")
    if X_train.shape[0] != y_train.shape[0] or X_val.shape[0] != y_val.shape[0]:
        raise ValueError("feature and target sample counts differ.")
    if X_train.shape[2] != y_train.shape[1]:
        raise ValueError("input feature count must equal target feature count.")
    if X_train.shape[2] != X_val.shape[2]:
        raise ValueError("train and validation feature counts differ.")


def _evaluate(model, X: np.ndarray, y: np.ndarray, batch_size: int, device: str, delta: float) -> Dict[str, float]:
    import torch

    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start : start + batch_size]).to(device)
            predictions.append(model(batch).cpu().numpy())
    stacked = np.vstack(predictions)
    return error_metrics(y, stacked, delta=delta)


def train_experiment(
    config: ExperimentConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_dir: Path,
    feature_names: Sequence[str] | None = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Train one variant and write its own artifacts. Never touches other runs."""
    config.validate()
    _shapes(X_train, y_train, X_val, y_val)

    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "best_model.pt"
    if checkpoint_path.exists() and not overwrite:
        raise FileExistsError(
            f"{checkpoint_path} already exists. Experiments never overwrite checkpoints; "
            "choose a new output directory or pass overwrite=True explicitly."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config.seed)
    import torch
    from torch import nn

    device = _resolve_device(config.device)
    model = build_world_model(
        variant=config.variant,
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.HuberLoss(delta=config.huber_delta, reduction="mean")

    steps_per_epoch = max(1, int(np.ceil(len(X_train) / config.batch_size)))
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(total_steps * config.warmup_fraction)

    def learning_rate_at(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return config.learning_rate * (step + 1) / warmup_steps
        if config.scheduler == "none":
            return config.learning_rate
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return config.learning_rate * 0.5 * (1 + np.cos(np.pi * min(1.0, progress)))

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)

    baseline_validation = error_metrics(y_val, persistence_forecast(X_val), delta=config.huber_delta)
    best_huber = float("inf")
    best_epoch = 0
    history: list[Dict[str, float]] = []
    step = 0
    started = time.perf_counter()
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        permutation = torch.randperm(len(X_train_t))
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(X_train_t), config.batch_size):
            indices = permutation[start : start + config.batch_size]
            batch_x = X_train_t[indices].to(device)
            batch_y = y_train_t[indices].to(device)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_at(step)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            epoch_loss += float(loss.detach())
            batches += 1
            step += 1

        validation = _evaluate(
            model, X_val, y_val, config.batch_size, device, config.huber_delta
        )
        entry = {
            "epoch": epoch,
            "train_huber": epoch_loss / max(batches, 1),
            "val_huber": validation["huber"],
            "val_mae": validation["mae"],
            "val_rmse": validation["rmse"],
            "learning_rate": learning_rate_at(step),
        }
        history.append(entry)

        if validation["huber"] < best_huber - 1e-9:
            best_huber = validation["huber"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_features": config.input_size,
                    "hidden_size": config.hidden_size,
                    "num_layers": config.num_layers,
                    "dropout": config.dropout,
                    "sequence_length": config.sequence_length,
                    "feature_names": list(feature_names) if feature_names is not None else [
                        f"feature_{index}" for index in range(config.input_size)
                    ],
                    "epoch": epoch,
                    "variant": config.variant,
                    "config": asdict(config),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    duration = time.perf_counter() - started
    final_validation = _evaluate(model, X_val, y_val, config.batch_size, device, config.huber_delta)

    report: Dict[str, Any] = {
        "experiment": config.name,
        "variant": config.variant,
        "variant_description": VARIANT_DESCRIPTIONS[config.variant],
        "config": asdict(config),
        "hardware": _hardware(),
        "device_used": device,
        "duration_seconds": round(duration, 3),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_huber": best_huber,
        "final_validation": final_validation,
        "persistence_validation": baseline_validation,
        "relative_improvement_percent": {
            "mae": 100.0 * (baseline_validation["mae"] - final_validation["mae"]) / baseline_validation["mae"],
            "rmse": 100.0 * (baseline_validation["rmse"] - final_validation["rmse"]) / baseline_validation["rmse"],
            "huber": 100.0 * (baseline_validation["huber"] - final_validation["huber"]) / baseline_validation["huber"],
        },
        "history": history,
        "checkpoint": str(checkpoint_path),
        "checkpoint_written": checkpoint_path.exists(),
        "test_partition_used": False,
        "notes": [
            "Validation-only comparison; the test partition is not read by this experiment.",
            "A checkpoint is written to this experiment's own directory and never replaces a preserved run.",
        ],
    }
    (output_dir / "experiment_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def recommend_promotion(
    candidate: Dict[str, Any],
    control: Dict[str, Any],
    min_mae_gain: float = PROMOTION_MIN_MAE_GAIN,
    min_rmse_gain: float = PROMOTION_MIN_RMSE_GAIN,
) -> Dict[str, Any]:
    """Decide whether a candidate beats the control on validation data.

    Inputs are experiment reports containing ``final_validation`` blocks. The
    recommendation is conservative: both MAE and RMSE must improve by at least
    the configured relative margin, and the candidate must have a checkpoint.
    """
    reasons: list[str] = []

    def metric(report: Dict[str, Any], key: str) -> float | None:
        block = report.get("final_validation") or report.get("validation") or {}
        value = block.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    candidate_mae, control_mae = metric(candidate, "mae"), metric(control, "mae")
    candidate_rmse, control_rmse = metric(candidate, "rmse"), metric(control, "rmse")

    if not candidate.get("checkpoint_written", False):
        reasons.append("Candidate has no written checkpoint.")
    if None in (candidate_mae, control_mae, candidate_rmse, control_rmse):
        reasons.append("MAE/RMSE missing from one or both reports.")
        return {"promote": False, "reasons": reasons}

    mae_gain = (control_mae - candidate_mae) / max(control_mae, 1e-12)
    rmse_gain = (control_rmse - candidate_rmse) / max(control_rmse, 1e-12)
    if mae_gain < min_mae_gain:
        reasons.append(
            f"MAE gain {mae_gain:.4%} is below the {min_mae_gain:.2%} promotion margin."
        )
    if rmse_gain < min_rmse_gain:
        reasons.append(
            f"RMSE gain {rmse_gain:.4%} is below the {min_rmse_gain:.2%} promotion margin."
        )
    if candidate.get("test_partition_used"):
        reasons.append("Candidate used the test partition; it cannot be promoted on that basis.")

    promote = not reasons
    if promote:
        reasons.append(
            "Both MAE and RMSE improved on validation beyond the promotion margin. "
            "Re-run the frozen test evaluation before any production claim."
        )
    return {
        "promote": promote,
        "mae_gain": mae_gain,
        "rmse_gain": rmse_gain,
        "reasons": reasons,
    }
