"""On-demand world-model inference.

Inference is deliberately opt-in and strictly validated: it needs the run's
checkpoint, a torch runtime and (for raw states) the frozen preprocessing
artifacts. When anything is missing the endpoint reports why instead of
returning placeholder predictions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ..config import Settings
from .artifacts import ArtifactStore

REQUIRED_FEATURES = 209
REQUIRED_SEQUENCE = 10


def runtime_status(settings: Settings, store: ArtifactStore, run_id: str) -> dict[str, Any]:
    run = store.run(run_id)
    if run is None:
        raise KeyError(run_id)
    import importlib.util

    torch_available = importlib.util.find_spec("torch") is not None
    sklearn_available = importlib.util.find_spec("sklearn") is not None
    joblib_available = importlib.util.find_spec("joblib") is not None
    preprocessing = {
        "features": (settings.datasets_dir / "sentinelx_model_features_v3.json").is_file(),
        "medians": (settings.datasets_dir / "sentinelx_train_medians_v3.npz").is_file(),
        "scaler": (settings.datasets_dir / "sentinelx_standard_scaler_v3.joblib").is_file(),
    }
    checkpoint = run.checkpoints[0] if run.checkpoints else None
    reasons: list[str] = []
    if checkpoint is None:
        reasons.append("No .pt checkpoint is present for this run in this checkout.")
    if not torch_available:
        reasons.append("PyTorch is not installed in this runtime.")
    if not all(preprocessing.values()):
        missing = [name for name, present in preprocessing.items() if not present]
        reasons.append(
            "Frozen preprocessing artifacts are missing ("
            + ", ".join(missing)
            + "); raw-state inference is unavailable."
        )
    return {
        "available": not reasons[:1],
        "model_space_available": checkpoint is not None and torch_available,
        "raw_state_available": checkpoint is not None and torch_available and all(preprocessing.values())
        and sklearn_available
        and joblib_available,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "torch_available": torch_available,
        "sklearn_available": sklearn_available,
        "preprocessing": preprocessing,
        "reasons": reasons,
    }


def _load_model(settings: Settings, store: ArtifactStore, run_id: str):
    run = store.run(run_id)
    if run is None:
        raise KeyError(run_id)
    status = runtime_status(settings, store, run_id)
    if not status["model_space_available"]:
        raise RuntimeError(" ".join(status["reasons"]) or "Inference runtime is unavailable.")
    src = settings.root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from sentinelx.inference.world_model import load_world_model  # type: ignore

    model, metadata = load_world_model(run.checkpoints[0], device="cpu")
    return model, metadata


def predict_model_space(
    settings: Settings,
    store: ArtifactStore,
    run_id: str,
    sequence: list[list[float]],
) -> dict[str, Any]:
    if len(sequence) != REQUIRED_SEQUENCE:
        raise ValueError(f"Expected exactly {REQUIRED_SEQUENCE} states, received {len(sequence)}.")
    for index, row in enumerate(sequence):
        if len(row) != REQUIRED_FEATURES:
            raise ValueError(
                f"State {index} has {len(row)} features; the V3 model requires {REQUIRED_FEATURES}."
            )
        for value in row:
            if value is None or value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"State {index} contains a non-finite value.")

    model, metadata = _load_model(settings, store, run_id)
    import numpy as np

    from sentinelx.inference.world_model import predict_next_state  # type: ignore

    array = np.asarray(sequence, dtype=np.float32)
    prediction = predict_next_state(model, array)
    return {
        "run_id": run_id,
        "device": metadata.get("device"),
        "sequence_length": metadata.get("sequence_length"),
        "features": metadata.get("input_features"),
        "space": "Preprocessed model feature space",
        "prediction": [float(value) for value in prediction],
        "interpretation": (
            "Predicted next one-second network state in standardized model space. This is a "
            "forecast, not an attack classification."
        ),
    }


def predict_raw_state(
    settings: Settings,
    store: ArtifactStore,
    run_id: str,
    states: list[dict[str, Any]],
) -> dict[str, Any]:
    status = runtime_status(settings, store, run_id)
    if not status["raw_state_available"]:
        raise RuntimeError(" ".join(status["reasons"]) or "Raw-state inference is unavailable.")
    if len(states) != REQUIRED_SEQUENCE:
        raise ValueError(f"Expected exactly {REQUIRED_SEQUENCE} states, received {len(states)}.")
    import pandas as pd

    src = settings.root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from sentinelx.features.preprocessor_v3 import SentinelXV3Preprocessor  # type: ignore
    from sentinelx.inference.world_model import predict_next_state  # type: ignore

    model, metadata = _load_model(settings, store, run_id)
    preprocessor = SentinelXV3Preprocessor.load(settings.datasets_dir)
    frame = pd.DataFrame(states)
    expected = ["Timestamp", "Segment ID"]
    for column in expected:
        if column not in frame.columns:
            raise ValueError(f"Each raw state needs a {column!r} column.")
    processed = preprocessor.transform_features(frame)
    prediction = predict_next_state(model, processed)
    return {
        "run_id": run_id,
        "device": metadata.get("device"),
        "features": metadata.get("input_features"),
        "space": "Preprocessed model feature space",
        "prediction": [float(value) for value in prediction],
        "feature_names": preprocessor.feature_names,
        "interpretation": "Predicted next network state after frozen V3 preprocessing.",
    }
