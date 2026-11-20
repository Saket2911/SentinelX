"""Central configuration for the SentinelX backend.

All paths are resolved relative to the repository root so the service can be
started from anywhere. Nothing here is hardcoded to a machine-specific path.
"""

from __future__ import annotations

import os
from pathlib import Path

# backend/app/config.py -> backend/app -> backend -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = REPO_ROOT / "Models"
REPORTS_DIR = REPO_ROOT / "reports"
DATASETS_DIR = REPO_ROOT / "Datasets"

# Writable runtime state (model-activation pointer, user settings, generated
# reports). Kept out of the research artifact tree on purpose.
VAR_DIR = Path(os.environ.get("SENTINELX_VAR_DIR", REPO_ROOT / "backend" / "var"))

SETTINGS_FILE = VAR_DIR / "settings.json"
ACTIVE_MODEL_FILE = VAR_DIR / "active_model.json"
GENERATED_REPORTS_DIR = VAR_DIR / "reports"

# Model versions this backend knows how to describe. `model_dir` is the folder
# under Models/ that holds the committed metadata artifacts.
MODEL_VERSIONS: dict[str, dict] = {
    "v1": {
        "id": "v1",
        "name": "SentinelXLSTM V1",
        "model_dir": "lstm_world_model_v1",
        "checkpoint": "best_model.pt",
        "input_features": 206,
        "sequence_length": 10,
        "epochs_best": 2,
        "status": "historical",
        "note": (
            "206-feature baseline. Superseded by later feature definitions; "
            "kept for provenance."
        ),
    },
    "v2": {
        "id": "v2",
        "name": "SentinelXLSTM V2",
        "model_dir": "lstm_world_model_v2",
        "checkpoint": "best_model.pt",
        "input_features": 212,
        "sequence_length": 10,
        "epochs_best": 8,
        "status": "production_candidate",
        "note": (
            "212-feature expanded pipeline. Final test evaluation recorded "
            "against the persistence baseline."
        ),
    },
    "v3": {
        "id": "v3",
        "name": "SentinelXLSTM V3",
        "model_dir": "lstm_world_model_v3_gpu_run2",
        "checkpoint": "best_model.pt",
        "input_features": 209,
        "sequence_length": 10,
        "epochs_best": 11,
        "status": "experimental",
        "note": (
            "209-feature ablation pipeline with the deepest analysis coverage. "
            "Not interchangeable with V2: different feature definition, scaler "
            "and sequence arrays."
        ),
    },
}

# Default active model. V3 is the version with the most recent recorded
# evaluation coverage; this is a reporting choice, not a superiority claim.
DEFAULT_ACTIVE_MODEL = "v3"

# Frozen anomaly-review threshold recorded in the V3 validation analysis.
RECORDED_ANOMALY_THRESHOLD = 1.881396770477295
RECORDED_THRESHOLD_PERCENTILE = 99.0
