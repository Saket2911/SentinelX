"""Public inference API for SentinelX.

The heavy runtime dependencies used by the forecasting pipeline (notably PyTorch)
are optional at import time. Keep the package importable for lightweight tooling,
streaming logic, and tests while deferring the actual imports until the specific
feature is used.
"""

__all__ = [
    "load_world_model",
    "predict_next_state",
    "SentinelXV3Predictor",
    "SentinelXStreamingPredictor",
]


def __getattr__(name):
    if name in {"load_world_model", "predict_next_state"}:
        from sentinelx.inference.world_model import (
            load_world_model,
            predict_next_state,
        )
        return {
            "load_world_model": load_world_model,
            "predict_next_state": predict_next_state,
        }[name]

    if name == "SentinelXV3Predictor":
        from sentinelx.inference.pipeline import SentinelXV3Predictor

        return SentinelXV3Predictor

    if name == "SentinelXStreamingPredictor":
        from sentinelx.inference.streaming import SentinelXStreamingPredictor

        return SentinelXStreamingPredictor

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
