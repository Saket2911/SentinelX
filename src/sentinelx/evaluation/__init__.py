"""Evaluation helpers.

Metrics and multi-step utilities are NumPy-only and always importable. The
torch-dependent one-step evaluator is imported lazily so analysis tooling and
unit tests work in environments without PyTorch.
"""

__all__ = [
    "calculate_metrics",
    "calculate_improvements",
    "evaluate_one_step",
    "evaluate_multi_step",
]


def __getattr__(name):
    if name in {"calculate_metrics", "calculate_improvements"}:
        from sentinelx.evaluation.metrics import (
            calculate_improvements,
            calculate_metrics,
        )

        return {
            "calculate_metrics": calculate_metrics,
            "calculate_improvements": calculate_improvements,
        }[name]

    if name == "evaluate_one_step":
        from sentinelx.evaluation.evaluator import evaluate_one_step

        return evaluate_one_step

    if name == "evaluate_multi_step":
        from sentinelx.evaluation.multi_step import evaluate_multi_step

        return evaluate_multi_step

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
