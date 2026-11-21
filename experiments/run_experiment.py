#!/usr/bin/env python3
"""Run a controlled SentinelX world-model experiment.

This script never rewrites a preserved run. Every experiment writes its own
directory under ``Models/experiments/`` containing ``best_model.pt`` (ignored by
git) and ``experiment_report.json``.

Expected arrays (NumPy, float32/float64):
    X_train  (samples, sequence_length, features)
    y_train  (samples, features)
    X_val    (samples, sequence_length, features)
    y_val    (samples, features)

Conventional filenames under the dataset directory:
    sentinelx_train_sequences_v3.npy
    sentinelx_train_targets_v3.npy
    sentinelx_validation_sequences_v3.npy
    sentinelx_validation_targets_v3.npy

The test partition is deliberately not accepted: experiments are promoted on
validation evidence, and the frozen test evaluation is re-run separately.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentinelx.models.variants import VARIANT_DESCRIPTIONS, VARIANT_NAMES  # noqa: E402
from sentinelx.training.experiments import (  # noqa: E402
    ExperimentConfig,
    recommend_promotion,
    train_experiment,
)

CONVENTIONAL = {
    "X_train": "sentinelx_train_sequences_v3.npy",
    "y_train": "sentinelx_train_targets_v3.npy",
    "X_val": "sentinelx_validation_sequences_v3.npy",
    "y_val": "sentinelx_validation_targets_v3.npy",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=VARIANT_NAMES, default="baseline_v3")
    parser.add_argument("--name", default=None, help="Experiment label (defaults to variant + timestamp).")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "Datasets")
    parser.add_argument("--train-x", type=Path, default=None)
    parser.add_argument("--train-y", type=Path, default=None)
    parser.add_argument("--val-x", type=Path, default=None)
    parser.add_argument("--val-y", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "Models" / "experiments")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=("cosine", "none"), default="cosine")
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compare", type=Path, default=None, help="Control experiment_report.json to compare against.")
    parser.add_argument("--feature-manifest", type=Path, default=None, help="JSON list of feature names.")
    return parser.parse_args(argv)


def resolve_arrays(args: argparse.Namespace) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for key, filename in CONVENTIONAL.items():
        explicit = getattr(args, key.lower().replace("x_", "x_").replace("y_", "y_"), None)
        path = explicit or (args.data_dir / filename)
        resolved[key] = Path(path)
    return resolved


def fail(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    arrays = resolve_arrays(args)
    missing = [str(path) for path in arrays.values() if not path.is_file()]
    if missing:
        fail(
            "missing sequence arrays:\n  "
            + "\n  ".join(missing)
            + "\n\nGenerate them with scripts/pipeline and scripts/training, or pass explicit "
            "--train-x/--train-y/--val-x/--val-y paths. No data is synthesized."
        )

    feature_names = None
    if args.feature_manifest:
        try:
            feature_names = json.loads(Path(args.feature_manifest).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            fail(f"cannot read feature manifest: {exc}")
    else:
        conventional = args.data_dir / "sentinelx_model_features_v3.json"
        if conventional.is_file():
            try:
                feature_names = json.loads(conventional.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                feature_names = None

    X_train = np.load(arrays["X_train"])
    y_train = np.load(arrays["y_train"])
    X_val = np.load(arrays["X_val"])
    y_val = np.load(arrays["y_val"])

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = args.name or f"{args.variant}-{timestamp}"
    output_dir = Path(args.output_dir) / f"{timestamp}_{args.variant}"

    config = ExperimentConfig(
        name=name,
        variant=args.variant,
        input_size=int(X_train.shape[2]),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        scheduler=args.scheduler,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
    )

    print(f"SentinelX experiment: {name}")
    print(f"  variant    : {args.variant} — {VARIANT_DESCRIPTIONS[args.variant]}")
    print(f"  train/val  : {X_train.shape[0]:,} / {X_val.shape[0]:,} sequences")
    print(f"  output     : {output_dir}")
    print("  note       : validation-only comparison; test partition untouched")

    report = train_experiment(
        config,
        X_train,
        y_train,
        X_val,
        y_val,
        output_dir,
        feature_names=feature_names,
    )

    print("\nValidation results")
    for key in ("mae", "rmse", "huber"):
        candidate = report["final_validation"][key]
        baseline = report["persistence_validation"][key]
        gain = report["relative_improvement_percent"][key]
        print(f"  {key:<5} model={candidate:.6f}  persistence={baseline:.6f}  gain={gain:+.2f}%")
    print(f"  best epoch {report['best_epoch']} · {report['duration_seconds']:.1f}s · {report['device_used']}")
    print(f"  report: {output_dir / 'experiment_report.json'}")

    if args.compare:
        try:
            control = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            fail(f"cannot read control report: {exc}")
        decision = recommend_promotion(report, control)
        print("\nPromotion recommendation (validation only)")
        print(f"  promote: {decision['promote']}")
        for reason in decision["reasons"]:
            print(f"  - {reason}")
        (output_dir / "promotion_decision.json").write_text(
            json.dumps(decision, indent=2), encoding="utf-8"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
