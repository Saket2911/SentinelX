import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sentinelx.inference import load_world_model
from sentinelx.evaluation import evaluate_one_step


DATA_DIR = Path("Datasets")
CHECKPOINT = Path(
    "Models/lstm_world_model_v3_gpu_run2/best_model.pt"
)
OUTPUT_DIR = Path("Reports/v3_reliability_audit_v1")

EXPECTED_HASH = (
    "19e117e3cc407fe656aa3e6840003031371419a2a367c31128a8b720a900450c"
)

BATCH_SIZE = 1024
TOP_N = 100


def grouped_summary(frame, group_column):
    return (
        frame.groupby(group_column, dropna=False)
        .agg(
            samples=("sample_mae", "size"),
            mean_mae=("sample_mae", "mean"),
            median_mae=("sample_mae", "median"),
            p95_mae=("sample_mae", lambda x: x.quantile(0.95)),
            p99_mae=("sample_mae", lambda x: x.quantile(0.99)),
            mean_rmse=("sample_rmse", "mean"),
            median_rmse=("sample_rmse", "median"),
            p95_rmse=("sample_rmse", lambda x: x.quantile(0.95)),
            p99_max_error=(
                "max_absolute_error",
                lambda x: x.quantile(0.99),
            ),
            maximum_absolute_error=("max_absolute_error", "max"),
        )
        .reset_index()
    )


def main():
    print("=" * 72)
    print("SENTINELX V3 — RARE-ERROR RELIABILITY AUDIT")
    print("=" * 72)

    # Prevent overwriting earlier audit results.
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output directory already contains files: {OUTPUT_DIR}"
        )

    # Verify the exact checkpoint.
    checkpoint_hash = hashlib.sha256(
        CHECKPOINT.read_bytes()
    ).hexdigest()

    if checkpoint_hash != EXPECTED_HASH:
        raise RuntimeError(
            "Checkpoint SHA256 does not match the verified V3 checkpoint."
        )

    print("PASS | Checkpoint SHA256")

    # Load the V3 model.
    model, model_metadata = load_world_model(
        CHECKPOINT,
        device="cpu",
    )

    feature_names = model_metadata["feature_names"]

    assert model_metadata["input_features"] == 209
    assert model_metadata["sequence_length"] == 10
    assert len(feature_names) == 209
    assert len(set(feature_names)) == 209

    print("PASS | V3 model loaded")
    print("Checkpoint epoch:", model_metadata["epoch"])
    print("Features:", len(feature_names))
    print("Device:", model_metadata["device"])

    # Load existing test sequences and targets.
    X = np.load(
        DATA_DIR / "sentinelx_test_X_seq10_v3.npy",
        mmap_mode="r",
    )

    y = np.load(
        DATA_DIR / "sentinelx_test_y_seq10_v3.npy",
        mmap_mode="r",
    )

    metadata = pd.read_csv(
        DATA_DIR / "sentinelx_test_metadata_seq10_v3.csv"
    )

    assert X.shape == (31406, 10, 209)
    assert y.shape == (31406, 209)
    assert len(metadata) == 31406

    required_columns = {
        "Target Timestamp",
        "Target Label",
        "Segment ID",
    }

    if not required_columns.issubset(metadata.columns):
        raise ValueError("Test metadata is missing required columns.")

    timestamps = pd.to_datetime(
        metadata["Target Timestamp"],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise ValueError("Invalid target timestamps found.")

    if metadata["Segment ID"].isna().any():
        raise ValueError("Missing Segment IDs found.")

    print("PASS | Test arrays and metadata alignment")

    # Generate predictions using the existing evaluator.
    evaluation = evaluate_one_step(
        model=model,
        X=X,
        y=y,
        metadata=metadata,
        feature_names=feature_names,
        batch_size=BATCH_SIZE,
    )

    predictions = evaluation["predictions"]
    baseline_predictions = evaluation["baseline_predictions"]

    assert predictions.shape == (31406, 209)
    assert baseline_predictions.shape == (31406, 209)

    print("PASS | One-step predictions")

    # Calculate per-sample errors.
    targets = np.asarray(y, dtype=np.float64)

    errors = predictions.astype(np.float64) - targets
    baseline_errors = (
        baseline_predictions.astype(np.float64) - targets
    )

    absolute_errors = np.abs(errors)
    baseline_absolute_errors = np.abs(baseline_errors)

    sample_mae = absolute_errors.mean(axis=1)
    sample_rmse = np.sqrt(np.mean(errors ** 2, axis=1))

    baseline_mae = baseline_absolute_errors.mean(axis=1)
    baseline_rmse = np.sqrt(
        np.mean(baseline_errors ** 2, axis=1)
    )

    max_indices = np.argmax(absolute_errors, axis=1)
    max_absolute_error = absolute_errors.max(axis=1)

    largest_feature = np.asarray(feature_names)[max_indices]

    largest_signed_error = errors[
        np.arange(len(errors)),
        max_indices,
    ]

    improvement = np.divide(
        baseline_mae - sample_mae,
        baseline_mae,
        out=np.full_like(sample_mae, np.nan),
        where=baseline_mae != 0,
    ) * 100.0

    # Build sample-level table.
    results = pd.DataFrame({
        "target_timestamp": timestamps,
        "target_label": (
            metadata["Target Label"]
            .fillna("UNKNOWN")
            .astype(str)
        ),
        "segment_id": metadata["Segment ID"],
        "five_minute_window": timestamps.dt.floor("5min"),
        "sample_mae": sample_mae,
        "sample_rmse": sample_rmse,
        "persistence_mae": baseline_mae,
        "persistence_rmse": baseline_rmse,
        "mae_improvement_percent": improvement,
        "max_absolute_error": max_absolute_error,
        "largest_error_feature": largest_feature,
        "largest_feature_signed_error": largest_signed_error,
    })

    assert len(results) == 31406
    assert np.isfinite(
        results[
            ["sample_mae", "sample_rmse", "max_absolute_error"]
        ].to_numpy()
    ).all()

    print("PASS | Per-sample error calculations")

    # Summaries by label, segment and five-minute window.
    label_summary = grouped_summary(
        results, "target_label"
    )

    segment_summary = grouped_summary(
        results, "segment_id"
    )

    time_summary = grouped_summary(
        results, "five_minute_window"
    )

    # Feature-level residual summary.
    feature_summary = pd.DataFrame({
        "feature": feature_names,
        "mean_absolute_error": absolute_errors.mean(axis=0),
        "median_absolute_error": np.median(
            absolute_errors, axis=0
        ),
        "p95_absolute_error": np.quantile(
            absolute_errors, 0.95, axis=0
        ),
        "p99_absolute_error": np.quantile(
            absolute_errors, 0.99, axis=0
        ),
        "maximum_absolute_error": absolute_errors.max(axis=0),
        "persistence_mean_absolute_error": (
            baseline_absolute_errors.mean(axis=0)
        ),
    })

    feature_summary["mae_improvement_percent"] = (
        (
            feature_summary["persistence_mean_absolute_error"]
            - feature_summary["mean_absolute_error"]
        )
        / feature_summary[
            "persistence_mean_absolute_error"
        ].replace(0, np.nan)
    ) * 100.0

    feature_summary = feature_summary.sort_values(
        "mean_absolute_error",
        ascending=False,
    ).reset_index(drop=True)

    # Highest-error samples.
    worst_mae = results.nlargest(TOP_N, "sample_mae")
    worst_max_error = results.nlargest(
        TOP_N, "max_absolute_error"
    )

    # Overall quantiles.
    quantiles = [0.50, 0.90, 0.95, 0.99, 0.999]

    distributions = {
        "sample_mae": {
            str(q): float(np.quantile(sample_mae, q))
            for q in quantiles
        },
        "sample_rmse": {
            str(q): float(np.quantile(sample_rmse, q))
            for q in quantiles
        },
        "maximum_absolute_error": {
            str(q): float(np.quantile(max_absolute_error, q))
            for q in quantiles
        },
    }

    # Save artifacts.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        OUTPUT_DIR / "sample_level_errors.csv",
        index=False,
    )

    label_summary.to_csv(
        OUTPUT_DIR / "label_error_summary.csv",
        index=False,
    )

    segment_summary.to_csv(
        OUTPUT_DIR / "segment_error_summary.csv",
        index=False,
    )

    time_summary.to_csv(
        OUTPUT_DIR / "five_minute_error_summary.csv",
        index=False,
    )

    feature_summary.to_csv(
        OUTPUT_DIR / "feature_error_summary.csv",
        index=False,
    )

    worst_mae.to_csv(
        OUTPUT_DIR / "worst_samples_by_mae.csv",
        index=False,
    )

    worst_max_error.to_csv(
        OUTPUT_DIR / "worst_samples_by_max_error.csv",
        index=False,
    )

    summary = {
        "analysis": "V3 one-step rare-error reliability audit",
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": int(model_metadata["epoch"]),
        "samples": len(results),
        "features": len(feature_names),
        "sequence_length": int(model_metadata["sequence_length"]),
        "segments": int(results["segment_id"].nunique()),
        "five_minute_windows": len(time_summary),
        "label_counts": {
            str(k): int(v)
            for k, v in results["target_label"].value_counts().items()
        },
        "overall_error_quantiles": distributions,
        "metrics_space": "Standardized preprocessed model feature space",
        "interpretation": (
            "Descriptive prediction-error analysis only. "
            "Quantiles are not attack-detection thresholds."
        ),
        "test_partition_note": (
            "This partition was previously used for exploratory "
            "evaluation and is not an untouched independent test."
        ),
    }

    (OUTPUT_DIR / "reliability_summary.json").write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("AUDIT SUMMARY")
    print("=" * 72)

    print("Samples:", len(results))
    print("Segments:", summary["segments"])
    print("Five-minute windows:", len(time_summary))

    print("\nSample MAE quantiles:")
    for q, value in distributions["sample_mae"].items():
        print(f"P{float(q) * 100:g}: {value:.6f}")

    print("\nSample RMSE quantiles:")
    for q, value in distributions["sample_rmse"].items():
        print(f"P{float(q) * 100:g}: {value:.6f}")

    print("\nMaximum absolute error quantiles:")
    for q, value in distributions[
        "maximum_absolute_error"
    ].items():
        print(f"P{float(q) * 100:g}: {value:.6f}")

    print("\nTop five features by mean absolute error:")
    print(
        feature_summary[
            ["feature", "mean_absolute_error"]
        ].head(5).to_string(index=False)
    )

    print("\nArtifacts saved to:", OUTPUT_DIR)
    print("V3 RARE-ERROR RELIABILITY AUDIT: COMPLETE")


if __name__ == "__main__":
    main()
