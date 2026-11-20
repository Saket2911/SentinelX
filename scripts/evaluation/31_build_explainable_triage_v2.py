from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path("/mnt/f/Projects/SentinelX")
DATA = ROOT / "Datasets"
MODEL = ROOT / "Models/lstm_world_model_v3_gpu_run2"
REPLAY = MODEL / "historical_replay_v1"
OUT = MODEL / "explainable_triage_v2"

TOP_K = 5
HISTORY_WINDOW = 5

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("SENTINELX STEP 31B — EXPLAINABLE THREAT TRIAGE V2")
    print("=" * 72)

    # Load verified replay predictions and actual targets
    with np.load(REPLAY / "replay_arrays.npz", allow_pickle=False) as z:
        predictions = z["predictions"].astype(np.float32)
        actual = z["actual"].astype(np.float32)
        features = z["feature_names"].astype(str)

    metadata = pd.read_csv(
        DATA / "sentinelx_test_metadata_seq10_v3.csv"
    )

    n, feature_count = actual.shape

    if predictions.shape != actual.shape:
        raise ValueError("Prediction and actual shapes do not match.")

    if len(metadata) != n or len(features) != feature_count:
        raise ValueError("Metadata or feature count does not match replay.")

    if not np.isfinite(predictions).all() or not np.isfinite(actual).all():
        raise ValueError("Non-finite values found in replay arrays.")

    print(f"Observations: {n:,}")
    print(f"Features: {feature_count}")
    print(f"Segments: {metadata['Segment ID'].nunique()}")

    # Feature-wise signed and absolute prediction residuals
    signed_residuals = actual - predictions
    abs_residuals = np.abs(signed_residuals)

    # Overall mean absolute residual for every timestamp
    scores = abs_residuals.mean(axis=1)

    # Temporal context is calculated separately within each segment.
    previous_median = np.full(n, np.nan, dtype=np.float64)
    rolling_mean = np.full(n, np.nan, dtype=np.float64)

    segment_groups = metadata.groupby("Segment ID", sort=False).indices

    for _, indices in segment_groups.items():
        indices = np.asarray(indices, dtype=np.int64)
        segment_scores = pd.Series(scores[indices])

        previous_median[indices] = (
            segment_scores
            .shift(1)
            .rolling(HISTORY_WINDOW, min_periods=1)
            .median()
            .to_numpy()
        )

        rolling_mean[indices] = (
            segment_scores
            .rolling(HISTORY_WINDOW, min_periods=1)
            .mean()
            .to_numpy()
        )

    # Identify top five feature contributors for every timestamp.
    top_indices = np.argpartition(
        abs_residuals,
        -TOP_K,
        axis=1
    )[:, -TOP_K:]

    # Sort selected features from largest to smallest residual.
    top_errors = np.take_along_axis(
        abs_residuals, top_indices, axis=1
    )

    order = np.argsort(-top_errors, axis=1)
    top_indices = np.take_along_axis(top_indices, order, axis=1)
    top_errors = np.take_along_axis(top_errors, order, axis=1)

    row_totals = abs_residuals.sum(axis=1)
    contributions = np.divide(
        top_errors,
        row_totals[:, None],
        out=np.zeros_like(top_errors),
        where=row_totals[:, None] > 0
    ) * 100.0

    # Build the explainability table.
    output = metadata[
        [
            "Target State Row",
            "Target Timestamp",
            "Target CaptureDay",
            "Segment ID"
        ]
    ].copy()

    output["Prediction MAE (model space)"] = scores
    output["Previous 5-step Median MAE"] = previous_median
    output["5-step Rolling Mean MAE"] = rolling_mean
    output["Change vs Previous Median"] = scores - previous_median

    for rank in range(TOP_K):
        indices = top_indices[:, rank]
        names = features[indices]

        output[f"Top {rank + 1} Feature"] = names
        output[f"Top {rank + 1} Signed Residual"] = (
            signed_residuals[np.arange(n), indices]
        )
        output[f"Top {rank + 1} Absolute Error"] = top_errors[:, rank]
        output[f"Top {rank + 1} Contribution Percent"] = (
            contributions[:, rank]
        )

    # Save per-timestamp explanations.
    csv_path = OUT / "explainable_triage.csv"
    output.to_csv(csv_path, index=False)

    # Save full feature-wise residual matrices for later dashboard use.
    residual_path = OUT / "feature_residuals.npz"
    np.savez_compressed(
        residual_path,
        signed_residuals=signed_residuals,
        absolute_residuals=abs_residuals,
        feature_names=features
    )

    # Aggregate how often each feature appears among the top five.
    top_feature_names = features[top_indices.reshape(-1)]
    top_counts = (
        pd.Series(top_feature_names)
        .value_counts()
        .rename_axis("Feature")
        .reset_index(name="Top 5 Appearance Count")
    )
    top_counts["Appearance Percent"] = (
        top_counts["Top 5 Appearance Count"] / n * 100.0
    )

    top_counts.to_csv(
        OUT / "most_frequent_error_contributors.csv",
        index=False
    )

    # Global feature contribution summary.
    global_feature_mae = abs_residuals.mean(axis=0)
    global_contribution = (
        global_feature_mae / global_feature_mae.sum() * 100.0
    )

    global_summary = pd.DataFrame({
        "Feature": features,
        "Mean Absolute Residual": global_feature_mae,
        "Contribution Percent": global_contribution
    }).sort_values(
        "Mean Absolute Residual", ascending=False
    )

    global_summary.to_csv(
        OUT / "global_feature_error_summary.csv",
        index=False
    )

    summary = {
        "engine": "SentinelX Explainable Threat Triage V2",
        "observations": int(n),
        "features": int(feature_count),
        "segments": int(metadata["Segment ID"].nunique()),
        "mean_prediction_mae_model_space": float(scores.mean()),
        "median_prediction_mae_model_space": float(np.median(scores)),
        "maximum_prediction_mae_model_space": float(scores.max()),
        "top_features_per_observation": TOP_K,
        "temporal_history_window": HISTORY_WINDOW,
        "temporal_comparison": (
            "Current score compared with the preceding five observations "
            "within the same segment. No cross-segment history is used."
        ),
        "ground_truth_labels_used_for_explanations": False,
        "threshold_based_attack_decision": False,
        "interpretation": (
            "Feature residuals explain differences between predicted and "
            "actual network states. They do not establish malicious activity. "
            "Temporal comparisons provide context, not a validated detector."
        ),
        "comparison_space": "Preprocessed model feature space",
        "outputs": {
            "explanations": str(csv_path.relative_to(ROOT)),
            "feature_residuals": str(residual_path.relative_to(ROOT)),
            "frequent_contributors": str(
                (OUT / "most_frequent_error_contributors.csv").relative_to(ROOT)
            ),
            "global_feature_summary": str(
                (OUT / "global_feature_error_summary.csv").relative_to(ROOT)
            )
        }
    }

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print("EXPLAINABLE TRIAGE V2 COMPLETE")
    print("=" * 72)
    print(f"Mean prediction MAE: {scores.mean():.6f}")
    print(f"Explanation rows: {len(output):,}")

    print("\nMost frequent top-five error contributors:")
    print(top_counts.head(10).to_string(index=False))

    print("\nLargest global mean residuals:")
    print(global_summary.head(10).to_string(index=False))

    print("\nOutput folder:", OUT.relative_to(ROOT))
    print("=" * 72)

if __name__ == "__main__":
    main()
