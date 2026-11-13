from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

ROOT = Path("/mnt/f/Projects/SentinelX")
DATA = ROOT / "Datasets"
MODEL = ROOT / "Models/lstm_world_model_v3_gpu_run2"
REPLAY = MODEL / "historical_replay_v1"

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    return bool(condition)

def main():
    print("=" * 72)
    print("SENTINELX STEP 30B — FULL HISTORICAL REPLAY INTEGRITY AUDIT")
    print("=" * 72)

    results = {}

    # Load replay artifacts
    with np.load(REPLAY / "replay_arrays.npz", allow_pickle=False) as z:
        predictions = z["predictions"]
        actual = z["actual"]
        replay_features = z["feature_names"].astype(str)

    replay_csv = pd.read_csv(REPLAY / "test_replay.csv")
    metadata = pd.read_csv(DATA / "sentinelx_test_metadata_seq10_v3.csv")
    test_y = np.load(DATA / "sentinelx_test_y_seq10_v3.npy", mmap_mode="r")

    with open(REPLAY / "replay_manifest.json", "r") as f:
        manifest = json.load(f)

    checkpoint = torch.load(
        MODEL / "best_model.pt",
        map_location="cpu",
        weights_only=False
    )
    checkpoint_features = np.asarray(
        checkpoint["feature_names"], dtype=str
    )

    n = len(metadata)
    print(f"\nExpected test steps: {n:,}")
    print(f"Replay CSV rows:     {len(replay_csv):,}")
    print(f"Replay array rows:   {len(predictions):,}")
    print(f"Test target rows:    {len(test_y):,}")

    # 1. Shape and count checks
    results["prediction_shape"] = check(
        "Prediction array shape",
        predictions.shape == (n, 209),
        str(predictions.shape)
    )
    results["actual_shape"] = check(
        "Actual array shape",
        actual.shape == (n, 209),
        str(actual.shape)
    )
    results["test_y_shape"] = check(
        "Original test target shape",
        test_y.shape == (n, 209),
        str(test_y.shape)
    )
    results["csv_count"] = check(
        "Replay CSV row count",
        len(replay_csv) == n,
        f"{len(replay_csv):,} rows"
    )
    results["manifest_count"] = check(
        "Manifest step count",
        manifest.get("number_of_steps") == n,
        str(manifest.get("number_of_steps"))
    )

    # 2. Feature-order checks
    results["feature_count"] = check(
        "Feature count",
        len(replay_features) == 209
    )
    results["feature_order"] = check(
        "Replay feature order matches checkpoint",
        np.array_equal(replay_features, checkpoint_features)
    )

    # 3. Numerical validity
    results["predictions_finite"] = check(
        "All predictions are finite",
        bool(np.isfinite(predictions).all())
    )
    results["actual_finite"] = check(
        "All actual values are finite",
        bool(np.isfinite(actual).all())
    )

    # 4. Actual target alignment against original test y
    actual_diff = np.abs(
        actual.astype(np.float64) -
        np.asarray(test_y, dtype=np.float64)
    )
    max_actual_diff = float(actual_diff.max())
    results["actual_matches_test_y"] = check(
        "Replay actual values match original test targets",
        max_actual_diff == 0.0,
        f"maximum absolute difference={max_actual_diff:.10g}"
    )

    # 5. Metadata alignment
    metadata_columns = [
        "Target State Row",
        "Target Timestamp",
        "Target CaptureDay",
        "Target Label",
        "Segment ID"
    ]
    metadata_matches = True
    for col in metadata_columns:
        if col not in replay_csv.columns:
            metadata_matches = False
            print(f"[FAIL] Replay CSV missing metadata column: {col}")
            continue

        left = metadata[col].astype(str).fillna("")
        right = replay_csv[col].astype(str).fillna("")
        same = left.equals(right)
        metadata_matches = metadata_matches and same
        check(f"Metadata alignment: {col}", same)

    results["metadata_alignment"] = metadata_matches

    # 6. Replay step ordering
    expected_steps = np.arange(1, n + 1)
    steps_ok = (
        "Replay Step" in replay_csv.columns
        and np.array_equal(
            replay_csv["Replay Step"].to_numpy(),
            expected_steps
        )
    )
    results["step_order"] = check(
        "Replay steps are sequential",
        steps_ok
    )

    timestamps = pd.to_datetime(
        metadata["Target Timestamp"], errors="coerce"
    )
    results["timestamps_valid"] = check(
        "All target timestamps parse correctly",
        bool(timestamps.notna().all())
    )

    chronological = True
    for _, indices in metadata.groupby("Segment ID", sort=False).groups.items():
        segment_times = timestamps.iloc[list(indices)]
        if not segment_times.is_monotonic_increasing:
            chronological = False
            break

    results["chronological"] = check(
        "Timestamps are chronological within each segment",
        chronological
    )

    # 7. Recompute per-step prediction MAE
    recomputed_mae = np.mean(
        np.abs(
            predictions.astype(np.float64) -
            actual.astype(np.float64)
        ),
        axis=1
    )
    saved_mae = replay_csv[
        "Prediction MAE (model space)"
    ].to_numpy(dtype=np.float64)

    mae_max_diff = float(np.max(np.abs(recomputed_mae - saved_mae)))
    results["step_mae"] = check(
        "CSV per-step MAE matches replay arrays",
        mae_max_diff < 1e-6,
        f"maximum difference={mae_max_diff:.10g}"
    )

    overall_mae = float(recomputed_mae.mean())
    results["overall_mae"] = check(
        "Overall replay MAE matches manifest",
        abs(overall_mae - float(manifest["mean_test_mae"])) < 1e-6,
        f"recomputed={overall_mae:.9f}, manifest={manifest['mean_test_mae']:.9f}"
    )

    # 8. Selected-feature CSV values
    selected = manifest.get("selected_features", [])
    selected_ok = True

    for feature in selected:
        if feature not in replay_features:
            selected_ok = False
            print(f"[FAIL] Selected feature missing: {feature}")
            continue

        idx = int(np.where(replay_features == feature)[0][0])
        checks = {
            f"Actual | {feature}": actual[:, idx],
            f"Predicted | {feature}": predictions[:, idx],
            f"Absolute Error | {feature}":
                np.abs(actual[:, idx] - predictions[:, idx])
        }

        for col, expected in checks.items():
            if col not in replay_csv.columns:
                selected_ok = False
                print(f"[FAIL] Missing CSV column: {col}")
                continue

            saved = replay_csv[col].to_numpy(dtype=np.float64)
            diff = float(np.max(np.abs(saved - expected)))
            ok = diff < 1e-5
            selected_ok = selected_ok and ok
            check(
                f"Selected feature CSV: {col}",
                ok,
                f"maximum difference={diff:.10g}"
            )

    results["selected_feature_csv"] = selected_ok

    # 9. Replay behavior metadata
    results["one_step_ahead"] = check(
        "Prediction horizon is one next state",
        manifest.get("prediction_horizon") == "one next state"
    )
    results["no_prediction_feedback"] = check(
        "Predictions are not fed back as inputs",
        manifest.get("predictions_fed_back_as_inputs") is False
    )

    # Save audit report
    passed = sum(v is True for v in results.values())
    failed = sum(v is False for v in results.values())

    report = {
        "audit": "SentinelX Historical Replay Integrity Audit",
        "steps": int(n),
        "features": int(predictions.shape[1]),
        "overall_mean_mae_model_space": overall_mae,
        "maximum_actual_target_difference": max_actual_diff,
        "maximum_step_mae_difference": mae_max_diff,
        "checks_passed": passed,
        "checks_failed": failed,
        "results": results
    }

    report_path = REPLAY / "replay_integrity_audit.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 72)
    print(f"AUDIT RESULT: {passed} PASSED / {failed} FAILED")
    print("Report:", report_path.relative_to(ROOT))
    print("=" * 72)

    if failed:
        raise SystemExit("Audit found failures. Do not proceed until investigated.")

if __name__ == "__main__":
    main()
