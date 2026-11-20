import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sentinelx.inference import (
    SentinelXV3Predictor,
    SentinelXStreamingPredictor,
)


DATA = Path("Datasets")
CHECKPOINT = Path(
    "Models/lstm_world_model_v3_gpu_run2/best_model.pt"
)
OUTPUT_DIR = Path(
    "reports/streaming_direct_integration_v1"
)

EXPECTED_SHA256 = (
    "19e117e3cc407fe656aa3e6840003031371419a2a367c31128a8b720a900450c"
)

ROWS = 100
TOLERANCE = 1e-6


def main():
    print("=" * 72)
    print("SENTINELX V3 — STREAMING/DIRECT INFERENCE INTEGRATION")
    print("=" * 72)

    report_path = OUTPUT_DIR / "integration_summary.json"

    if report_path.exists():
        raise FileExistsError(
            f"Report already exists: {report_path}"
        )

    # 1. Check checkpoint integrity.
    checkpoint_hash = hashlib.sha256(
        CHECKPOINT.read_bytes()
    ).hexdigest()

    assert checkpoint_hash == EXPECTED_SHA256
    print("PASS | Checkpoint integrity")

    # 2. Load the same real predictor used by streaming.
    direct_predictor = SentinelXV3Predictor.load(
        checkpoint_path=CHECKPOINT,
        data_dir=DATA,
        device="cpu",
    )

    stream = SentinelXStreamingPredictor(
        direct_predictor
    )

    print("PASS | Real V3 predictor and streaming initialization")

    # 3. Load 100 consecutive real network states.
    data = pd.read_csv(
        DATA / "sentinelx_expanded_train_v1.csv",
        nrows=ROWS,
    )

    assert len(data) == ROWS
    assert data["Segment ID"].nunique() == 1

    timestamps = pd.to_datetime(data["Timestamp"])

    assert timestamps.diff().dropna().eq(
        pd.Timedelta(seconds=1)
    ).all()

    print("PASS | 100 consecutive real network states")

    # 4. Compare streaming predictions with direct inference.
    predictions_checked = 0
    maximum_absolute_difference = 0.0
    comparison_errors = []

    for index in range(ROWS):
        current_row = data.iloc[[index]]

        stream_result = stream.process_state(current_row)

        if index < 9:
            assert stream_result["status"] == "WARMING_UP"
            continue

        assert stream_result["status"] == "PREDICTION_READY"

        # Exactly the same 10 observations used by streaming.
        direct_window = data.iloc[index - 9:index + 1].copy()

        direct_prediction = (
            direct_predictor.predict_raw_sequence(direct_window)
        )

        streaming_prediction = stream_result["prediction"]

        assert direct_prediction.shape == (209,)
        assert streaming_prediction.shape == (209,)

        direct_values = direct_prediction.to_numpy(
            dtype=np.float64
        )

        streaming_values = streaming_prediction.to_numpy(
            dtype=np.float64
        )

        assert np.isfinite(direct_values).all()
        assert np.isfinite(streaming_values).all()

        difference = np.abs(
            direct_values - streaming_values
        )

        maximum_difference = float(difference.max())

        maximum_absolute_difference = max(
            maximum_absolute_difference,
            maximum_difference,
        )

        if maximum_difference > TOLERANCE:
            comparison_errors.append({
                "input_row": index,
                "maximum_difference": maximum_difference,
            })

        assert stream_result["predicted_timestamp"] == (
            timestamps.iloc[index] + pd.Timedelta(seconds=1)
        )

        predictions_checked += 1

    assert predictions_checked == 91
    assert not comparison_errors

    print("PASS | Nine streaming warm-up observations")
    print("PASS | 91 streaming predictions")
    print("PASS | All 209 features match direct inference")
    print("PASS | Predicted timestamps are correct")

    # 5. Save integration report.
    summary = {
        "checkpoint_sha256": checkpoint_hash,
        "input_rows": ROWS,
        "segments": int(data["Segment ID"].nunique()),
        "warmup_observations": 9,
        "streaming_predictions": predictions_checked,
        "feature_count": 209,
        "comparison_method": (
            "Streaming versus direct inference on identical "
            "10-state windows"
        ),
        "absolute_tolerance": TOLERANCE,
        "maximum_absolute_difference": (
            maximum_absolute_difference
        ),
        "all_predictions_finite": True,
        "all_comparisons_within_tolerance": True,
        "all_predicted_timestamps_correct": True,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("STREAMING INTEGRATION SUMMARY")
    print("=" * 72)

    print("Input observations:", ROWS)
    print("Warm-up observations:", 9)
    print("Predictions checked:", predictions_checked)
    print("Features per prediction:", 209)
    print(
        "Maximum absolute difference:",
        f"{maximum_absolute_difference:.10f}",
    )
    print("Tolerance:", TOLERANCE)
    print("Report:", report_path)

    print("\nSTREAMING/DIRECT INTEGRATION: PASSED")


if __name__ == "__main__":
    main()
