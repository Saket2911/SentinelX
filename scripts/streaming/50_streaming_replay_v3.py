import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sentinelx.inference import SentinelXStreamingPredictor


def main():
    parser = argparse.ArgumentParser(
        description="SentinelX V3 offline streaming replay"
    )
    parser.add_argument(
        "--input",
        default="Datasets/sentinelx_expanded_train_v1.csv",
    )
    parser.add_argument(
        "--checkpoint",
        default="Models/lstm_world_model_v3_gpu_run2/best_model.pt",
    )
    parser.add_argument("--data-dir", default="Datasets")
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        default="reports/streaming_replay_v1",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.rows < 10:
        raise ValueError("--rows must be at least 10.")

    output_dir = Path(args.output_dir)
    predictions_path = output_dir / "streaming_predictions.csv"
    summary_path = output_dir / "streaming_replay_summary.json"

    if predictions_path.exists() or summary_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}. "
            "Choose another --output-dir to avoid overwriting."
        )

    print("=" * 70)
    print("SENTINELX V3 — OFFLINE STREAMING REPLAY")
    print("=" * 70)

    data = pd.read_csv(args.input, nrows=args.rows)

    if len(data) == 0:
        raise ValueError("Input dataset is empty.")

    stream = SentinelXStreamingPredictor.load(
        args.checkpoint,
        data_dir=args.data_dir,
        device=args.device,
    )

    records = []
    latencies = []
    warmup_count = 0
    reset_counts = {
        "segment_change": 0,
        "timestamp_gap": 0,
    }

    for index in range(len(data)):
        row = data.iloc[[index]]

        start = time.perf_counter()
        result = stream.process_state(row)
        elapsed = time.perf_counter() - start

        if result["reset_reason"] in reset_counts:
            reset_counts[result["reset_reason"]] += 1

        if result["status"] == "WARMING_UP":
            warmup_count += 1
            continue

        prediction = result["prediction"]

        if prediction.shape != (209,):
            raise ValueError(
                f"Unexpected prediction shape at row {index}: "
                f"{prediction.shape}"
            )

        values = prediction.to_numpy(dtype=np.float32)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite prediction at input row {index}."
            )

        record = {
            "Input_Row_Index": index,
            "Input_Timestamp": result["timestamp"].isoformat(),
            "Predicted_Timestamp": (
                result["predicted_timestamp"].isoformat()
            ),
            "Segment_ID": row["Segment ID"].iloc[0],
        }

        record.update({
            f"Predicted_{name}": float(value)
            for name, value in zip(stream.feature_names, values)
        })

        records.append(record)
        latencies.append(elapsed)

    if not records:
        raise RuntimeError("Replay produced no predictions.")

    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.DataFrame(records)
    predictions_df.to_csv(predictions_path, index=False)

    latency_array = np.asarray(latencies, dtype=np.float64)

    summary = {
        "input_file": args.input,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "input_rows": len(data),
        "warmup_observations": warmup_count,
        "predictions_produced": len(records),
        "sequence_length": stream.sequence_length,
        "feature_count": len(stream.feature_names),
        "prediction_shape": [209],
        "all_predictions_finite": True,
        "reset_counts": reset_counts,
        "latency_ms": {
            "mean": float(latency_array.mean() * 1000),
            "median": float(np.median(latency_array) * 1000),
            "maximum": float(latency_array.max() * 1000),
        },
        "prediction_csv": str(predictions_path),
    }

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nReplay results:")
    print("Input observations:", len(data))
    print("Warm-up observations:", warmup_count)
    print("Predictions:", len(records))
    print("Prediction features:", len(stream.feature_names))
    print("Segment resets:", reset_counts["segment_change"])
    print("Timestamp-gap resets:", reset_counts["timestamp_gap"])

    print("\nInference latency:")
    print(f"Mean:   {summary['latency_ms']['mean']:.3f} ms")
    print(f"Median: {summary['latency_ms']['median']:.3f} ms")
    print(f"Max:    {summary['latency_ms']['maximum']:.3f} ms")

    print("\nSaved files:")
    print(predictions_path)
    print(summary_path)

    print("\nSTREAMING REPLAY COMPLETED")


if __name__ == "__main__":
    main()
