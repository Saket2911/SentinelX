import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sentinelx.inference import SentinelXStreamingPredictor


def main():
    parser = argparse.ArgumentParser(
        description="SentinelX V3 representative segment replay"
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        default="Reports/streaming_representative_replay_v1",
    )
    args = parser.parse_args()

    # Segment selection: medium, large, and one-state segment.
    segment_limits = {
        114: 500,
        195: 500,
        443: 1,
    }
    selected_counts = {sid: 0 for sid in segment_limits}

    output_dir = Path(args.output_dir)
    predictions_path = output_dir / "streaming_predictions.csv"
    summary_path = output_dir / "streaming_replay_summary.json"

    if predictions_path.exists() or summary_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}. "
            "Choose another --output-dir."
        )

    print("=" * 70)
    print("SENTINELX V3 — REPRESENTATIVE SEGMENT REPLAY")
    print("=" * 70)

    # Read in chunks to avoid loading the entire dataset into memory.
    selected_chunks = []
    source_row_offset = 0

    for chunk in pd.read_csv(args.input, chunksize=5000):
        keep_indices = []

        for index, segment_id in zip(
            chunk.index,
            chunk["Segment ID"],
        ):
            sid = int(segment_id)

            if sid not in segment_limits:
                continue

            if selected_counts[sid] < segment_limits[sid]:
                keep_indices.append(index)
                selected_counts[sid] += 1

        if keep_indices:
            selected = chunk.loc[keep_indices].copy()
            selected["Source_Row_Index"] = (
                selected.index + source_row_offset
            )
            selected_chunks.append(selected)

        source_row_offset += len(chunk)

        if all(
            selected_counts[sid] >= limit
            for sid, limit in segment_limits.items()
        ):
            break

    if not selected_chunks:
        raise RuntimeError("No selected observations were found.")

    data = pd.concat(selected_chunks, ignore_index=True)

    if selected_counts != segment_limits:
        raise RuntimeError(
            f"Could not collect all requested rows: {selected_counts}"
        )

    print("\nSelected segment observations:")
    for sid, count in selected_counts.items():
        print(f"Segment {sid}: {count} observations")

    print("\nTotal selected observations:", len(data))

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

    for position in range(len(data)):
        row = data.iloc[[position]]

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
                f"Unexpected prediction shape at selected row {position}: "
                f"{prediction.shape}"
            )

        values = prediction.to_numpy(dtype=np.float32)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite prediction at selected row {position}."
            )

        record = {
            "Input_Row_Index": int(row["Source_Row_Index"].iloc[0]),
            "Input_Timestamp": result["timestamp"].isoformat(),
            "Predicted_Timestamp": (
                result["predicted_timestamp"].isoformat()
            ),
            "Segment_ID": int(row["Segment ID"].iloc[0]),
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
        "selected_segments": {
            str(sid): {
                "requested_observations": segment_limits[sid],
                "selected_observations": selected_counts[sid],
            }
            for sid in segment_limits
        },
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

    print("\nREPRESENTATIVE STREAMING REPLAY COMPLETED")


if __name__ == "__main__":
    main()
