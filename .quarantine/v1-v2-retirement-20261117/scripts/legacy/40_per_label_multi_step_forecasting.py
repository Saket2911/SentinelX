import os
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"F:\Projects\SentinelX")
DATA = ROOT / "Datasets"
MODEL_DIR = ROOT / "Models" / "lstm_world_model_v1"
OUTPUT_DIR = MODEL_DIR / "per_label_multi_step_forecasting_v1"

CHECKPOINT = MODEL_DIR / "best_model.pt"
if not CHECKPOINT.exists():
    CHECKPOINT = MODEL_DIR / "best_model.pt"

# Do not overwrite previous experiment results.
if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Output folder already exists: {OUTPUT_DIR}\n"
        "Do not overwrite it. Rename the folder or inspect its contents first."
    )

OUTPUT_DIR.mkdir(parents=True)

HORIZONS = [1, 2, 5, 10, 30]
SEQ_LEN = 10
INPUT_SIZE = 206
HIDDEN_SIZE = 128
OUTPUT_SIZE = 206
BATCH_SIZE = 128

class LSTMWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.2)
        self.output_layer = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.output_layer(self.dropout(last))

def huber_sum(pred, target, delta=1.0):
    error = torch.abs(pred - target)
    loss = torch.where(
        error <= delta,
        0.5 * error ** 2,
        delta * (error - 0.5 * delta)
    )
    return loss.sum().item()

def add_metrics(store, key, pred, target):
    diff = pred - target
    store[key]["abs"] += np.abs(diff).sum(dtype=np.float64)
    store[key]["sq"] += np.square(diff).sum(dtype=np.float64)
    store[key]["huber"] += huber_sum(
        torch.from_numpy(pred),
        torch.from_numpy(target)
    )
    store[key]["elements"] += diff.size
    store[key]["states"] += len(pred)

def finalize(store):
    result = {}
    for key, values in store.items():
        n = values["elements"]
        result[key] = {
            "states": values["states"],
            "mae": values["abs"] / n,
            "rmse": np.sqrt(values["sq"] / n),
            "huber": values["huber"] / n
        }
    return result

def main():
    print("Loading test arrays...")

    X = np.load(DATA / "sentinelx_test_X.npy", mmap_mode="r")
    y = np.load(DATA / "sentinelx_test_y.npy", mmap_mode="r")
    labels = np.load(DATA / "sentinelx_test_target_labels.npy")
    timestamps = np.load(DATA / "sentinelx_test_target_timestamps.npy")
    segment_ids = np.load(DATA / "sentinelx_test_target_segment_ids.npy")

    n = len(X)
    if not (len(y) == len(labels) == len(timestamps) == len(segment_ids) == n):
        raise ValueError("Test array lengths do not match.")

    if X.shape[1:] != (SEQ_LEN, INPUT_SIZE):
        raise ValueError(f"Unexpected X shape: {X.shape}")
    if y.shape[1] != OUTPUT_SIZE:
        raise ValueError(f"Unexpected y shape: {y.shape}")

    origins = n - max(HORIZONS) + 1
    if origins <= 0:
        raise ValueError("Not enough test sequences.")

    print(f"Test sequences: {n}")
    print(f"Common forecast origins: {origins}")
    print("Labels:", dict(zip(*np.unique(labels, return_counts=True))))

    # Validate all target steps and segment boundaries.
    for h in HORIZONS:
        target_idx = np.arange(origins) + h - 1
        if not np.all(np.diff(timestamps[target_idx]) == np.timedelta64(1, "s")):
            raise ValueError(f"Timestamp continuity failed at horizon {h}.")
        if not np.all(segment_ids[target_idx] == segment_ids[:origins]):
            raise ValueError(f"Segment alignment failed at horizon {h}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = LSTMWorldModel().to(device)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    print("Loaded checkpoint:", CHECKPOINT)

    # Accumulate errors separately by horizon and actual future label.
    lstm_store = defaultdict(lambda: {
        "abs": 0.0, "sq": 0.0, "huber": 0.0,
        "elements": 0, "states": 0
    })
    persistence_store = defaultdict(lambda: {
        "abs": 0.0, "sq": 0.0, "huber": 0.0,
        "elements": 0, "states": 0
    })

    # Process in batches to control memory use.
    with torch.no_grad():
        for start in range(0, origins, BATCH_SIZE):
            end = min(start + BATCH_SIZE, origins)
            batch_n = end - start

            # Copy only this batch from the memory-mapped test arrays.
            current = torch.from_numpy(
                np.array(X[start:end], dtype=np.float32, copy=True)
            ).to(device)

            # Persistence repeats the most recent observed state.
            last_observed = current[:, -1, :].clone()

            forecasts = {}
            for h in range(1, max(HORIZONS) + 1):
                prediction = model(current)
                forecasts[h] = prediction.cpu().numpy()

                # Recursively append the prediction and discard oldest state.
                current = torch.cat(
                    [current[:, 1:, :], prediction.unsqueeze(1)],
                    dim=1
                )

            for h in HORIZONS:
                target_indices = np.arange(start, end) + h - 1
                actual = np.array(y[target_indices], dtype=np.float32, copy=True)
                predicted = forecasts[h]
                baseline = last_observed.cpu().numpy()

                future_labels = labels[target_indices]

                for label in np.unique(future_labels):
                    mask = future_labels == label
                    key = (h, str(label))

                    add_metrics(
                        lstm_store, key,
                        predicted[mask], actual[mask]
                    )
                    add_metrics(
                        persistence_store, key,
                        baseline[mask], actual[mask]
                    )

            if start == 0 or end == origins or (start // BATCH_SIZE) % 20 == 0:
                print(f"Processed origins: {end}/{origins}")

    lstm_results = finalize(lstm_store)
    persistence_results = finalize(persistence_store)

    rows = []
    for key in sorted(lstm_results):
        h, label = key
        lm = lstm_results[key]
        pm = persistence_results[key]

        rows.append({
            "Horizon_Seconds": h,
            "Target_Label": label,
            "Valid_Origins": lm["states"],
            "LSTM_MAE": lm["mae"],
            "Persistence_MAE": pm["mae"],
            "MAE_Improvement_Percent": (
                (pm["mae"] - lm["mae"]) / pm["mae"] * 100
                if pm["mae"] else 0.0
            ),
            "LSTM_RMSE": lm["rmse"],
            "Persistence_RMSE": pm["rmse"],
            "RMSE_Improvement_Percent": (
                (pm["rmse"] - lm["rmse"]) / pm["rmse"] * 100
                if pm["rmse"] else 0.0
            ),
            "LSTM_Huber": lm["huber"],
            "Persistence_Huber": pm["huber"]
        })

    csv_path = OUTPUT_DIR / "per_label_multi_step_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report_path = OUTPUT_DIR / "per_label_multi_step_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("SENTINELX - PER-LABEL MULTI-STEP FORECASTING\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test sequences: {n}\n")
        f.write(f"Common forecast origins: {origins}\n")
        f.write(f"Sequence length: {SEQ_LEN}\n")
        f.write(f"Input/output features: {INPUT_SIZE}\n")
        f.write(f"Horizons: {HORIZONS}\n")
        f.write(f"Device: {device}\n")
        f.write("Model: Existing LSTM World Model v1; no retraining\n")
        f.write("Grouping: Actual future target label at i + horizon - 1\n")
        f.write("Baseline: Repeat the last observed state\n\n")

        for row in rows:
            f.write(
                f"Horizon {row['Horizon_Seconds']} sec | "
                f"Label: {row['Target_Label']} | "
                f"States: {row['Valid_Origins']}\n"
            )
            f.write(
                f"  LSTM MAE: {row['LSTM_MAE']:.6f} | "
                f"Persistence MAE: {row['Persistence_MAE']:.6f} | "
                f"Improvement: {row['MAE_Improvement_Percent']:.2f}%\n"
            )
            f.write(
                f"  LSTM RMSE: {row['LSTM_RMSE']:.6f} | "
                f"Persistence RMSE: {row['Persistence_RMSE']:.6f} | "
                f"Improvement: {row['RMSE_Improvement_Percent']:.2f}%\n"
            )
            f.write(
                f"  LSTM Huber: {row['LSTM_Huber']:.6f} | "
                f"Persistence Huber: {row['Persistence_Huber']:.6f}\n\n"
            )

        f.write("INTERPRETATION LIMITATIONS\n")
        f.write("- These are future-state forecasting metrics, not attack-detection metrics.\n")
        f.write("- Labels are used only to group actual future targets, not as model inputs.\n")
        f.write("- The test partition contains only the labels observed in this partition.\n")
        f.write("- Results describe this dataset partition, not all real networks.\n")
        f.write("- Test data have been used in earlier exploratory analyses.\n")

    print("\nEXPERIMENT COMPLETE")
    print("Metrics:", csv_path)
    print("Report:", report_path)
    print("\nResults:")
    for row in rows:
        print(
            f"{row['Horizon_Seconds']:>2}s | "
            f"{row['Target_Label']:<8} | "
            f"N={row['Valid_Origins']:>5} | "
            f"MAE {row['LSTM_MAE']:.5f} vs "
            f"{row['Persistence_MAE']:.5f} | "
            f"Improvement {row['MAE_Improvement_Percent']:.2f}%"
        )

if __name__ == "__main__":
    main()
