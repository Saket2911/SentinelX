from pathlib import Path
from collections import defaultdict
import csv

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

ROOT = Path(r"F:\Projects\SentinelX")
DATA_DIR = ROOT / "Datasets"
MODEL_DIR = ROOT / "Models" / "lstm_world_model_v1"

OUTPUT_DIR = MODEL_DIR / "prediction_error_analysis_v1"

CHECKPOINT = MODEL_DIR / "best_model.pt"

BATCH_SIZE = 64
SEQUENCE_LENGTH = 10
INPUT_FEATURES = 206
HIDDEN_SIZE = 128

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Output directory already exists: {OUTPUT_DIR}\n"
        "Existing analysis will not be overwritten."
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("SENTINELX - PREDICTION ERROR ANALYSIS")
print("=" * 65)
print("Device:", DEVICE)


# ---------------------------------------------------------
# 2. Load validation arrays
# ---------------------------------------------------------

def load_array(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(path)

    return np.load(path, mmap_mode="r")


X = load_array("sentinelx_validation_X.npy")
y = load_array("sentinelx_validation_y.npy")

labels = load_array(
    "sentinelx_validation_target_labels.npy"
)

timestamps = load_array(
    "sentinelx_validation_target_timestamps.npy"
)

segment_ids = load_array(
    "sentinelx_validation_target_segment_ids.npy"
)

feature_names_path = (
    DATA_DIR / "sentinelx_sequence_feature_names.txt"
)

feature_names = [
    line.strip()
    for line in feature_names_path.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

if len(feature_names) != INPUT_FEATURES:
    raise ValueError(
        f"Expected {INPUT_FEATURES} feature names, "
        f"found {len(feature_names)}."
    )

if not (
    len(X) == len(y) == len(labels)
    == len(timestamps) == len(segment_ids)
):
    raise ValueError("Validation array lengths do not match.")

if X.shape[1:] != (SEQUENCE_LENGTH, INPUT_FEATURES):
    raise ValueError(f"Unexpected X shape: {X.shape}")

print("Validation sequences:", len(X))
print("Input shape:", X.shape)
print("Target shape:", y.shape)
print("Features:", len(feature_names))


# ---------------------------------------------------------
# 3. Reconstruct the trained model
# ---------------------------------------------------------

class SentinelXLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=INPUT_FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.2)

        self.output_layer = nn.Linear(
            HIDDEN_SIZE,
            INPUT_FEATURES
        )

    def forward(self, x):
        sequence_output, _ = self.lstm(x)
        final_state = sequence_output[:, -1, :]
        final_state = self.dropout(final_state)

        return self.output_layer(final_state)


model = SentinelXLSTM().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Loaded checkpoint from epoch:", checkpoint["epoch"])


# ---------------------------------------------------------
# 4. Error accumulators
# ---------------------------------------------------------

feature_abs_sum = np.zeros(
    INPUT_FEATURES, dtype=np.float64
)

feature_squared_sum = np.zeros(
    INPUT_FEATURES, dtype=np.float64
)

sequence_mae = np.zeros(len(X), dtype=np.float32)
sequence_rmse = np.zeros(len(X), dtype=np.float32)
sequence_huber = np.zeros(len(X), dtype=np.float32)

label_stats = defaultdict(
    lambda: {
        "count": 0,
        "mae_sum": 0.0,
        "rmse_sum": 0.0,
        "huber_sum": 0.0
    }
)


def decode_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


# ---------------------------------------------------------
# 5. Generate predictions and calculate errors
# ---------------------------------------------------------

print("\nCalculating validation prediction errors...")

with torch.no_grad():

    for start in range(0, len(X), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(X))

        xb_np = np.array(
            X[start:end], dtype=np.float32, copy=True
        )

        yb_np = np.array(
            y[start:end], dtype=np.float32, copy=True
        )

        xb = torch.from_numpy(xb_np).to(DEVICE)

        predictions = model(xb).cpu().numpy()

        difference = predictions - yb_np
        absolute = np.abs(difference)

        squared = difference ** 2

        huber = np.where(
            absolute <= 1.0,
            0.5 * squared,
            absolute - 0.5
        )

        # Feature-wise error accumulation.
        feature_abs_sum += absolute.sum(axis=0)
        feature_squared_sum += squared.sum(axis=0)

        # Per-sequence errors.
        batch_mae = absolute.mean(axis=1)
        batch_rmse = np.sqrt(squared.mean(axis=1))
        batch_huber = huber.mean(axis=1)

        sequence_mae[start:end] = batch_mae
        sequence_rmse[start:end] = batch_rmse
        sequence_huber[start:end] = batch_huber

        # Per-label statistics.
        for i, label_value in enumerate(labels[start:end]):

            label = decode_text(label_value)

            stats = label_stats[label]

            stats["count"] += 1
            stats["mae_sum"] += float(batch_mae[i])
            stats["rmse_sum"] += float(batch_rmse[i])
            stats["huber_sum"] += float(batch_huber[i])

        print(f"Processed {end}/{len(X)}")


# ---------------------------------------------------------
# 6. Feature-wise error report
# ---------------------------------------------------------

feature_mae = feature_abs_sum / len(X)

feature_rmse = np.sqrt(
    feature_squared_sum / len(X)
)

feature_order = np.argsort(feature_mae)[::-1]

feature_csv = OUTPUT_DIR / "feature_error_analysis.csv"

with feature_csv.open("w", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)

    writer.writerow([
        "feature",
        "mean_absolute_error",
        "root_mean_squared_error"
    ])

    for index in feature_order:

        writer.writerow([
            feature_names[index],
            f"{feature_mae[index]:.8f}",
            f"{feature_rmse[index]:.8f}"
        ])


# ---------------------------------------------------------
# 7. Per-sequence chronological error report
# ---------------------------------------------------------

sequence_csv = OUTPUT_DIR / "sequence_error_analysis.csv"

with sequence_csv.open("w", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)

    writer.writerow([
        "timestamp",
        "label",
        "segment_id",
        "mae",
        "rmse",
        "huber_loss"
    ])

    for i in range(len(X)):

        writer.writerow([
            decode_text(timestamps[i]),
            decode_text(labels[i]),
            int(segment_ids[i]),
            f"{sequence_mae[i]:.8f}",
            f"{sequence_rmse[i]:.8f}",
            f"{sequence_huber[i]:.8f}"
        ])


# ---------------------------------------------------------
# 8. Per-label error report
# ---------------------------------------------------------

label_csv = OUTPUT_DIR / "label_error_analysis.csv"

with label_csv.open("w", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)

    writer.writerow([
        "label",
        "sequence_count",
        "mean_mae",
        "mean_rmse",
        "mean_huber_loss"
    ])

    for label in sorted(label_stats):

        stats = label_stats[label]
        count = stats["count"]

        writer.writerow([
            label,
            count,
            f"{stats['mae_sum'] / count:.8f}",
            f"{stats['rmse_sum'] / count:.8f}",
            f"{stats['huber_sum'] / count:.8f}"
        ])


# ---------------------------------------------------------
# 9. Overall distribution statistics
# ---------------------------------------------------------

percentiles = np.percentile(
    sequence_mae,
    [50, 75, 90, 95, 99]
)

overall_mae = float(sequence_mae.mean())
overall_rmse = float(
    np.sqrt(np.mean(sequence_rmse ** 2))
)

report_path = OUTPUT_DIR / "prediction_error_report.txt"

report = []

report.append("SENTINELX - PREDICTION ERROR ANALYSIS")
report.append("=" * 55)

report.append("\nMODEL")
report.append(f"Checkpoint epoch: {checkpoint['epoch']}")
report.append(f"Device: {DEVICE}")
report.append(f"Validation sequences: {len(X)}")
report.append(f"Features: {INPUT_FEATURES}")

report.append("\nOVERALL VALIDATION ERRORS")
report.append(f"Mean sequence MAE: {overall_mae:.8f}")
report.append(f"Overall RMSE: {overall_rmse:.8f}")

report.append("\nSEQUENCE MAE DISTRIBUTION")

for percentile, value in zip(
    [50, 75, 90, 95, 99],
    percentiles
):
    report.append(
        f"{percentile}th percentile: {value:.8f}"
    )

report.append("\nPER-LABEL RESULTS")

for label in sorted(label_stats):

    stats = label_stats[label]
    count = stats["count"]

    report.append(
        f"\n{label} ({count} sequences)"
    )

    report.append(
        f"  MAE: {stats['mae_sum'] / count:.8f}"
    )

    report.append(
        f"  RMSE: {stats['rmse_sum'] / count:.8f}"
    )

    report.append(
        f"  Huber: {stats['huber_sum'] / count:.8f}"
    )

report.append("\nTOP 15 FEATURES WITH HIGHEST MAE")

for index in feature_order[:15]:

    report.append(
        f"{feature_names[index]}: "
        f"MAE={feature_mae[index]:.8f}, "
        f"RMSE={feature_rmse[index]:.8f}"
    )

report.append("\nLOWEST 15 FEATURES BY MAE")

for index in feature_order[-15:][::-1]:

    report.append(
        f"{feature_names[index]}: "
        f"MAE={feature_mae[index]:.8f}, "
        f"RMSE={feature_rmse[index]:.8f}"
    )

report.append("\nOUTPUT FILES")

report.append(str(feature_csv))
report.append(str(sequence_csv))
report.append(str(label_csv))

report.append(
    "\nNOTE: Errors are measured in preprocessed feature space. "
    "High prediction error alone does not prove malicious activity."
)

report_path.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# ---------------------------------------------------------
# 10. Display summary
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("PREDICTION ERROR ANALYSIS COMPLETED")
print("=" * 65)

print(f"Mean sequence MAE: {overall_mae:.6f}")
print(f"Overall RMSE: {overall_rmse:.6f}")

print("\nSequence MAE percentiles:")

for percentile, value in zip(
    [50, 75, 90, 95, 99],
    percentiles
):
    print(f"{percentile}th: {value:.6f}")

print("\nPer-label results:")

for label in sorted(label_stats):

    stats = label_stats[label]
    count = stats["count"]

    print(
        f"{label}: "
        f"Count={count}, "
        f"MAE={stats['mae_sum'] / count:.6f}"
    )

print("\nTop 15 highest-error features:")

for index in feature_order[:15]:

    print(
        f"{feature_names[index]}: "
        f"{feature_mae[index]:.6f}"
    )

print("\nReports saved in:")
print(OUTPUT_DIR)

print("\nValidation analysis completed.")
print("Test dataset was not used.")
