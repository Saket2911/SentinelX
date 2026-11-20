from pathlib import Path
import json
import csv
import time
import random

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# SENTINELX — LSTM WORLD MODEL V2
# ============================================================

ROOT = Path(r"F:\Projects\SentinelX")
DATA = ROOT / "Datasets"
OUTPUT = ROOT / "Models" / "lstm_world_model_v2"

SEQ_LEN = 10
INPUT_FEATURES = 212
HIDDEN_SIZE = 128
NUM_LAYERS = 1
DROPOUT = 0.2

BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 30
PATIENCE = 6
SEED = 42

TRAIN_X = DATA / "sentinelx_train_X_seq10_v1.npy"
TRAIN_Y = DATA / "sentinelx_train_y_next_v1.npy"
VAL_X = DATA / "sentinelx_validation_X_seq10_v1.npy"
VAL_Y = DATA / "sentinelx_validation_y_next_v1.npy"
FEATURES_FILE = DATA / "sentinelx_model_features_v1.json"


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# PRE-FLIGHT CHECKS
# ============================================================

print("=" * 75)
print("SENTINELX — LSTM WORLD MODEL V2")
print("=" * 75)

required_files = [
    TRAIN_X, TRAIN_Y, VAL_X, VAL_Y, FEATURES_FILE
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")

if OUTPUT.exists():
    raise FileExistsError(
        f"Output directory already exists: {OUTPUT}\n"
        "Nothing was overwritten. Choose a new output directory "
        "after inspecting the existing one."
    )

features = json.loads(FEATURES_FILE.read_text(encoding="utf-8"))

if len(features) != INPUT_FEATURES:
    raise ValueError(
        f"Expected {INPUT_FEATURES} features, found {len(features)}"
    )

print("\nLoading sequence arrays using memory mapping...")

X_train = np.load(TRAIN_X, mmap_mode="r")
y_train = np.load(TRAIN_Y, mmap_mode="r")
X_val = np.load(VAL_X, mmap_mode="r")
y_val = np.load(VAL_Y, mmap_mode="r")

expected_train = (len(X_train), SEQ_LEN, INPUT_FEATURES)
expected_val = (len(X_val), SEQ_LEN, INPUT_FEATURES)

if X_train.shape != expected_train:
    raise ValueError(f"Invalid training X shape: {X_train.shape}")

if X_val.shape != expected_val:
    raise ValueError(f"Invalid validation X shape: {X_val.shape}")

if y_train.shape != (len(X_train), INPUT_FEATURES):
    raise ValueError(f"Invalid training y shape: {y_train.shape}")

if y_val.shape != (len(X_val), INPUT_FEATURES):
    raise ValueError(f"Invalid validation y shape: {y_val.shape}")

if not (
    len(X_train) == len(y_train)
    and len(X_val) == len(y_val)
):
    raise ValueError("X/y sample counts do not match.")

print("Training X:", X_train.shape)
print("Training y:", y_train.shape)
print("Validation X:", X_val.shape)
print("Validation y:", y_val.shape)
print("Feature count:", len(features))

# Only after all pre-flight checks, create the output directory.
OUTPUT.mkdir(parents=True, exist_ok=False)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3),
            2
        ),
        "GB"
    )


# ============================================================
# MODEL
# ============================================================

class SentinelXLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=INPUT_FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True
        )

        self.dropout = nn.Dropout(DROPOUT)

        self.output_layer = nn.Linear(
            HIDDEN_SIZE,
            INPUT_FEATURES
        )

    def forward(self, x):
        sequence_output, _ = self.lstm(x)

        last_state = sequence_output[:, -1, :]

        last_state = self.dropout(last_state)

        prediction = self.output_layer(last_state)

        return prediction


model = SentinelXLSTM().to(device)

criterion = nn.HuberLoss(delta=1.0)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

print("\nModel parameters:", sum(
    p.numel() for p in model.parameters()
))


# ============================================================
# BATCH HELPERS
# ============================================================

def get_batch(X, y, indices):
    xb = np.array(X[indices], dtype=np.float32, copy=True)
    yb = np.array(y[indices], dtype=np.float32, copy=True)

    xb = torch.from_numpy(xb).to(device)
    yb = torch.from_numpy(yb).to(device)

    return xb, yb


def evaluate(X, y):
    model.eval()

    total_loss = 0.0
    total_absolute_error = 0.0
    total_squared_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(X))
            indices = np.arange(start, end)

            xb, yb = get_batch(X, y, indices)

            predictions = model(xb)
            loss = criterion(predictions, yb)

            batch_size = len(indices)

            total_loss += loss.item() * batch_size

            errors = predictions - yb

            total_absolute_error += (
                errors.abs().sum().item()
            )

            total_squared_error += (
                errors.square().sum().item()
            )

            total_samples += batch_size

    total_values = total_samples * INPUT_FEATURES

    return {
        "huber": total_loss / total_samples,
        "mae": total_absolute_error / total_values,
        "rmse": (total_squared_error / total_values) ** 0.5
    }


# ============================================================
# TRAINING
# ============================================================

history = []
best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0

print("\nStarting training...")
print("Test data will NOT be loaded or evaluated.")
print("-" * 75)

training_start = time.time()

for epoch in range(1, MAX_EPOCHS + 1):

    epoch_start = time.time()

    model.train()

    shuffled_indices = np.random.permutation(len(X_train))

    total_train_loss = 0.0
    total_train_samples = 0

    for start in range(0, len(X_train), BATCH_SIZE):

        batch_indices = shuffled_indices[
            start:start + BATCH_SIZE
        ]

        xb, yb = get_batch(
            X_train,
            y_train,
            batch_indices
        )

        optimizer.zero_grad(set_to_none=True)

        predictions = model(xb)
        loss = criterion(predictions, yb)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        batch_size = len(batch_indices)

        completed_batches = start // BATCH_SIZE + 1
        total_batches = (len(X_train) + BATCH_SIZE - 1) // BATCH_SIZE

        if completed_batches % 500 == 0 or completed_batches == total_batches:
            print(
                f"  Epoch {epoch:02d} | "
                f"Training batches: {completed_batches}/{total_batches}",
                flush=True
            )

        total_train_loss += loss.item() * batch_size
        total_train_samples += batch_size

    train_loss = total_train_loss / total_train_samples

    val_metrics = evaluate(X_val, y_val)

    scheduler.step(val_metrics["huber"])

    current_lr = optimizer.param_groups[0]["lr"]
    epoch_time = time.time() - epoch_start

    row = {
        "epoch": epoch,
        "train_huber": train_loss,
        "validation_huber": val_metrics["huber"],
        "validation_mae": val_metrics["mae"],
        "validation_rmse": val_metrics["rmse"],
        "learning_rate": current_lr,
        "epoch_seconds": epoch_time
    }

    history.append(row)

    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
        f"Train Huber: {train_loss:.6f} | "
        f"Val Huber: {val_metrics['huber']:.6f} | "
        f"Val MAE: {val_metrics['mae']:.6f} | "
        f"Val RMSE: {val_metrics['rmse']:.6f} | "
        f"LR: {current_lr:.7f} | "
        f"Time: {epoch_time:.1f}s"
    )

    if val_metrics["huber"] < best_val_loss:

        best_val_loss = val_metrics["huber"]
        best_epoch = epoch
        epochs_without_improvement = 0

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_huber_loss": best_val_loss,
            "validation_mae": val_metrics["mae"],
            "validation_rmse": val_metrics["rmse"],
            "sequence_length": SEQ_LEN,
            "input_features": INPUT_FEATURES,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "feature_names": features,
            "seed": SEED
        }

        torch.save(
            checkpoint,
            OUTPUT / "best_model.pt"
        )

        print("  -> Best checkpoint saved.")

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping: validation loss did not improve.")
        break


# ============================================================
# SAVE RESULTS
# ============================================================

training_seconds = time.time() - training_start

history_file = OUTPUT / "training_history.csv"

with open(history_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(history[0].keys())
    )
    writer.writeheader()
    writer.writerows(history)

config = {
    "model": "SentinelXLSTM V2",
    "sequence_length": SEQ_LEN,
    "input_features": INPUT_FEATURES,
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
    "dropout": DROPOUT,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "optimizer": "AdamW",
    "loss": "HuberLoss",
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
    "seed": SEED,
    "device": str(device),
    "training_samples": len(X_train),
    "validation_samples": len(X_val),
    "test_data_used": False,
    "best_epoch": best_epoch,
    "best_validation_huber": best_val_loss,
    "training_seconds": training_seconds
}

(OUTPUT / "config.json").write_text(
    json.dumps(config, indent=4),
    encoding="utf-8"
)

report = f"""
SENTINELX — LSTM WORLD MODEL V2
================================

ARCHITECTURE
------------
Input features: {INPUT_FEATURES}
Sequence length: {SEQ_LEN}
Hidden size: {HIDDEN_SIZE}
LSTM layers: {NUM_LAYERS}
Dropout: {DROPOUT}
Output features: {INPUT_FEATURES}

TRAINING
--------
Training samples: {len(X_train)}
Validation samples: {len(X_val)}
Batch size: {BATCH_SIZE}
Optimizer: AdamW
Initial learning rate: {LEARNING_RATE}
Loss: HuberLoss
Maximum epochs: {MAX_EPOCHS}
Early stopping patience: {PATIENCE}

RESULTS
-------
Best epoch: {best_epoch}
Best validation Huber loss: {best_val_loss:.8f}
Training time: {training_seconds:.2f} seconds

TEST DATA
---------
Test data was not loaded or evaluated during training.

CHECKPOINT
----------
best_model.pt

This model predicts the next numerical network state.
Its regression metrics are not attack detection accuracy.
"""

(OUTPUT / "training_report.txt").write_text(
    report,
    encoding="utf-8"
)

print("\n" + "=" * 75)
print("TRAINING COMPLETED")
print("=" * 75)
print("Best epoch:", best_epoch)
print("Best validation Huber:", best_val_loss)
print("Training time:", round(training_seconds, 2), "seconds")
print("Output directory:", OUTPUT)
print("Checkpoint:", OUTPUT / "best_model.pt")
print("Test data used: NO")
