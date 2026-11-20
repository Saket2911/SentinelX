
from pathlib import Path
import json
import csv
import time
import random

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# SENTINELX - LSTM WORLD MODEL V3
# GPU-OPTIMIZED CHUNK TRAINING
# ============================================================

ROOT = Path(r"F:\Projects\SentinelX")
DATA = ROOT / "Datasets"

# New directory: preserve the previous training run.
OUTPUT = ROOT / "Models" / "lstm_world_model_v3_gpu_run2"

SEQ_LEN = 10
INPUT_FEATURES = 209
HIDDEN_SIZE = 128
NUM_LAYERS = 1
DROPOUT = 0.2

# GPU optimization
BATCH_SIZE = 128
CHUNK_SIZE = 8192

LEARNING_RATE = 0.001
MAX_EPOCHS = 30
PATIENCE = 6
SEED = 42

TRAIN_X = DATA / "sentinelx_train_X_seq10_v3.npy"
TRAIN_Y = DATA / "sentinelx_train_y_seq10_v3.npy"
VAL_X = DATA / "sentinelx_validation_X_seq10_v3.npy"
VAL_Y = DATA / "sentinelx_validation_y_seq10_v3.npy"

FEATURES_FILE = DATA / "sentinelx_model_features_v3.json"


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# GPU INITIALIZATION
# ============================================================

print("=" * 75)
print("SENTINELX - LSTM WORLD MODEL V3")
print("GPU-OPTIMIZED CHUNK TRAINING")
print("=" * 75)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable. CPU fallback is disabled.")

device = torch.device("cuda:0")

torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

torch.set_num_threads(4)

print("\nGPU CONFIGURATION")
print("-" * 75)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

gpu_properties = torch.cuda.get_device_properties(0)

print(
    "Total GPU memory:",
    round(gpu_properties.total_memory / (1024 ** 3), 2),
    "GB"
)

print("cuDNN:", torch.backends.cudnn.enabled)
print("Batch size:", BATCH_SIZE)
print("GPU chunk size:", CHUNK_SIZE)


# ============================================================
# PRE-FLIGHT CHECKS
# ============================================================

required_files = [
    TRAIN_X,
    TRAIN_Y,
    VAL_X,
    VAL_Y,
    FEATURES_FILE
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

if OUTPUT.exists():
    raise FileExistsError(
        f"Output directory already exists:\n{OUTPUT}\n"
        "Choose a new output directory to preserve previous results."
    )

features = json.loads(
    FEATURES_FILE.read_text(encoding="utf-8")
)

if len(features) != INPUT_FEATURES:
    raise ValueError(
        f"Expected {INPUT_FEATURES} features, "
        f"found {len(features)}."
    )


# ============================================================
# LOAD DATA AS MEMORY-MAPPED ARRAYS
# ============================================================

print("\nOpening sequence arrays...")

X_train = np.load(TRAIN_X, mmap_mode="r")
y_train = np.load(TRAIN_Y, mmap_mode="r")

X_val = np.load(VAL_X, mmap_mode="r")
y_val = np.load(VAL_Y, mmap_mode="r")


def validate_arrays(X, y, name):

    if X.ndim != 3:
        raise ValueError(f"{name} X must be 3-dimensional.")

    if X.shape[1:] != (SEQ_LEN, INPUT_FEATURES):
        raise ValueError(f"Invalid {name} X shape: {X.shape}")

    if y.shape != (len(X), INPUT_FEATURES):
        raise ValueError(f"Invalid {name} y shape: {y.shape}")

    if len(X) != len(y):
        raise ValueError(f"{name} X/y sample counts differ.")


validate_arrays(X_train, y_train, "Training")
validate_arrays(X_val, y_val, "Validation")

print("\nDATASET INFORMATION")
print("-" * 75)
print("Training X:", X_train.shape)
print("Training y:", y_train.shape)
print("Validation X:", X_val.shape)
print("Validation y:", y_val.shape)
print("Features:", len(features))
print("Sequence length:", SEQ_LEN)
print("Test data: untouched")


# Create a new output directory.
OUTPUT.mkdir(parents=True, exist_ok=False)


# ============================================================
# LSTM WORLD MODEL
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

        return self.output_layer(last_state)


# ============================================================
# GPU CHUNK LOADER
# ============================================================

def load_gpu_chunk(X, y, start, end):

    # Read one contiguous chunk from disk-backed arrays.
    x_numpy = np.array(
        X[start:end],
        dtype=np.float32,
        copy=True
    )

    y_numpy = np.array(
        y[start:end],
        dtype=np.float32,
        copy=True
    )

    # Transfer the chunk to GPU once.
    x_gpu = torch.from_numpy(x_numpy).to(device)
    y_gpu = torch.from_numpy(y_numpy).to(device)

    return x_gpu, y_gpu


# ============================================================
# MODEL INITIALIZATION
# ============================================================

model = SentinelXLSTM().to(device)

if next(model.parameters()).device.type != "cuda":
    raise RuntimeError("Model failed to initialize on CUDA.")

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

parameter_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\nMODEL INFORMATION")
print("-" * 75)
print("Architecture: LSTM World Model V3")
print("Input features:", INPUT_FEATURES)
print("Hidden size:", HIDDEN_SIZE)
print("LSTM layers:", NUM_LAYERS)
print("Output features:", INPUT_FEATURES)
print("Trainable parameters:", f"{parameter_count:,}")
print("Model device:", next(model.parameters()).device)


# ============================================================
# VALIDATION - GPU CHUNK PROCESSING
# ============================================================

@torch.no_grad()
def evaluate(X, y):

    model.eval()

    total_loss = torch.zeros((), device=device)
    total_absolute_error = torch.zeros((), device=device)
    total_squared_error = torch.zeros((), device=device)

    total_samples = 0

    for chunk_start in range(0, len(X), CHUNK_SIZE):

        chunk_end = min(
            chunk_start + CHUNK_SIZE,
            len(X)
        )

        x_chunk, y_chunk = load_gpu_chunk(
            X, y, chunk_start, chunk_end
        )

        chunk_length = len(x_chunk)

        # Mini-batch indexing happens on GPU.
        for start in range(0, chunk_length, BATCH_SIZE):

            end = min(start + BATCH_SIZE, chunk_length)

            xb = x_chunk[start:end]
            yb = y_chunk[start:end]

            predictions = model(xb)

            loss = criterion(predictions, yb)

            batch_size = len(xb)

            errors = predictions - yb

            total_loss += loss * batch_size
            total_absolute_error += errors.abs().sum()
            total_squared_error += errors.square().sum()

            total_samples += batch_size

    total_values = total_samples * INPUT_FEATURES

    return {
        "huber": (total_loss / total_samples).item(),

        "mae": (
            total_absolute_error / total_values
        ).item(),

        "rmse": (
            total_squared_error / total_values
        ).sqrt().item()
    }


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

history = []

best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0

total_batches = (
    len(X_train) + BATCH_SIZE - 1
) // BATCH_SIZE

total_chunks = (
    len(X_train) + CHUNK_SIZE - 1
) // CHUNK_SIZE

print("\nSTARTING GPU TRAINING")
print("=" * 75)
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Batches per epoch:", total_batches)
print("GPU chunks per epoch:", total_chunks)
print("Training device:", device)
print("Test data: untouched")
print("=" * 75)

training_start = time.time()


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

for epoch in range(1, MAX_EPOCHS + 1):

    epoch_start = time.time()

    model.train()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    # Shuffle the order in which chunks are processed.
    chunk_order = np.random.permutation(total_chunks)

    epoch_loss_sum = torch.zeros((), device=device)
    total_train_samples = 0

    completed_batches = 0

    for chunk_number in chunk_order:

        chunk_start = int(chunk_number) * CHUNK_SIZE

        chunk_end = min(
            chunk_start + CHUNK_SIZE,
            len(X_train)
        )

        # Transfer an entire chunk instead of every batch.
        x_chunk, y_chunk = load_gpu_chunk(
            X_train,
            y_train,
            chunk_start,
            chunk_end
        )

        chunk_length = len(x_chunk)

        # Shuffle samples directly on GPU.
        shuffled_indices = torch.randperm(
            chunk_length,
            device=device
        )

        for start in range(0, chunk_length, BATCH_SIZE):

            end = min(start + BATCH_SIZE, chunk_length)

            batch_indices = shuffled_indices[start:end]

            # GPU-side mini-batch selection.
            xb = x_chunk.index_select(0, batch_indices)
            yb = y_chunk.index_select(0, batch_indices)

            if epoch == 1 and completed_batches == 0:

                if xb.device.type != "cuda":
                    raise RuntimeError("Input batch is not on CUDA.")

                if yb.device.type != "cuda":
                    raise RuntimeError("Target batch is not on CUDA.")

                print("\nGPU EXECUTION VERIFIED")
                print("Input:", xb.device)
                print("Target:", yb.device)
                print("Model:", next(model.parameters()).device)

            optimizer.zero_grad(set_to_none=True)

            predictions = model(xb)

            loss = criterion(predictions, yb)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            batch_size = len(xb)

            # Accumulate metrics on GPU.
            # Avoid calling .item() after every batch.
            epoch_loss_sum += loss.detach() * batch_size

            total_train_samples += batch_size

            completed_batches += 1

            if (
                completed_batches % 500 == 0
                or completed_batches == total_batches
            ):

                print(
                    f"Epoch {epoch:02d} | "
                    f"Batch {completed_batches}/{total_batches}",
                    flush=True
                )

        # Release references to the previous chunk.
        del x_chunk, y_chunk, shuffled_indices

    torch.cuda.synchronize(device)

    train_loss = (
        epoch_loss_sum / total_train_samples
    ).item()

    # GPU-based chunked validation.
    val_metrics = evaluate(X_val, y_val)

    torch.cuda.synchronize(device)

    scheduler.step(val_metrics["huber"])

    current_lr = optimizer.param_groups[0]["lr"]

    epoch_time = time.time() - epoch_start

    allocated_memory = (
        torch.cuda.memory_allocated(device) / (1024 ** 2)
    )

    reserved_memory = (
        torch.cuda.memory_reserved(device) / (1024 ** 2)
    )

    peak_memory = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    )

    row = {
        "epoch": epoch,
        "train_huber": train_loss,
        "validation_huber": val_metrics["huber"],
        "validation_mae": val_metrics["mae"],
        "validation_rmse": val_metrics["rmse"],
        "learning_rate": current_lr,
        "epoch_seconds": epoch_time,
        "gpu_allocated_mb": allocated_memory,
        "gpu_reserved_mb": reserved_memory,
        "gpu_peak_mb": peak_memory
    }

    history.append(row)

    print("\n" + "-" * 75)

    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
        f"Train Huber: {train_loss:.6f} | "
        f"Val Huber: {val_metrics['huber']:.6f} | "
        f"Val MAE: {val_metrics['mae']:.6f} | "
        f"Val RMSE: {val_metrics['rmse']:.6f} | "
        f"LR: {current_lr:.7f} | "
        f"Time: {epoch_time:.1f}s"
    )

    print(
        f"GPU Memory | "
        f"Allocated: {allocated_memory:.1f} MB | "
        f"Reserved: {reserved_memory:.1f} MB | "
        f"Peak: {peak_memory:.1f} MB"
    )

    print("-" * 75)

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_metrics["huber"] < best_val_loss:

        best_val_loss = val_metrics["huber"]
        best_epoch = epoch

        epochs_without_improvement = 0

        checkpoint = {
            "epoch": epoch,

            "model_state_dict": {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
            },

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

        print(" -> Best checkpoint saved.")

    else:

        epochs_without_improvement += 1

        print(
            "No improvement:",
            epochs_without_improvement,
            "/",
            PATIENCE
        )

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")
        break


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

torch.cuda.synchronize(device)

training_seconds = time.time() - training_start

history_file = OUTPUT / "training_history.csv"

with open(
    history_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(history[0].keys())
    )

    writer.writeheader()
    writer.writerows(history)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

config = {
    "model": "SentinelXLSTM V3",

    "sequence_length": SEQ_LEN,
    "input_features": INPUT_FEATURES,
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
    "dropout": DROPOUT,

    "batch_size": BATCH_SIZE,
    "chunk_size": CHUNK_SIZE,

    "learning_rate": LEARNING_RATE,
    "optimizer": "AdamW",
    "loss": "HuberLoss",

    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
    "seed": SEED,

    "device": str(device),
    "gpu_name": torch.cuda.get_device_name(0),

    "gpu_memory_gb": round(
        gpu_properties.total_memory / (1024 ** 3), 2
    ),

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


# ============================================================
# TRAINING REPORT
# ============================================================

report = f"""
SENTINELX - LSTM WORLD MODEL V3
================================

ARCHITECTURE
------------
Input features: {INPUT_FEATURES}
Sequence length: {SEQ_LEN}
Hidden size: {HIDDEN_SIZE}
LSTM layers: {NUM_LAYERS}
Output features: {INPUT_FEATURES}

GPU CONFIGURATION
-----------------
PyTorch: {torch.__version__}
CUDA: {torch.version.cuda}
Device: {device}
GPU: {torch.cuda.get_device_name(0)}
GPU memory: {gpu_properties.total_memory / (1024 ** 3):.2f} GB

GPU OPTIMIZATION
----------------
Batch size: {BATCH_SIZE}
Chunk size: {CHUNK_SIZE}
Chunk-based GPU data loading: Enabled
GPU-side batch indexing: Enabled

TRAINING
--------
Training samples: {len(X_train)}
Validation samples: {len(X_val)}
Optimizer: AdamW
Learning rate: {LEARNING_RATE}
Loss: HuberLoss
Maximum epochs: {MAX_EPOCHS}
Patience: {PATIENCE}

RESULTS
-------
Best epoch: {best_epoch}
Best validation Huber: {best_val_loss:.8f}
Training time: {training_seconds:.2f} seconds

TEST DATA
---------
Test data was not loaded or evaluated during training.

OBJECTIVE
---------
Predict the next numerical network state using the
previous {SEQ_LEN} network states.

Regression metrics do not represent attack detection accuracy.
"""

(OUTPUT / "training_report.txt").write_text(
    report,
    encoding="utf-8"
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 75)
print("SENTINELX V3 GPU TRAINING COMPLETED")
print("=" * 75)

print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0))
print("Best epoch:", best_epoch)

print("Best validation Huber:", f"{best_val_loss:.8f}")

print(
    "Total training time:",
    round(training_seconds, 2),
    "seconds"
)

print("Output:", OUTPUT)
print("Checkpoint:", OUTPUT / "best_model.pt")
print("History:", history_file)
print("Test data used: NO")

print("=" * 75)
