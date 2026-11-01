import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------
# 1. Paths and configuration
# ---------------------------------------------------------

ROOT = Path(r"F:\Projects\SentinelX")
DATA_DIR = ROOT / "Datasets"
OUTPUT_DIR = ROOT / "Models" / "lstm_world_model_v1"

SEQUENCE_LENGTH = 10
INPUT_FEATURES = 206
HIDDEN_SIZE = 128
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_EPOCHS = 30
PATIENCE = 6
SEED = 42

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Output directory already exists: {OUTPUT_DIR}\n"
        "To protect existing results, this script will not overwrite it."
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.cuda.manual_seed_all(SEED)
else:
    DEVICE = torch.device("cpu")

torch.set_num_threads(4)

print("=" * 65)
print("SENTINELX - LSTM WORLD MODEL TRAINING")
print("=" * 65)
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3), 2
        ),
        "GB"
    )


# ---------------------------------------------------------
# 2. Load arrays using memory mapping
# ---------------------------------------------------------

def load_split(split):
    x_path = DATA_DIR / f"sentinelx_{split}_X.npy"
    y_path = DATA_DIR / f"sentinelx_{split}_y.npy"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing {split} sequence files.")

    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")

    expected_x = (SEQUENCE_LENGTH, INPUT_FEATURES)
    expected_y = (INPUT_FEATURES,)

    if X.ndim != 3 or X.shape[1:] != expected_x:
        raise ValueError(f"Unexpected {split} X shape: {X.shape}")

    if y.ndim != 2 or y.shape[1:] != expected_y:
        raise ValueError(f"Unexpected {split} y shape: {y.shape}")

    if len(X) != len(y):
        raise ValueError(f"{split}: X/y row count mismatch.")

    print(f"{split.title()} X: {X.shape}")
    print(f"{split.title()} y: {y.shape}")

    return X, y


X_train, y_train = load_split("train")
X_val, y_val = load_split("validation")

print("\nTest data will remain untouched during training.")


# ---------------------------------------------------------
# 3. Define the LSTM World Model
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

        # Use the final time step's hidden representation.
        final_state = sequence_output[:, -1, :]

        final_state = self.dropout(final_state)

        predicted_next_state = self.output_layer(final_state)

        return predicted_next_state


model = SentinelXLSTM().to(DEVICE)

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
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("\nTrainable parameters:", f"{parameter_count:,}")
print("Model initialized successfully.")


# ---------------------------------------------------------
# 4. Batch conversion
# ---------------------------------------------------------

def make_tensor_batch(array, indices):
    # Copy only the current mini-batch into RAM.
    batch = np.array(array[indices], dtype=np.float32, copy=True)

    return torch.from_numpy(batch).to(DEVICE)


# ---------------------------------------------------------
# 5. Training and evaluation functions
# ---------------------------------------------------------

def train_one_epoch():
    model.train()

    permutation = np.random.permutation(len(X_train))

    total_loss = 0.0
    total_elements = 0

    for start in range(0, len(permutation), BATCH_SIZE):
        indices = permutation[start:start + BATCH_SIZE]

        xb = make_tensor_batch(X_train, indices)
        yb = make_tensor_batch(y_train, indices)

        optimizer.zero_grad(set_to_none=True)

        predictions = model(xb)
        loss = criterion(predictions, yb)

        loss.backward()

        # Keep gradients stable.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        elements = yb.numel()
        total_loss += loss.item() * elements
        total_elements += elements

    return total_loss / total_elements


@torch.no_grad()
def evaluate():
    model.eval()

    total_loss = 0.0
    total_absolute_error = 0.0
    total_squared_error = 0.0
    total_elements = 0

    for start in range(0, len(X_val), BATCH_SIZE):
        indices = np.arange(
            start,
            min(start + BATCH_SIZE, len(X_val))
        )

        xb = make_tensor_batch(X_val, indices)
        yb = make_tensor_batch(y_val, indices)

        predictions = model(xb)

        loss = criterion(predictions, yb)

        difference = predictions - yb

        elements = yb.numel()

        total_loss += loss.item() * elements
        total_absolute_error += difference.abs().sum().item()
        total_squared_error += (
            difference.square().sum().item()
        )

        total_elements += elements

    average_loss = total_loss / total_elements
    mae = total_absolute_error / total_elements
    rmse = (total_squared_error / total_elements) ** 0.5

    return average_loss, mae, rmse


# ---------------------------------------------------------
# 6. Training history
# ---------------------------------------------------------

history_path = OUTPUT_DIR / "training_history.csv"

with history_path.open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "epoch",
        "train_huber_loss",
        "validation_huber_loss",
        "validation_mae",
        "validation_rmse",
        "learning_rate",
        "epoch_seconds"
    ])


# ---------------------------------------------------------
# 7. Training loop with best-checkpoint saving
# ---------------------------------------------------------

best_validation_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0

checkpoint_path = OUTPUT_DIR / "best_model.pt"

print("\nStarting training...\n")

training_start = time.time()

for epoch in range(1, MAX_EPOCHS + 1):

    epoch_start = time.time()

    train_loss = train_one_epoch()

    val_loss, val_mae, val_rmse = evaluate()

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]
    epoch_seconds = time.time() - epoch_start

    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"Val MAE: {val_mae:.6f} | "
        f"Val RMSE: {val_rmse:.6f} | "
        f"Time: {epoch_seconds:.1f}s"
    )

    with history_path.open("a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            epoch,
            train_loss,
            val_loss,
            val_mae,
            val_rmse,
            current_lr,
            epoch_seconds
        ])

    if val_loss < best_validation_loss:
        best_validation_loss = val_loss
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_huber_loss": val_loss,
            "validation_mae": val_mae,
            "validation_rmse": val_rmse,
            "sequence_length": SEQUENCE_LENGTH,
            "input_features": INPUT_FEATURES,
            "hidden_size": HIDDEN_SIZE,
            "feature_names_file": str(
                DATA_DIR / "sentinelx_sequence_feature_names.txt"
            )
        }, checkpoint_path)

        print("  Best model checkpoint saved.")

    else:
        epochs_without_improvement += 1
        print(
            "  No validation improvement:",
            epochs_without_improvement,
            "/",
            PATIENCE
        )

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping triggered.")
        break


# ---------------------------------------------------------
# 8. Final training report
# ---------------------------------------------------------

total_seconds = time.time() - training_start

report_path = OUTPUT_DIR / "training_report.txt"

report = f"""
SENTINELX - LSTM WORLD MODEL TRAINING REPORT

Model: SentinelXLSTM
Framework: PyTorch {torch.__version__}
Device: {DEVICE}

Architecture:
Input sequence: {SEQUENCE_LENGTH} x {INPUT_FEATURES}
LSTM hidden units: {HIDDEN_SIZE}
LSTM layers: 1
Dropout: 0.2
Output features: {INPUT_FEATURES}

Training:
Training sequences: {len(X_train)}
Validation sequences: {len(X_val)}
Batch size: {BATCH_SIZE}
Maximum epochs: {MAX_EPOCHS}
Early stopping patience: {PATIENCE}
Optimizer: AdamW
Loss: Huber Loss

Results:
Best epoch: {best_epoch}
Best validation Huber loss: {best_validation_loss:.8f}
Total training time: {total_seconds:.2f} seconds

The test dataset was not used during training or model selection.

Objective:
Predict the next network state from the previous
{SEQUENCE_LENGTH} network states.

Checkpoint:
{checkpoint_path}
"""

report_path.write_text(report, encoding="utf-8")

print("\n" + "=" * 65)
print("TRAINING COMPLETED")
print("=" * 65)
print("Best epoch:", best_epoch)
print("Best validation loss:", f"{best_validation_loss:.8f}")
print("Training time:", f"{total_seconds:.1f} seconds")
print("Checkpoint:", checkpoint_path)
print("History:", history_path)
print("Report:", report_path)
print("\nTest data remains untouched.")
