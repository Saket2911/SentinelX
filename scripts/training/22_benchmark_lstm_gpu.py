
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# SENTINELX V3 — GPU PERFORMANCE BENCHMARK
# Does NOT train or save the model
# ============================================================

ROOT = Path(r"F:\Projects\SentinelX")
DATA = ROOT / "Datasets"

TRAIN_X = DATA / "sentinelx_train_X_seq10_v3.npy"
TRAIN_Y = DATA / "sentinelx_train_y_seq10_v3.npy"

SEQ_LEN = 10
INPUT_FEATURES = 209
HIDDEN_SIZE = 128
NUM_LAYERS = 1

BATCH_SIZE = 32
BENCHMARK_BATCHES = 20


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("SENTINELX GPU BENCHMARK")
print("=" * 70)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device:", device)

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
# LOAD ARRAYS
# ============================================================

if not TRAIN_X.exists():
    raise FileNotFoundError(TRAIN_X)

if not TRAIN_Y.exists():
    raise FileNotFoundError(TRAIN_Y)

print("\nLoading arrays using memory mapping...")

X = np.load(TRAIN_X, mmap_mode="r")
y = np.load(TRAIN_Y, mmap_mode="r")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("X dtype:", X.dtype)
print("y dtype:", y.dtype)

if X.shape[1:] != (SEQ_LEN, INPUT_FEATURES):
    raise ValueError(f"Unexpected X shape: {X.shape}")

if y.shape[1:] != (INPUT_FEATURES,):
    raise ValueError(f"Unexpected y shape: {y.shape}")


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

        self.output_layer = nn.Linear(
            HIDDEN_SIZE,
            INPUT_FEATURES
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        last_state = output[:, -1, :]
        return self.output_layer(last_state)


model = SentinelXLSTM().to(device)

criterion = nn.HuberLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)

print("\nModel created.")
print("Parameters:", sum(p.numel() for p in model.parameters()))


# ============================================================
# BENCHMARK
# ============================================================

data_times = []
compute_times = []

print("\nStarting benchmark...")
print("Benchmark batches:", BENCHMARK_BATCHES)
print("Batch size:", BATCH_SIZE)

model.train()

for batch_number in range(BENCHMARK_BATCHES):

    start_index = batch_number * BATCH_SIZE
    end_index = min(start_index + BATCH_SIZE, len(X))

    indices = np.arange(start_index, end_index)

    # --------------------------------------------------------
    # Batch preparation and transfer
    # --------------------------------------------------------

    data_start = time.perf_counter()

    xb_np = np.array(
        X[indices],
        dtype=np.float32,
        copy=True
    )

    yb_np = np.array(
        y[indices],
        dtype=np.float32,
        copy=True
    )

    xb = torch.from_numpy(xb_np).to(device)
    yb = torch.from_numpy(yb_np).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    data_elapsed = time.perf_counter() - data_start

    # --------------------------------------------------------
    # GPU forward + backward + optimizer
    # --------------------------------------------------------

    compute_start = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)

    predictions = model(xb)
    loss = criterion(predictions, yb)

    loss.backward()
    optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    compute_elapsed = time.perf_counter() - compute_start

    data_times.append(data_elapsed)
    compute_times.append(compute_elapsed)

    print(
        f"Batch {batch_number + 1:02d}/{BENCHMARK_BATCHES} | "
        f"Data: {data_elapsed:.4f}s | "
        f"Compute: {compute_elapsed:.4f}s | "
        f"Loss: {loss.item():.5f}",
        flush=True
    )


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BENCHMARK RESULTS")
print("=" * 70)

print(
    "Average batch preparation + transfer:",
    f"{np.mean(data_times):.4f} seconds"
)

print(
    "Average forward + backward + optimizer:",
    f"{np.mean(compute_times):.4f} seconds"
)

total_time = np.mean(data_times) + np.mean(compute_times)

print(
    "Estimated time per training batch:",
    f"{total_time:.4f} seconds"
)

print(
    "Estimated time per epoch:",
    f"{total_time * ((len(X) + BATCH_SIZE - 1) // BATCH_SIZE):.1f} seconds"
)

if device.type == "cuda":
    print(
        "Peak GPU memory:",
        round(
            torch.cuda.max_memory_allocated() / (1024 ** 2),
            1
        ),
        "MB"
    )

print("\nBenchmark completed.")
print("No model checkpoint was saved.")
