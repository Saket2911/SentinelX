import numpy as np
import pandas as pd
import torch

from sentinelx.models.lstm_world_model_v3 import SentinelXLSTM
from sentinelx.inference.world_model import predict_next_state
from sentinelx.evaluation.metrics import (
    calculate_metrics,
    calculate_improvements,
)

print("=" * 65)
print("SENTINELX V3 — EDGE CASE BASELINE TESTS")
print("=" * 65)

model = SentinelXLSTM(
    input_size=209,
    hidden_size=128,
    num_layers=1,
    dropout=0.2,
)

model.eval()

# 1. Valid sequence
valid = np.random.default_rng(42).normal(
    size=(10, 209)
).astype(np.float32)

prediction = predict_next_state(model, valid)
assert prediction.shape == (209,)
assert np.isfinite(prediction).all()
print("PASS | Valid single-sequence prediction")

# 2. Valid batch
batch = np.random.default_rng(42).normal(
    size=(4, 10, 209)
).astype(np.float32)

prediction_batch = predict_next_state(model, batch)
assert prediction_batch.shape == (4, 209)
print("PASS | Valid batch prediction")

# 3. Incorrect feature count
try:
    predict_next_state(model, np.zeros((10, 208), dtype=np.float32))
    print("FAIL | Incorrect feature count was accepted")
except ValueError:
    print("PASS | Incorrect feature count rejected")

# 4. Non-finite input
bad = valid.copy()
bad[0, 0] = np.nan

try:
    predict_next_state(model, bad)
    print("FAIL | Non-finite input was accepted")
except ValueError:
    print("PASS | Non-finite input rejected")

# 5. Empty metric arrays
try:
    calculate_metrics(
        np.empty((0, 209)),
        np.empty((0, 209)),
    )
    print("FAIL | Empty metric arrays were accepted")
except ValueError:
    print("PASS | Empty metric arrays rejected")

# 6. Shape mismatch
try:
    calculate_metrics(
        np.zeros((5, 209)),
        np.zeros((5, 208)),
    )
    print("FAIL | Metric shape mismatch was accepted")
except ValueError:
    print("PASS | Metric shape mismatch rejected")

# 7. Zero persistence baseline
improvements = calculate_improvements(
    {"mae": 0.1},
    {"mae": 0.0},
)
assert improvements["mae"] is None
print("PASS | Zero-baseline improvement handled")

# 8. Non-finite Huber delta
try:
    calculate_metrics(
        np.zeros((2, 2)),
        np.ones((2, 2)),
        huber_delta=float("nan"),
    )
    print("FAIL | NaN Huber delta was accepted")
except ValueError:
    print("PASS | NaN Huber delta rejected")

print("\nEDGE-CASE BASELINE TESTS COMPLETED")
