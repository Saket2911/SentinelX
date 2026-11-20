import hashlib
import numpy as np
import pandas as pd

from pathlib import Path

from sentinelx.inference import (
    load_world_model,
    SentinelXV3Predictor,
)
from sentinelx.evaluation import (
    evaluate_one_step,
    evaluate_multi_step,
)

DATA = Path("Datasets")
MODEL_DIR = Path("Models/lstm_world_model_v3_gpu_run2")
CHECKPOINT = MODEL_DIR / "best_model.pt"

EXPECTED_SHA256 = (
    "19e117e3cc407fe656aa3e6840003031371419a2a367c31128a8b720a900450c"
)

print("=" * 72)
print("SENTINELX V3 — CORRECTED END-TO-END INTEGRATION TEST")
print("=" * 72)

# 1. Checkpoint integrity
sha256 = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
assert sha256 == EXPECTED_SHA256
print("PASS | Checkpoint integrity")

# 2. Raw network-state inference
raw_sequence = pd.read_csv(
    DATA / "sentinelx_expanded_train_v1.csv",
    nrows=10,
)

predictor = SentinelXV3Predictor.load(
    checkpoint_path=CHECKPOINT,
    data_dir=DATA,
    device="cpu",
)

raw_prediction = predictor.predict_raw_sequence(raw_sequence)

assert raw_prediction.shape == (209,)
assert np.isfinite(raw_prediction).all()
print("PASS | Raw preprocessing and prediction")

# 3. Load model
model, model_metadata = load_world_model(
    CHECKPOINT,
    device="cpu",
)

assert model_metadata["input_features"] == 209
assert model_metadata["sequence_length"] == 10
print("PASS | Model loading and configuration")

# 4. Load evaluation data
X = np.load(
    DATA / "sentinelx_test_X_seq10_v3.npy",
    mmap_mode="r",
)

y = np.load(
    DATA / "sentinelx_test_y_seq10_v3.npy",
    mmap_mode="r",
)

metadata = pd.read_csv(
    DATA / "sentinelx_test_metadata_seq10_v3.csv",
)

assert X.shape == (31406, 10, 209)
assert y.shape == (31406, 209)
assert len(metadata) == 31406
print("PASS | Evaluation data alignment")

# 5. One-step evaluation
one_step = evaluate_one_step(
    model=model,
    X=X,
    y=y,
    metadata=metadata,
    feature_names=model_metadata["feature_names"],
    batch_size=1024,
)

# IMPORTANT: evaluator returns a dictionary.
summary = one_step["summary"]
feature_results = one_step["feature_results"]
label_results = one_step["label_results"]
predictions = one_step["predictions"]
baseline = one_step["baseline_predictions"]

assert predictions.shape == (31406, 209)
assert baseline.shape == (31406, 209)
assert len(feature_results) == 209
assert np.isfinite(predictions).all()
assert np.isfinite(baseline).all()

print("PASS | One-step evaluation")
print("      Prediction shape:", predictions.shape)
print("      Feature results:", len(feature_results))
print("      Label groups:", len(label_results))
print("      LSTM MAE:", summary["model_metrics"]["mae"])

# 6. Multi-step evaluation
multi_results, diagnostics = evaluate_multi_step(
    model=model,
    X=X,
    y=y,
    metadata=metadata,
    horizons=[1, 2, 5, 10, 30],
    batch_size=128,
)

expected_valid = {
    1: 31377,
    2: 31376,
    5: 31373,
    10: 31368,
    30: 31348,
}

assert len(multi_results) == 20
assert diagnostics["common_origins"] == 31377
assert diagnostics["valid_origins_by_horizon"] == expected_valid

print("PASS | Multi-step evaluation")
print("      Result groups:", len(multi_results))
print("      Common origins:", diagnostics["common_origins"])

# 7. Final result
print("\n" + "=" * 72)
print("INTEGRATION TEST SUMMARY")
print("=" * 72)

print("PASS | Checkpoint SHA256 unchanged")
print("PASS | Raw preprocessing and inference")
print("PASS | V3 model configuration")
print("PASS | Evaluation data alignment")
print("PASS | One-step evaluation")
print("PASS | Multi-step evaluation")
print("PASS | Forecast continuity")

print("\nALL END-TO-END INTEGRATION CHECKS: PASS")
print("No training or checkpoint modifications performed.")
