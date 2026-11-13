from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path("/mnt/f/Projects/SentinelX")
DATA = ROOT / "Datasets"
MODEL_DIR = ROOT / "Models/lstm_world_model_v3_gpu_run2"
EXPLAIN_DIR = MODEL_DIR / "explainable_triage_v2"
OUT = MODEL_DIR / "explanation_validation_v1"

BATCH_SIZE = 256

def load_model(checkpoint):
    class SentinelXLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=209,
                hidden_size=128,
                num_layers=1,
                batch_first=True
            )
            self.dropout = nn.Dropout(0.2)
            self.output_layer = nn.Linear(128, 209)

        def forward(self, x):
            output, _ = self.lstm(x)
            last_state = output[:, -1, :]
            last_state = self.dropout(last_state)
            return self.output_layer(last_state)

    model = SentinelXLSTM()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def predict(model, X):
    predictions = np.empty(
        (len(X), 209),
        dtype=np.float32
    )

    with torch.inference_mode():
        for start in range(0, len(X), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(X))

            batch_np = np.array(
                X[start:end],
                dtype=np.float32,
                copy=True
            )
            batch = torch.from_numpy(batch_np)

            predictions[start:end] = (
                model(batch).numpy()
            )

            if end % 5000 < BATCH_SIZE or end == len(X):
                print(f"Predicted {end:,}/{len(X):,}")

    return predictions

def feature_summary(residuals, features, metadata, split, label):
    mask = metadata["Target Label"].astype(str).to_numpy() == label

    if not mask.any():
        print(f"WARNING: No {label} observations in {split}")
        return None

    values = residuals[mask]

    return pd.DataFrame({
        "Feature": features,
        "Mean Absolute Residual": np.mean(values, axis=0),
        "Median Absolute Residual": np.median(values, axis=0),
        "P95 Absolute Residual": np.percentile(values, 95, axis=0),
        "Sample Count": len(values),
        "Split": split,
        "Label": label
    })

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("SENTINELX STEP 31C — VALIDATION/TEST EXPLANATION COMPARISON")
    print("=" * 72)

    checkpoint = torch.load(
        MODEL_DIR / "best_model.pt",
        map_location="cpu",
        weights_only=False
    )

    features = np.asarray(
        checkpoint["feature_names"],
        dtype=str
    )

    model = load_model(checkpoint)

    # Validation data
    val_X = np.load(
        DATA / "sentinelx_validation_X_seq10_v3.npy",
        mmap_mode="r"
    )
    val_y = np.load(
        DATA / "sentinelx_validation_y_seq10_v3.npy",
        mmap_mode="r"
    )
    val_metadata = pd.read_csv(
        DATA / "sentinelx_validation_metadata_seq10_v3.csv"
    )

    print("\nValidation sequences:", val_X.shape)
    print("Validation targets:", val_y.shape)

    val_predictions = predict(model, val_X)

    val_residuals = np.asarray(val_y) - val_predictions
    val_abs = np.abs(val_residuals)

    np.savez_compressed(
        OUT / "validation_feature_residuals.npz",
        signed_residuals=val_residuals,
        absolute_residuals=val_abs,
        feature_names=features
    )

    # Test residuals generated from the verified replay.
    with np.load(
        EXPLAIN_DIR / "feature_residuals.npz",
        allow_pickle=False
    ) as z:
        test_residuals = z["signed_residuals"]
        test_abs = z["absolute_residuals"]
        test_features = z["feature_names"].astype(str)

    test_metadata = pd.read_csv(
        DATA / "sentinelx_test_metadata_seq10_v3.csv"
    )

    if not np.array_equal(features, test_features):
        raise ValueError("Validation/test feature order mismatch.")

    if len(val_metadata) != len(val_abs):
        raise ValueError("Validation metadata alignment failure.")

    if len(test_metadata) != len(test_abs):
        raise ValueError("Test metadata alignment failure.")

    # Label-stratified summaries are diagnostic only.
    # Labels are not used to make model or triage decisions.
    summaries = []

    groups = [
        (val_abs, val_metadata, "Validation", "Benign"),
        (val_abs, val_metadata, "Validation", "Infilteration"),
        (test_abs, test_metadata, "Test", "Benign"),
        (test_abs, test_metadata, "Test", "Bot")
    ]

    for residuals, metadata, split, label in groups:
        result = feature_summary(
            residuals, features, metadata, split, label
        )
        if result is not None:
            summaries.append(result)

    combined = pd.concat(summaries, ignore_index=True)

    combined.to_csv(
        OUT / "feature_residual_group_summary.csv",
        index=False
    )

    # Direct benign-to-benign comparison.
    val_benign = combined[
        (combined["Split"] == "Validation") &
        (combined["Label"] == "Benign")
    ].set_index("Feature")

    test_benign = combined[
        (combined["Split"] == "Test") &
        (combined["Label"] == "Benign")
    ].set_index("Feature")

    comparison = pd.DataFrame(index=features)

    comparison["Validation Benign MAE"] = (
        val_benign["Mean Absolute Residual"]
    )
    comparison["Test Benign MAE"] = (
        test_benign["Mean Absolute Residual"]
    )
    comparison["MAE Difference (Validation - Test)"] = (
        comparison["Validation Benign MAE"] -
        comparison["Test Benign MAE"]
    )
    comparison["Absolute MAE Difference"] = (
        comparison["MAE Difference (Validation - Test)"].abs()
    )

    comparison = comparison.sort_values(
        "Absolute MAE Difference",
        ascending=False
    )

    comparison.to_csv(
        OUT / "benign_feature_residual_comparison.csv"
    )

    # Split-level summary.
    split_summary = {}

    for residuals, metadata, split in [
        (val_abs, val_metadata, "Validation"),
        (test_abs, test_metadata, "Test")
    ]:
        split_summary[split] = {
            "observations": int(len(residuals)),
            "mean_mae_model_space": float(residuals.mean()),
            "labels_for_diagnostic_grouping": {
                str(k): int(v)
                for k, v in metadata["Target Label"]
                    .value_counts().items()
            }
        }

    summary = {
        "step": "31C",
        "purpose": "Compare feature-level prediction residuals across splits",
        "model_checkpoint_epoch": int(checkpoint["epoch"]),
        "feature_count": int(len(features)),
        "validation_test_benign_comparison": True,
        "labels_used_for_diagnostic_grouping_only": True,
        "labels_used_for_triage_decisions": False,
        "threshold_tuning_performed": False,
        "split_summary": split_summary,
        "outputs": [
            "validation_feature_residuals.npz",
            "feature_residual_group_summary.csv",
            "benign_feature_residual_comparison.csv"
        ],
        "interpretation": (
            "Residual differences describe model prediction behavior "
            "across dataset splits. They do not establish attack indicators "
            "or justify a new detection threshold."
        )
    }

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print("VALIDATION/TEST COMPARISON COMPLETE")
    print("=" * 72)

    print("\nSplit summary:")
    print(json.dumps(split_summary, indent=2))

    print("\nLargest benign feature residual differences:")
    print(comparison.head(15).to_string())

    print("\nOutputs:", OUT.relative_to(ROOT))
    print("=" * 72)

if __name__ == "__main__":
    main()
