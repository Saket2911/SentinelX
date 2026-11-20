from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(r"F:\Projects\SentinelX")
DATA = ROOT / "Datasets"
MODEL_DIR = ROOT / "Models" / "lstm_world_model_v2"

CHECKPOINT_PATH = MODEL_DIR / "best_model.pt"
FEATURE_PATH = DATA / "sentinelx_model_features_v1.json"

OUTPUT_DIR = MODEL_DIR / "feature_residual_analysis_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32


class SentinelXLSTM(nn.Module):
    def __init__(self, input_features, hidden_size, num_layers, dropout):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_layer = nn.Linear(hidden_size, input_features)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.output_layer(output[:, -1, :])


def predict(model, X, device):
    predictions = np.empty(
        (len(X), X.shape[-1]),
        dtype=np.float32
    )

    model.eval()

    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(X))

            batch = torch.from_numpy(
                np.array(X[start:end], dtype=np.float32, copy=True)
            ).to(device)

            predictions[start:end] = model(batch).cpu().numpy()

    return predictions


def analyze_split(model, split, feature_names, device):
    print(f"\nAnalyzing {split.upper()}...")

    X_path = DATA / f"sentinelx_{split}_X_seq10_v1.npy"
    Y_path = DATA / f"sentinelx_{split}_y_next_v1.npy"
    META_path = DATA / f"sentinelx_{split}_sequence_metadata_v1.csv"

    for path in [X_path, Y_path, META_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    X = np.load(X_path, mmap_mode="r")
    Y = np.load(Y_path, mmap_mode="r")
    metadata = pd.read_csv(META_path)

    if len(X) != len(Y) or len(X) != len(metadata):
        raise ValueError("Sequence, target, and metadata counts mismatch.")

    labels = metadata["Target Label"].astype(str).to_numpy()

    predictions = predict(model, X, device)
    targets = np.array(Y, dtype=np.float32, copy=True)

    absolute_errors = np.abs(predictions - targets)

    rows = []

    for label in sorted(np.unique(labels)):
        mask = labels == label
        label_errors = absolute_errors[mask]

        feature_mae = label_errors.mean(axis=0)

        for i, feature in enumerate(feature_names):
            rows.append({
                "Split": split,
                "Label": label,
                "Samples": int(mask.sum()),
                "Feature": feature,
                "Feature_MAE": float(feature_mae[i]),
            })

        print(
            f"  {label}: {int(mask.sum()):,} samples | "
            f"mean overall MAE={label_errors.mean():.6f}"
        )

    result = pd.DataFrame(rows)

    output_path = OUTPUT_DIR / f"{split}_feature_residuals.csv"
    result.to_csv(output_path, index=False)

    print("Saved:", output_path)

    return result


def compare_labels(results, split, label_a, label_b):
    subset = results[results["Split"] == split]

    a = subset[subset["Label"] == label_a][
        ["Feature", "Feature_MAE"]
    ].rename(columns={"Feature_MAE": f"{label_a}_MAE"})

    b = subset[subset["Label"] == label_b][
        ["Feature", "Feature_MAE"]
    ].rename(columns={"Feature_MAE": f"{label_b}_MAE"})

    if a.empty or b.empty:
        print(
            f"\nCannot compare {label_a} vs {label_b} "
            f"on {split}: one label is missing."
        )
        return

    comparison = a.merge(b, on="Feature")

    comparison["MAE_Difference"] = (
        comparison[f"{label_b}_MAE"] -
        comparison[f"{label_a}_MAE"]
    )

    comparison["Relative_Difference_Percent"] = (
        comparison["MAE_Difference"] /
        comparison[f"{label_a}_MAE"].replace(0, np.nan) * 100
    )

    comparison["Absolute_Difference"] = (
        comparison["MAE_Difference"].abs()
    )

    comparison = comparison.sort_values(
        "Absolute_Difference",
        ascending=False
    )

    path = OUTPUT_DIR / (
        f"{split}_{label_a}_vs_{label_b}_feature_comparison.csv"
    )

    comparison.to_csv(path, index=False)

    print(f"\nTop 15 feature residual differences: {split}")
    print(
        comparison.head(15).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print("\nSaved comparison:", path)


def main():
    print("=" * 75)
    print("SENTINELX V2 — FEATURE-WISE RESIDUAL ANALYSIS")
    print("=" * 75)

    with open(FEATURE_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

    if len(feature_names) != checkpoint["input_features"]:
        raise ValueError(
            f"Feature metadata has {len(feature_names)} features, "
            f"checkpoint expects {checkpoint['input_features']}."
        )

    if checkpoint["input_features"] != 212:
        raise ValueError("Expected the 212-feature V2 checkpoint.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SentinelXLSTM(
        checkpoint["input_features"],
        checkpoint["hidden_size"],
        checkpoint["num_layers"],
        checkpoint["dropout"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("Checkpoint:", CHECKPOINT_PATH)
    print("Features:", len(feature_names))
    print("Device:", device)
    print("Checkpoint epoch:", checkpoint["epoch"])

    validation = analyze_split(
        model, "validation", feature_names, device
    )

    test = analyze_split(
        model, "test", feature_names, device
    )

    results = pd.concat(
        [validation, test],
        ignore_index=True
    )

    compare_labels(
        results, "validation", "Benign", "Infilteration"
    )

    compare_labels(
        results, "test", "Benign", "Bot"
    )

    print("\n" + "=" * 75)
    print("FEATURE RESIDUAL ANALYSIS COMPLETED")
    print("Model checkpoint was not modified.")
    print("Output directory:", OUTPUT_DIR)
    print("=" * 75)


if __name__ == "__main__":
    main()
