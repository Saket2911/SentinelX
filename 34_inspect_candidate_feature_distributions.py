from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/mnt/f/Projects/SentinelX")
MODEL = ROOT / "Models/lstm_world_model_v3_gpu_run2"
OUT = MODEL / "validation_attack_diagnostics_v1"

FEATURES = [
    "CWE Flag Count__max",
    "CWE Flag Count__std",
    "CWE Flag Count__mean",
    "SYN Flag Cnt__mean",
    "SYN Flag Cnt__max",
    "SYN Flag Cnt__std",
    "Fwd PSH Flags__mean",
    "Fwd PSH Flags__max",
    "Fwd PSH Flags__std",
    "Flow IAT Mean__max",
]

def main():
    with np.load(
        MODEL / "explanation_validation_v1/validation_feature_residuals.npz",
        allow_pickle=False
    ) as z:
        print("Residual archive keys:", z.files)
        residuals = z["absolute_residuals"]
        feature_names = z["feature_names"].astype(str)

    targets = np.load(
        ROOT / "Datasets/sentinelx_validation_y_seq10_v3.npy",
        mmap_mode="r"
    )

    metadata = pd.read_csv(
        ROOT / "Datasets/sentinelx_validation_metadata_seq10_v3.csv"
    )

    if len(targets) != len(metadata) or len(residuals) != len(metadata):
        raise ValueError("Target, residual and metadata row counts differ.")

    indices = {name: i for i, name in enumerate(feature_names)}
    labels = metadata["Target Label"].astype(str).to_numpy()

    rows = []

    for feature in FEATURES:
        if feature not in indices:
            print("Missing feature:", feature)
            continue

        j = indices[feature]

        for label in ["Benign", "Infilteration"]:
            mask = labels == label
            values = np.asarray(targets[mask, j], dtype=np.float64)
            errors = residuals[mask, j].astype(np.float64)

            rows.append({
                "Feature": feature,
                "Label": label,
                "Count": len(values),
                "Target Mean": np.mean(values),
                "Target Std": np.std(values),
                "Target Min": np.min(values),
                "Target P25": np.percentile(values, 25),
                "Target Median": np.median(values),
                "Target P75": np.percentile(values, 75),
                "Target P95": np.percentile(values, 95),
                "Target Max": np.max(values),
                "Unique Values": len(np.unique(values)),
                "Mean Abs Residual": np.mean(errors),
                "Median Abs Residual": np.median(errors),
            })

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "candidate_feature_distribution_details.csv", index=False)

    for feature in FEATURES:
        subset = result[result["Feature"] == feature]
        if subset.empty:
            continue

        print("\n" + "=" * 75)
        print(feature)
        print("=" * 75)
        print(subset.to_string(index=False))

    print("\nSaved:", OUT / "candidate_feature_distribution_details.csv")
    print("NOTE: Target values are in transformed model space, not original units.")

if __name__ == "__main__":
    main()
