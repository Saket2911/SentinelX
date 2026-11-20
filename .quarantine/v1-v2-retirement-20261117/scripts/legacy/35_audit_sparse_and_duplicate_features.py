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
]

PAIRS = [
    ("SYN Flag Cnt__mean", "Fwd PSH Flags__mean"),
    ("SYN Flag Cnt__max", "Fwd PSH Flags__max"),
    ("SYN Flag Cnt__std", "Fwd PSH Flags__std"),
]

def main():
    with np.load(
        MODEL / "explanation_validation_v1/validation_feature_residuals.npz",
        allow_pickle=False
    ) as z:
        errors = z["absolute_residuals"]
        names = z["feature_names"].astype(str)

    targets = np.load(
        ROOT / "Datasets/sentinelx_validation_y_seq10_v3.npy",
        mmap_mode="r"
    )

    metadata = pd.read_csv(
        ROOT / "Datasets/sentinelx_validation_metadata_seq10_v3.csv"
    )

    if len(targets) != len(errors) or len(targets) != len(metadata):
        raise ValueError("Array and metadata row counts do not match.")

    index = {name: i for i, name in enumerate(names)}
    labels = metadata["Target Label"].astype(str).to_numpy()
    rows = []

    print("=" * 75)
    print("STEP 31F — SPARSE FEATURE AND DUPLICATE SIGNAL AUDIT")
    print("=" * 75)

    for feature in FEATURES:
        j = index[feature]
        values = np.asarray(targets[:, j], dtype=np.float64)
        residual = errors[:, j].astype(np.float64)

        baseline = float(pd.Series(values).mode().iloc[0])

        for label in ["Benign", "Infilteration"]:
            mask = labels == label
            v = values[mask]
            e = residual[mask]

            nonbaseline = ~np.isclose(v, baseline, rtol=0, atol=1e-7)

            rows.append({
                "Feature": feature,
                "Label": label,
                "Baseline Value": baseline,
                "Total Samples": len(v),
                "Nonbaseline Samples": int(nonbaseline.sum()),
                "Nonbaseline Percentage": float(nonbaseline.mean() * 100),
                "Mean Error All": float(e.mean()),
                "Mean Error Baseline": float(e[~nonbaseline].mean()),
                "Mean Error Nonbaseline": (
                    float(e[nonbaseline].mean())
                    if nonbaseline.any() else 0.0
                ),
                "Median Error": float(np.median(e)),
            })

    report = pd.DataFrame(rows)
    report.to_csv(OUT / "sparse_feature_error_audit.csv", index=False)

    print("\nSPARSE FEATURE RESULTS")
    print(report.to_string(index=False))

    print("\n" + "=" * 75)
    print("SYN / FORWARD PSH COLUMN EQUALITY")
    print("=" * 75)

    pair_rows = []

    for first, second in PAIRS:
        a = np.asarray(targets[:, index[first]], dtype=np.float64)
        b = np.asarray(targets[:, index[second]], dtype=np.float64)

        ea = errors[:, index[first]].astype(np.float64)
        eb = errors[:, index[second]].astype(np.float64)

        exact_targets = np.array_equal(a, b)
        exact_errors = np.array_equal(ea, eb)

        result = {
            "Feature A": first,
            "Feature B": second,
            "Targets Exactly Equal": exact_targets,
            "Residuals Exactly Equal": exact_errors,
            "Maximum Target Difference": float(np.max(np.abs(a - b))),
            "Maximum Residual Difference": float(np.max(np.abs(ea - eb))),
        }
        pair_rows.append(result)
        print(result)

    pd.DataFrame(pair_rows).to_csv(
        OUT / "syn_psh_duplicate_audit.csv", index=False
    )

    print("\nSaved reports to:", OUT)
    print("=" * 75)

if __name__ == "__main__":
    main()
