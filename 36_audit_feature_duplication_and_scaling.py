from pathlib import Path
import numpy as np
import joblib

ROOT = Path("/mnt/f/Projects/SentinelX")
DATA = ROOT / "Datasets"
MODEL = ROOT / "Models/lstm_world_model_v3_gpu_run2"

PAIRS = [
    ("SYN Flag Cnt__mean", "Fwd PSH Flags__mean"),
    ("SYN Flag Cnt__max", "Fwd PSH Flags__max"),
    ("SYN Flag Cnt__std", "Fwd PSH Flags__std"),
]

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

def main():
    with np.load(
        MODEL / "explanation_validation_v1/validation_feature_residuals.npz",
        allow_pickle=False
    ) as z:
        names = z["feature_names"].astype(str)

    indices = {name: i for i, name in enumerate(names)}

    scaler_path = DATA / "sentinelx_standard_scaler_v3.joblib"
    scaler = joblib.load(scaler_path)

    print("=" * 75)
    print("STEP 31G — CROSS-SPLIT DUPLICATION AND SCALING AUDIT")
    print("=" * 75)

    print("\nSCALER TYPE:", type(scaler).__name__)

    if hasattr(scaler, "scale_"):
        print("\nSCALER PARAMETERS")
        for feature in FEATURES:
            i = indices[feature]
            print(
                f"{feature:30s} "
                f"mean={scaler.mean_[i]:.8g} "
                f"scale={scaler.scale_[i]:.8g} "
                f"variance={scaler.var_[i]:.8g}"
            )
    else:
        print("Scaler does not expose scale_ and var_ attributes.")

    arrays = {}

    for split in ["train", "validation", "test"]:
        path = DATA / f"sentinelx_{split}_y_seq10_v3.npy"
        arrays[split] = np.load(path, mmap_mode="r")

    print("\n" + "=" * 75)
    print("SYN / FORWARD PSH DUPLICATION")
    print("=" * 75)

    for split, values in arrays.items():
        print(f"\n{split.upper()}")

        for first, second in PAIRS:
            a = np.asarray(values[:, indices[first]])
            b = np.asarray(values[:, indices[second]])

            print(
                f"{first} == {second}: "
                f"exact={np.array_equal(a, b)}, "
                f"max_difference={np.max(np.abs(a-b)):.10g}"
            )

    print("\n" + "=" * 75)
    print("CWE NON-BASELINE PREVALENCE")
    print("=" * 75)

    for split, values in arrays.items():
        print(f"\n{split.upper()}")

        for feature in [
            "CWE Flag Count__max",
            "CWE Flag Count__std",
            "CWE Flag Count__mean"
        ]:
            v = np.asarray(values[:, indices[feature]], dtype=np.float64)
            baseline = float(np.median(v))
            nonbaseline = ~np.isclose(v, baseline, rtol=0, atol=1e-7)

            print(
                f"{feature:25s} "
                f"baseline={baseline:.8g}, "
                f"nonbaseline={nonbaseline.sum():,}/{len(v):,} "
                f"({nonbaseline.mean()*100:.4f}%), "
                f"max={v.max():.6g}"
            )

    print("\nAudit completed.")
    print("=" * 75)

if __name__ == "__main__":
    main()
