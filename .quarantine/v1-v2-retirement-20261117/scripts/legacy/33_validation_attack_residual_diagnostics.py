from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path("/mnt/f/Projects/SentinelX")
MODEL = ROOT / "Models/lstm_world_model_v3_gpu_run2"
SOURCE = MODEL / "explanation_validation_v1"
OUT = MODEL / "validation_attack_diagnostics_v1"

def ks_statistic(a, b):
    a = np.sort(a)
    b = np.sort(b)

    combined = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, combined, side="right") / len(a)
    cdf_b = np.searchsorted(b, combined, side="right") / len(b)

    return float(np.max(np.abs(cdf_a - cdf_b)))

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("SENTINELX STEP 31D — VALIDATION ATTACK RESIDUAL DIAGNOSTICS")
    print("=" * 72)

    with np.load(
        SOURCE / "validation_feature_residuals.npz",
        allow_pickle=False
    ) as z:
        residuals = z["absolute_residuals"]
        features = z["feature_names"].astype(str)

    metadata = pd.read_csv(
        ROOT / "Datasets/sentinelx_validation_metadata_seq10_v3.csv"
    )

    if len(metadata) != len(residuals):
        raise ValueError("Metadata and residual row counts differ.")

    labels = metadata["Target Label"].astype(str).to_numpy()
    benign_mask = labels == "Benign"
    attack_mask = labels == "Infilteration"

    benign = residuals[benign_mask]
    attack = residuals[attack_mask]

    print("Validation observations:", len(metadata))
    print("Benign observations:", len(benign))
    print("Infilteration observations:", len(attack))
    print("Features:", len(features))

    if len(benign) == 0 or len(attack) == 0:
        raise ValueError("Required validation labels are missing.")

    rows = []

    for i, feature in enumerate(features):
        b = benign[:, i].astype(np.float64)
        a = attack[:, i].astype(np.float64)

        b_mean = float(np.mean(b))
        a_mean = float(np.mean(a))
        b_std = float(np.std(b))
        a_std = float(np.std(a))

        pooled_std = np.sqrt((b_std**2 + a_std**2) / 2)
        smd = (
            (a_mean - b_mean) / pooled_std
            if pooled_std > 0 else 0.0
        )

        rows.append({
            "Feature": feature,
            "Benign Mean Abs Residual": b_mean,
            "Infilteration Mean Abs Residual": a_mean,
            "Mean Difference (Attack - Benign)": a_mean - b_mean,
            "Benign Median Abs Residual": float(np.median(b)),
            "Infilteration Median Abs Residual": float(np.median(a)),
            "Benign P95 Abs Residual": float(np.percentile(b, 95)),
            "Infilteration P95 Abs Residual": float(np.percentile(a, 95)),
            "KS Statistic": ks_statistic(b, a),
            "Standardized Mean Difference": float(smd),
            "Absolute SMD": abs(float(smd))
        })

    result = pd.DataFrame(rows)

    result.sort_values(
        "KS Statistic", ascending=False
    ).to_csv(OUT / "features_ranked_by_ks.csv", index=False)

    result.sort_values(
        "Absolute SMD", ascending=False
    ).to_csv(OUT / "features_ranked_by_smd.csv", index=False)

    result.sort_values(
        "Mean Difference (Attack - Benign)",
        ascending=False
    ).to_csv(OUT / "features_attack_higher_error.csv", index=False)

    result.sort_values(
        "Mean Difference (Attack - Benign)",
        ascending=True
    ).to_csv(OUT / "features_benign_higher_error.csv", index=False)

    summary = {
        "step": "31D",
        "dataset_split": "validation_only",
        "benign_count": int(len(benign)),
        "infilteration_count": int(len(attack)),
        "feature_count": int(len(features)),
        "method": [
            "Mean and median absolute residual comparison",
            "95th percentile comparison",
            "Two-sample KS statistic",
            "Standardized mean difference"
        ],
        "p_values_calculated": False,
        "threshold_tuning": False,
        "test_split_used": False,
        "interpretation": (
            "Exploratory feature-level residual separation within validation. "
            "Temporal correlation limits independent-sample inference. "
            "Feature differences are not proof of attack indicators."
        )
    }

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print("TOP 15 FEATURES BY KS STATISTIC")
    print("=" * 72)
    print(
        result.nlargest(15, "KS Statistic")[
            [
                "Feature",
                "Benign Mean Abs Residual",
                "Infilteration Mean Abs Residual",
                "KS Statistic",
                "Standardized Mean Difference"
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 72)
    print("TOP 15 FEATURES WITH HIGHER INFILTRATION ERROR")
    print("=" * 72)
    print(
        result.nlargest(
            15, "Mean Difference (Attack - Benign)"
        )[
            [
                "Feature",
                "Benign Mean Abs Residual",
                "Infilteration Mean Abs Residual",
                "Mean Difference (Attack - Benign)"
            ]
        ].to_string(index=False)
    )

    print("\nOutputs:", OUT.relative_to(ROOT))
    print("=" * 72)

if __name__ == "__main__":
    main()
