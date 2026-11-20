from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"F:\Projects\SentinelX")
DATA = ROOT / "Datasets"
MODEL = ROOT / "Models" / "lstm_world_model_v1"

JOINED = MODEL / "attack_ratio_error_analysis_v1"
OUTPUT = MODEL / "controlled_feature_analysis_v1"

OUTPUT.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "validation": {
        "features": DATA / "sentinelx_validation_preprocessed.csv",
        "joined": JOINED / "validation_joined_states.csv",
        "label": "label"
    },
    "test": {
        "features": DATA / "sentinelx_test_preprocessed.csv",
        "joined": JOINED / "test_joined_states.csv",
        "label": "actual_label"
    }
}


def residualize(values, controls):
    """Remove the linear association with the control variables."""
    matrix = np.column_stack([
        np.ones(len(controls)),
        controls.to_numpy(dtype=float)
    ])

    coefficients = np.linalg.lstsq(
        matrix, values.to_numpy(dtype=float), rcond=None
    )[0]

    return values.to_numpy(dtype=float) - matrix @ coefficients


def analyze(name, config):

    print(f"\nProcessing {name}...")

    features = pd.read_csv(config["features"])
    states = pd.read_csv(config["joined"])

    label_col = config["label"]

    std_features = [
        c for c in features.columns if c.endswith("__std")
    ]

    if len(std_features) != 68:
        raise ValueError(
            f"Expected 68 features, found {len(std_features)}"
        )

    features["Timestamp"] = pd.to_datetime(
        features["Timestamp"], errors="coerce"
    )

    states["timestamp"] = pd.to_datetime(
        states["timestamp"], errors="coerce"
    )

    merged = states.merge(
        features[["Timestamp"] + std_features],
        left_on="timestamp",
        right_on="Timestamp",
        how="left",
        validate="one_to_one"
    )

    if merged[std_features].isna().all(axis=None):
        raise ValueError("Feature timestamp matching failed")

    if merged[std_features].isna().any().any():
        raise ValueError("Missing feature values after matching")

    required = [
        "Prediction MAE",
        "Flow Count",
        "Attack Flow Ratio",
        label_col
    ]

    for col in required:
        if col not in merged.columns:
            raise ValueError(f"Missing required column: {col}")

        merged[col] = pd.to_numeric(
            merged[col], errors="coerce"
        ) if col != label_col else merged[col]

    # Keep only Benign and Bot for a meaningful label comparison.
    merged = merged[
        merged[label_col].isin(["Benign", "Bot"])
    ].copy()

    merged["Label Binary"] = (
        merged[label_col] == "Bot"
    ).astype(float)

    merged["Log Flow Count"] = np.log1p(
        merged["Flow Count"].astype(float)
    )

    merged = merged.dropna(
        subset=[
            "Prediction MAE",
            "Flow Count",
            "Attack Flow Ratio"
        ]
    ).copy()

    # Rank-transform the response and control variables.
    ranked_error = merged["Prediction MAE"].rank(
        method="average"
    )

    controls = pd.DataFrame({
        "Flow_Count": merged["Log Flow Count"].rank(),
        "Attack_Ratio": merged["Attack Flow Ratio"].rank(),
        "Label": merged["Label Binary"]
    })

    records = []

    for feature in std_features:

        absolute_values = merged[feature].abs()

        ranked_feature = absolute_values.rank(
            method="average"
        )

        valid = pd.DataFrame({
            "feature": ranked_feature,
            "error": ranked_error
        }).dropna()

        if len(valid) != len(merged):
            raise ValueError(f"Missing values in {feature}")

        x_residual = residualize(
            ranked_feature, controls
        )

        y_residual = residualize(
            ranked_error, controls
        )

        if np.std(x_residual) == 0 or np.std(y_residual) == 0:
            partial_corr = np.nan
        else:
            partial_corr = np.corrcoef(
                x_residual, y_residual
            )[0, 1]

        raw_corr = absolute_values.corr(
            merged["Prediction MAE"],
            method="spearman"
        )

        records.append({
            "Feature": feature,
            "Raw_Spearman": raw_corr,
            "Controlled_Partial_Spearman": partial_corr
        })

    result = pd.DataFrame(records)

    result["Absolute_Controlled_Correlation"] = (
        result["Controlled_Partial_Spearman"].abs()
    )

    result = result.sort_values(
        "Absolute_Controlled_Correlation",
        ascending=False
    )

    result.to_csv(
        OUTPUT / f"{name}_controlled_correlations.csv",
        index=False
    )

    print(f"States analysed: {len(merged):,}")

    print("\nTop 15 controlled feature relationships:")
    print(result.head(15).to_string(index=False))

    # Detect exactly identical feature columns.
    duplicates = []

    for i, first in enumerate(std_features):
        for second in std_features[i + 1:]:
            if np.array_equal(
                merged[first].to_numpy(),
                merged[second].to_numpy(),
                equal_nan=True
            ):
                duplicates.append({
                    "Feature_1": first,
                    "Feature_2": second
                })

    pd.DataFrame(duplicates).to_csv(
        OUTPUT / f"{name}_duplicate_features.csv",
        index=False
    )

    print(f"\nExact duplicate feature pairs: {len(duplicates)}")
    for item in duplicates:
        print(
            f"  {item['Feature_1']} == {item['Feature_2']}"
        )

    return result


def main():

    print("=" * 70)
    print("SENTINELX - CONTROLLED FEATURE CORRELATION ANALYSIS")
    print("=" * 70)

    results = {}

    for name, config in CONFIG.items():
        for path in config.values():
            if isinstance(path, Path) and not path.exists():
                raise FileNotFoundError(path)

        results[name] = analyze(name, config)

    comparison = results["validation"].merge(
        results["test"],
        on="Feature",
        suffixes=("_Validation", "_Test"),
        validate="one_to_one"
    )

    comparison.to_csv(
        OUTPUT / "validation_test_comparison.csv",
        index=False
    )

    print("\nTop 15 validation-ranked features in both partitions:")

    ranked = comparison.sort_values(
        "Controlled_Partial_Spearman_Validation",
        key=lambda s: s.abs(),
        ascending=False
    )

    print(
        ranked[[
            "Feature",
            "Controlled_Partial_Spearman_Validation",
            "Controlled_Partial_Spearman_Test"
        ]].head(15).to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Output folder: {OUTPUT}")
    print("No model or original dataset files were modified.")


if __name__ == "__main__":
    main()
