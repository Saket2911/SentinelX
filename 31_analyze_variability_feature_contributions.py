from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"F:\Projects\SentinelX")
DATA_DIR = ROOT / "Datasets"
MODEL_DIR = ROOT / "Models" / "lstm_world_model_v1"
JOINED_DIR = MODEL_DIR / "attack_ratio_error_analysis_v1"

OUTPUT_DIR = MODEL_DIR / "variability_feature_analysis_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "validation": {
        "preprocessed": DATA_DIR / "sentinelx_validation_preprocessed.csv",
        "joined": JOINED_DIR / "validation_joined_states.csv"
    },
    "test": {
        "preprocessed": DATA_DIR / "sentinelx_test_preprocessed.csv",
        "joined": JOINED_DIR / "test_joined_states.csv"
    }
}


def analyze(name, config):

    print(f"\nProcessing {name}...")

    features = pd.read_csv(config["preprocessed"])
    errors = pd.read_csv(config["joined"])

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

    errors["timestamp"] = pd.to_datetime(
        errors["timestamp"], errors="coerce"
    )

    errors["Prediction MAE"] = pd.to_numeric(
        errors["Prediction MAE"], errors="coerce"
    )

    feature_values = features[
        ["Timestamp"] + std_features
    ].copy()

    for col in std_features:
        feature_values[col] = pd.to_numeric(
            feature_values[col], errors="coerce"
        )

    merged = errors[
        ["timestamp", "Prediction MAE"]
    ].merge(
        feature_values,
        left_on="timestamp",
        right_on="Timestamp",
        how="left",
        validate="one_to_one"
    )

    if merged[std_features].isna().all(axis=None):
        raise ValueError("No feature values matched.")

    if merged["Prediction MAE"].isna().any():
        raise ValueError("Missing prediction errors detected.")

    records = []

    for feature in std_features:

        values = merged[feature]
        valid = pd.DataFrame({
            "feature": values,
            "error": merged["Prediction MAE"]
        }).dropna()

        if len(valid) < 10:
            continue

        records.append({
            "Feature": feature,
            "Valid_States": len(valid),
            "Mean_Absolute_Standardized_Value":
                valid["feature"].abs().mean(),
            "Median_Absolute_Standardized_Value":
                valid["feature"].abs().median(),
            "Spearman_Correlation_With_MAE":
                valid["feature"].abs().corr(
                    valid["error"], method="spearman"
                ),
            "Pearson_Correlation_With_MAE":
                valid["feature"].abs().corr(
                    valid["error"], method="pearson"
                )
        })

    result = pd.DataFrame(records)

    result = result.sort_values(
        "Spearman_Correlation_With_MAE",
        ascending=False
    )

    result.to_csv(
        OUTPUT_DIR / f"{name}_feature_correlations.csv",
        index=False
    )

    print("\nTop 15 features positively associated with MAE:")
    print(result.head(15).to_string(index=False))

    print("\nTop 15 features by average absolute standardized value:")
    largest = result.sort_values(
        "Mean_Absolute_Standardized_Value",
        ascending=False
    )
    print(largest.head(15).to_string(index=False))

    print(f"\n{name} analysis completed.")

    return result


def main():

    print("=" * 70)
    print("SENTINELX - VARIABILITY FEATURE CONTRIBUTION ANALYSIS")
    print("=" * 70)

    results = {}

    for name, config in CONFIG.items():

        for key, path in config.items():
            if not path.exists():
                raise FileNotFoundError(path)

        results[name] = analyze(name, config)

    # Compare validation-ranked features against test results.
    comparison = results["validation"].merge(
        results["test"],
        on="Feature",
        suffixes=("_Validation", "_Test"),
        validate="one_to_one"
    )

    comparison["Validation_Rank"] = (
        comparison["Spearman_Correlation_With_MAE_Validation"]
        .rank(ascending=False, method="min")
    )

    comparison = comparison.sort_values("Validation_Rank")

    comparison.to_csv(
        OUTPUT_DIR / "validation_test_feature_comparison.csv",
        index=False
    )

    print("\nTop 15 validation-ranked features and test correlations:")

    display_columns = [
        "Feature",
        "Spearman_Correlation_With_MAE_Validation",
        "Spearman_Correlation_With_MAE_Test",
        "Mean_Absolute_Standardized_Value_Validation",
        "Mean_Absolute_Standardized_Value_Test"
    ]

    print(
        comparison[display_columns]
        .head(15)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Output folder: {OUTPUT_DIR}")
    print("No model or original dataset files were modified.")


if __name__ == "__main__":
    main()
