from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"F:\Projects\SentinelX")

DATA_DIR = ROOT / "Datasets"

MODEL_DIR = ROOT / "Models" / "lstm_world_model_v1"

JOINED_DIR = MODEL_DIR / "attack_ratio_error_analysis_v1"

OUTPUT_DIR = MODEL_DIR / "variability_error_analysis_v1"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "validation": {
        "preprocessed": DATA_DIR / "sentinelx_validation_preprocessed.csv",
        "joined": JOINED_DIR / "validation_joined_states.csv",
        "label_column": "label"
    },
    "test": {
        "preprocessed": DATA_DIR / "sentinelx_test_preprocessed.csv",
        "joined": JOINED_DIR / "test_joined_states.csv",
        "label_column": "actual_label"
    }
}


def summarize(group, error_col):
    return pd.Series({
        "State_Count": len(group),
        "Mean_Variability_Index": group["Variability Deviation Index"].mean(),
        "Mean_MAE": group[error_col].mean(),
        "Median_MAE": group[error_col].median(),
        "P90_MAE": group[error_col].quantile(0.90)
    })


def analyze_partition(name, config):

    print(f"\nProcessing {name}...")

    preprocessed = pd.read_csv(config["preprocessed"])
    errors = pd.read_csv(config["joined"])

    label_col = config["label_column"]

    if "Timestamp" not in preprocessed.columns:
        raise ValueError("Timestamp missing from preprocessed data")

    if "timestamp" not in errors.columns:
        raise ValueError("timestamp missing from prediction results")

    if "Prediction MAE" not in errors.columns:
        raise ValueError("Prediction MAE missing from joined data")

    std_features = [
        col for col in preprocessed.columns
        if col.endswith("__std")
    ]

    if len(std_features) != 68:
        raise ValueError(
            f"Expected 68 standard-deviation features, "
            f"found {len(std_features)}"
        )

    print(f"Standard-deviation features: {len(std_features)}")

    preprocessed["Timestamp"] = pd.to_datetime(
        preprocessed["Timestamp"], errors="coerce"
    )

    errors["timestamp"] = pd.to_datetime(
        errors["timestamp"], errors="coerce"
    )

    preprocessed[std_features] = preprocessed[
        std_features
    ].apply(pd.to_numeric, errors="coerce")

    errors["Prediction MAE"] = pd.to_numeric(
        errors["Prediction MAE"], errors="coerce"
    )

    if preprocessed["Timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps in preprocessed data")

    if errors["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps in prediction data")

    # Calculate average absolute standardized deviation.
    # This is a deviation index, not a raw physical variability measure.
    preprocessed["Variability Deviation Index"] = (
        preprocessed[std_features]
        .abs()
        .mean(axis=1)
    )

    feature_data = preprocessed[
        ["Timestamp", "Variability Deviation Index"]
    ]

    joined = errors.merge(
        feature_data,
        left_on="timestamp",
        right_on="Timestamp",
        how="left",
        validate="one_to_one",
        indicator=True
    )

    unmatched = int((joined["_merge"] != "both").sum())

    print(f"Prediction rows: {len(errors):,}")
    print(f"Unmatched timestamps: {unmatched}")

    if unmatched:
        raise ValueError("Timestamp matching failed")

    joined = joined.drop(columns=["_merge", "Timestamp_x", "Timestamp_y"], errors="ignore")

    joined = joined.dropna(
        subset=["Variability Deviation Index", "Prediction MAE"]
    ).copy()

    # Divide states into four equally sized index groups.
    percentile = joined[
        "Variability Deviation Index"
    ].rank(method="first", pct=True)

    joined["Variability Group"] = pd.cut(
        percentile,
        bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=["Q1 - Lowest", "Q2", "Q3", "Q4 - Highest"],
        include_lowest=True
    )

    # Overall summary
    overall = (
        joined.groupby("Variability Group", observed=False)
        .apply(
            lambda g: summarize(g, "Prediction MAE"),
            include_groups=False
        )
        .reset_index()
    )

    overall.to_csv(
        OUTPUT_DIR / f"{name}_error_by_variability.csv",
        index=False
    )

    print("\nPrediction error by variability group:")
    print(overall.to_string(index=False))

    # Compare separately by actual label
    by_label = (
        joined.groupby(
            ["Variability Group", label_col],
            observed=False
        )
        .apply(
            lambda g: summarize(g, "Prediction MAE"),
            include_groups=False
        )
        .reset_index()
    )

    by_label.to_csv(
        OUTPUT_DIR / f"{name}_error_by_variability_and_label.csv",
        index=False
    )

    print("\nPrediction error by variability group and label:")
    print(by_label.to_string(index=False))

    # Correlation analysis
    correlations = []

    for method in ["spearman", "pearson"]:
        value = joined[
            "Variability Deviation Index"
        ].corr(joined["Prediction MAE"], method=method)

        correlations.append({
            "Method": method,
            "Correlation": value
        })

    correlation_df = pd.DataFrame(correlations)

    correlation_df.to_csv(
        OUTPUT_DIR / f"{name}_correlations.csv",
        index=False
    )

    print("\nVariability index vs prediction error:")
    print(correlation_df.to_string(index=False))

    # Save joined data for further investigation
    joined.to_csv(
        OUTPUT_DIR / f"{name}_joined_analysis.csv",
        index=False
    )

    print(f"\n{name} analysis completed.")


def main():

    print("=" * 70)
    print("SENTINELX - VARIABILITY VS PREDICTION ERROR")
    print("=" * 70)

    for name, config in CONFIG.items():

        for key in ["preprocessed", "joined"]:
            if not config[key].exists():
                raise FileNotFoundError(config[key])

        analyze_partition(name, config)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Output folder: {OUTPUT_DIR}")
    print("Original datasets and model files were not modified.")


if __name__ == "__main__":
    main()

