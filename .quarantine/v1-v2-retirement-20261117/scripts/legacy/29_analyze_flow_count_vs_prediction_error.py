
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"F:\Projects\SentinelX")

INPUT_DIR = (
    ROOT / "Models" / "lstm_world_model_v1"
    / "attack_ratio_error_analysis_v1"
)

OUTPUT_DIR = (
    ROOT / "Models" / "lstm_world_model_v1"
    / "flow_count_error_analysis_v1"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "validation": (
        INPUT_DIR / "validation_joined_states.csv",
        "Prediction MAE",
        "label"
    ),
    "test": (
        INPUT_DIR / "test_joined_states.csv",
        "Prediction MAE",
        "actual_label"
    )
}

# Fixed flow-count groups
BINS = [-1, 0, 5, 10, 20, 50, 100, np.inf]

BIN_LABELS = [
    "No flows",
    "1-5 flows",
    "6-10 flows",
    "11-20 flows",
    "21-50 flows",
    "51-100 flows",
    "Above 100 flows"
]


def analyze_partition(name, file_path, error_col, label_col):

    print(f"\nProcessing {name}...")

    df = pd.read_csv(file_path)

    required = [
        "Flow Count",
        "Attack Flow Ratio",
        error_col,
        label_col
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["Flow Count"] = pd.to_numeric(
        df["Flow Count"], errors="coerce"
    )

    df[error_col] = pd.to_numeric(
        df[error_col], errors="coerce"
    )

    df["Attack Flow Ratio"] = pd.to_numeric(
        df["Attack Flow Ratio"], errors="coerce"
    )

    df = df.dropna(
        subset=["Flow Count", error_col, label_col]
    ).copy()

    df["Flow Group"] = pd.cut(
        df["Flow Count"],
        bins=BINS,
        labels=BIN_LABELS,
        include_lowest=True
    )

    def summarize(group):
        return pd.Series({
            "State_Count": len(group),
            "Mean_MAE": group[error_col].mean(),
            "Median_MAE": group[error_col].median(),
            "P90_MAE": group[error_col].quantile(0.90),
            "Mean_Flow_Count": group["Flow Count"].mean(),
            "Mean_Attack_Flow_Ratio":
                group["Attack Flow Ratio"].mean()
        })

    # Overall flow-count analysis
    overall = (
        df.groupby("Flow Group", observed=False)
        .apply(summarize, include_groups=False)
        .reset_index()
    )

    overall.to_csv(
        OUTPUT_DIR / f"{name}_error_by_flow_count.csv",
        index=False
    )

    print("\nPrediction error by flow count:")
    print(overall.to_string(index=False))

    # Compare separately by actual state label
    by_label = (
        df.groupby(
            ["Flow Group", label_col],
            observed=False
        )
        .apply(summarize, include_groups=False)
        .reset_index()
    )

    by_label.to_csv(
        OUTPUT_DIR / f"{name}_error_by_flow_count_and_label.csv",
        index=False
    )

    print("\nPrediction error by flow count and label:")
    print(by_label.to_string(index=False))

    # Correlation analysis
    correlations = []

    for column in ["Flow Count", "Attack Flow Ratio"]:
        valid = df[[column, error_col]].dropna()

        correlations.append({
            "Variable": column,
            "Spearman_Correlation": valid[column].corr(
                valid[error_col],
                method="spearman"
            ),
            "Pearson_Correlation": valid[column].corr(
                valid[error_col],
                method="pearson"
            )
        })

    correlation_df = pd.DataFrame(correlations)

    correlation_df.to_csv(
        OUTPUT_DIR / f"{name}_correlations.csv",
        index=False
    )

    print("\nCorrelations:")
    print(correlation_df.to_string(index=False))

    print(f"\n{name} analysis completed.")


def main():

    print("=" * 70)
    print("SENTINELX - FLOW COUNT VS PREDICTION ERROR")
    print("=" * 70)

    for name, (file_path, error_col, label_col) in FILES.items():

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        analyze_partition(
            name,
            file_path,
            error_col,
            label_col
        )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Output folder: {OUTPUT_DIR}")
    print("No original datasets or model files were modified.")


if __name__ == "__main__":
    main()

