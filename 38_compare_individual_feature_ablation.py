from pathlib import Path
import pandas as pd

ROOT = Path(r"F:\Projects\SentinelX")
MODEL_DIR = ROOT / "Models" / "lstm_world_model_v1"

VALIDATION_FILE = (
    MODEL_DIR / "individual_feature_ablation_validation_v1"
    / "ablation_results.csv"
)

TEST_FILE = (
    MODEL_DIR / "individual_feature_ablation_test_v1"
    / "ablation_results.csv"
)

OUTPUT_DIR = MODEL_DIR / "individual_feature_comparison_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "individual_feature_comparison.csv"
OUTPUT_REPORT = OUTPUT_DIR / "comparison_report.txt"


def main():

    print("=" * 70)
    print("SENTINELX - INDIVIDUAL FEATURE COMPARISON")
    print("=" * 70)

    for path in [VALIDATION_FILE, TEST_FILE]:
        if not path.exists():
            raise FileNotFoundError(path)

    val = pd.read_csv(VALIDATION_FILE)
    test = pd.read_csv(TEST_FILE)

    required = [
        "Experiment",
        "Masked_Group",
        "Overall_MAE_Change_Percent"
    ]

    for name, df in [("Validation", val), ("Test", test)]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{name} file missing columns: {missing}")

    val = val[val["Experiment"] == "Ablation"].copy()
    test = test[test["Experiment"] == "Ablation"].copy()

    val = val[
        ["Masked_Group", "Overall_MAE_Change_Percent"]
    ].rename(columns={
        "Overall_MAE_Change_Percent": "Validation_MAE_Change_Percent"
    })

    test = test[
        ["Masked_Group", "Overall_MAE_Change_Percent"]
    ].rename(columns={
        "Overall_MAE_Change_Percent": "Test_MAE_Change_Percent"
    })

    if val["Masked_Group"].duplicated().any():
        raise ValueError("Duplicate feature names in validation results")

    if test["Masked_Group"].duplicated().any():
        raise ValueError("Duplicate feature names in test results")

    comparison = val.merge(
        test,
        on="Masked_Group",
        how="outer",
        indicator=True,
        validate="one_to_one"
    )

    unmatched = comparison[comparison["_merge"] != "both"]

    if not unmatched.empty:
        raise ValueError(
            "Validation/test feature mismatch:\n"
            + unmatched.to_string(index=False)
        )

    comparison = comparison.drop(columns="_merge")

    comparison["Average_MAE_Change_Percent"] = (
        comparison["Validation_MAE_Change_Percent"]
        + comparison["Test_MAE_Change_Percent"]
    ) / 2

    comparison["Validation_Rank"] = (
        comparison["Validation_MAE_Change_Percent"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    comparison["Test_Rank"] = (
        comparison["Test_MAE_Change_Percent"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    comparison["Average_Rank"] = (
        comparison["Validation_Rank"]
        + comparison["Test_Rank"]
    ) / 2

    comparison["Positive_In_Both"] = (
        (comparison["Validation_MAE_Change_Percent"] > 0)
        & (comparison["Test_MAE_Change_Percent"] > 0)
    )

    comparison = comparison.sort_values(
        ["Average_Rank", "Average_MAE_Change_Percent"],
        ascending=[True, False]
    ).reset_index(drop=True)

    comparison.to_csv(OUTPUT_CSV, index=False)

    top = comparison.head(15)

    lines = [
        "SENTINELX - INDIVIDUAL FEATURE ABLATION COMPARISON",
        "=" * 65,
        "",
        f"Features compared: {len(comparison)}",
        f"Positive in both partitions: {comparison['Positive_In_Both'].sum()}",
        "",
        "TOP 15 FEATURES BY AVERAGE RANK",
        "-" * 65,
        top[[
            "Masked_Group",
            "Validation_MAE_Change_Percent",
            "Test_MAE_Change_Percent",
            "Average_MAE_Change_Percent",
            "Validation_Rank",
            "Test_Rank",
            "Average_Rank"
        ]].to_string(index=False),
        "",
        "INTERPRETATION",
        "-" * 65,
        "Positive MAE change means masking the feature worsened forecasting.",
        "Negative MAE change means masking the feature improved forecasting.",
        "Consistent positive effects indicate sensitivity in both partitions.",
        "",
        "This is a masking sensitivity analysis, not causal feature importance.",
        "Validation and test results are exploratory and do not establish",
        "independent generalization or attack-detection capability.",
        "",
        f"CSV: {OUTPUT_CSV}",
    ]

    report = "\n".join(lines)
    OUTPUT_REPORT.write_text(report, encoding="utf-8")

    print(report)

    print("\nComparison saved successfully.")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Report: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
