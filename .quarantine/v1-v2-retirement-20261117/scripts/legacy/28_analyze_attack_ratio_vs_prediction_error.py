from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"F:\Projects\SentinelX")

VALIDATION_FILE = (
    ROOT / "Models" / "lstm_world_model_v1"
    / "prediction_error_analysis_v1"
    / "sequence_error_analysis.csv"
)

TEST_FILE = (
    ROOT / "Models" / "lstm_world_model_v1"
    / "anomaly_experiment_v1"
    / "test_anomaly_predictions.csv"
)

COMPOSITION_FILE = (
    ROOT / "Datasets" / "diagnostics"
    / "mixed_traffic_analysis_v1"
    / "per_second_traffic_composition.csv"
)

OUTPUT_DIR = (
    ROOT / "Models" / "lstm_world_model_v1"
    / "attack_ratio_error_analysis_v1"
)

FILES = [
    VALIDATION_FILE,
    TEST_FILE,
    COMPOSITION_FILE,
]

for path in FILES:
    if not path.exists():
        raise FileNotFoundError(path)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_files = [
    OUTPUT_DIR / "validation_error_by_attack_ratio.csv",
    OUTPUT_DIR / "validation_error_by_ratio_and_label.csv",
    OUTPUT_DIR / "test_error_by_attack_ratio.csv",
    OUTPUT_DIR / "test_error_by_ratio_and_label.csv",
    OUTPUT_DIR / "validation_joined_states.csv",
    OUTPUT_DIR / "test_joined_states.csv",
    OUTPUT_DIR / "attack_ratio_error_analysis_report.txt",
]

existing = [str(path) for path in output_files if path.exists()]
if existing:
    raise FileExistsError(
        "Output files already exist. Nothing was overwritten:\n"
        + "\n".join(existing)
    )

print("=" * 70)
print("SENTINELX - ATTACK FLOW RATIO VS PREDICTION ERROR")
print("=" * 70)

# Load the one-second flow composition.
composition = pd.read_csv(COMPOSITION_FILE)
composition["Timestamp"] = pd.to_datetime(
    composition["Timestamp"], errors="coerce"
)

if composition["Timestamp"].isna().any():
    raise ValueError("Invalid timestamps in flow composition file.")

if composition["Timestamp"].duplicated().any():
    raise ValueError("Duplicate timestamps in flow composition file.")

composition_columns = [
    "Timestamp",
    "Flow Count",
    "Benign Flow Count",
    "Attack Flow Count",
    "Attack Flow Ratio",
    "Has Benign",
    "Has Attack",
    "Mixed Benign Attack",
]

composition = composition[composition_columns].copy()

def assign_ratio_group(row):
    # Empty windows are not the same as benign-only windows.
    if row["Flow Count"] == 0:
        return "No flows"

    ratio = row["Attack Flow Ratio"]

    if ratio == 0:
        return "0% (Benign-only)"
    if ratio <= 0.10:
        return "(0%, 10%]"
    if ratio <= 0.25:
        return "(10%, 25%]"
    if ratio <= 0.50:
        return "(25%, 50%]"
    return "(50%, 100%]"

def analyze_partition(name, input_file, error_column):
    print(f"\nProcessing {name}...")

    errors = pd.read_csv(input_file)

    if "timestamp" not in errors.columns:
        raise ValueError(f"{name}: timestamp column missing.")

    if error_column not in errors.columns:
        raise ValueError(
            f"{name}: {error_column} column missing."
        )

    label_column = (
        "label" if "label" in errors.columns else "actual_label"
    )

    errors["Timestamp"] = pd.to_datetime(
        errors["timestamp"], errors="coerce"
    )

    if errors["Timestamp"].isna().any():
        raise ValueError(f"{name}: invalid prediction timestamps.")

    if errors["Timestamp"].duplicated().any():
        raise ValueError(f"{name}: duplicate prediction timestamps.")

    errors[error_column] = pd.to_numeric(
        errors[error_column], errors="coerce"
    )

    if errors[error_column].isna().any():
        raise ValueError(f"{name}: invalid prediction errors.")

    errors = errors.rename(
        columns={error_column: "Prediction MAE"}
    )

    # Match prediction target timestamps with original flow composition.
    joined = errors.merge(
        composition,
        on="Timestamp",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = int((joined["_merge"] == "left_only").sum())

    # A timestamp absent from the observed-flow file represents
    # an empty one-second window in the state representation.
    numeric_columns = [
        "Flow Count",
        "Benign Flow Count",
        "Attack Flow Count",
        "Attack Flow Ratio",
        "Has Benign",
        "Has Attack",
        "Mixed Benign Attack",
    ]

    joined[numeric_columns] = joined[numeric_columns].fillna(0)

    joined["Traffic Group"] = joined.apply(
        assign_ratio_group, axis=1
    )

    # Check that unmatched timestamps correspond to NO_FLOWS labels.
    no_flow_labels = (
        joined[label_column]
        .astype(str)
        .str.upper()
        .eq("NO_FLOWS")
    )

    unexpected_unmatched = int(
        ((joined["_merge"] == "left_only") & ~no_flow_labels).sum()
    )

    if unexpected_unmatched:
        raise ValueError(
            f"{name}: {unexpected_unmatched} timestamps were absent "
            "from the flow-composition file but were not labelled "
            "NO_FLOWS. Investigate timestamp alignment."
        )

    # Check that NO_FLOWS labels have no original flow records.
    unexpected_no_flow = int(
    (no_flow_labels & (joined["Flow Count"] > 0)).sum()
)

    if unexpected_no_flow:
        raise ValueError(
            f"{name}: NO_FLOWS labels matched non-empty flow windows."
        )

    joined = joined.drop(columns=["_merge"])
    joined.to_csv(
        OUTPUT_DIR / f"{name.lower()}_joined_states.csv",
        index=False
    )

    # Aggregate prediction error by attack-flow ratio.
    order = [
        "No flows",
        "0% (Benign-only)",
        "(0%, 10%]",
        "(10%, 25%]",
        "(25%, 50%]",
        "(50%, 100%]",
    ]

    by_ratio = (
        joined.groupby("Traffic Group", observed=False)
        .agg(
            State_Count=("Prediction MAE", "size"),
            Mean_MAE=("Prediction MAE", "mean"),
            Median_MAE=("Prediction MAE", "median"),
            P90_MAE=(
                "Prediction MAE",
                lambda values: values.quantile(0.90)
            ),
            Mean_Attack_Flow_Ratio=("Attack Flow Ratio", "mean"),
            Mean_Flow_Count=("Flow Count", "mean"),
        )
        .reindex(order)
        .reset_index()
    )

    by_ratio.to_csv(
        OUTPUT_DIR / f"{name.lower()}_error_by_attack_ratio.csv",
        index=False
    )

    # Also separate the results by the target state's original label.
    by_ratio_label = (
        joined.groupby(
            ["Traffic Group", label_column],
            observed=False
        )
        .agg(
            State_Count=("Prediction MAE", "size"),
            Mean_MAE=("Prediction MAE", "mean"),
            Median_MAE=("Prediction MAE", "median"),
        )
        .reset_index()
    )

    by_ratio_label.to_csv(
        OUTPUT_DIR / f"{name.lower()}_error_by_ratio_and_label.csv",
        index=False
    )

    print(f"Prediction rows: {len(joined):,}")
    print(f"Unmatched composition timestamps: {unmatched:,}")
    print(f"Unexpected unmatched timestamps: {unexpected_unmatched:,}")
    print("\nError by attack-flow ratio:")
    print(by_ratio.to_string(index=False))

    return {
        "name": name,
        "rows": len(joined),
        "unmatched": unmatched,
        "unexpected_unmatched": unexpected_unmatched,
        "by_ratio": by_ratio,
        "by_ratio_label": by_ratio_label,
        "label_column": label_column,
    }

validation_result = analyze_partition(
    "validation",
    VALIDATION_FILE,
    "mae",
)

test_result = analyze_partition(
    "test",
    TEST_FILE,
    "prediction_error",
)

# Write a concise report with separate partition results.
report = [
    "SENTINELX - ATTACK FLOW RATIO VS PREDICTION ERROR",
    "=" * 65,
    "",
    "Purpose:",
    "Describe how next-state prediction error varies with the",
    "proportion of attack-labelled flows in each target second.",
    "",
    "Ratio groups:",
    "No flows",
    "0% (Benign-only)",
    "(0%, 10%]",
    "(10%, 25%]",
    "(25%, 50%]",
    "(50%, 100%]",
    "",
    "VALIDATION RESULTS",
    "-" * 65,
    f"Prediction rows: {validation_result['rows']:,}",
    f"Unmatched composition timestamps: "
    f"{validation_result['unmatched']:,}",
    "",
    validation_result["by_ratio"].to_string(index=False),
    "",
    "TEST RESULTS",
    "-" * 65,
    f"Prediction rows: {test_result['rows']:,}",
    f"Unmatched composition timestamps: "
    f"{test_result['unmatched']:,}",
    "",
    test_result["by_ratio"].to_string(index=False),
    "",
    "INTERPRETATION CAUTIONS",
    "-" * 65,
    "1. Validation and test results are reported separately.",
    "2. This is descriptive analysis, not threshold tuning.",
    "3. A higher attack-flow ratio does not automatically mean",
    "   higher prediction error.",
    "4. Attack-flow ratio is derived from dataset ground-truth labels.",
    "   It is used only for retrospective analysis, never as an",
    "   input feature for the LSTM.",
    "5. Test results must not be used to select model settings.",
    "6. These results do not establish early attack detection.",
]

(OUTPUT_DIR / "attack_ratio_error_analysis_report.txt").write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print("Output folder:", OUTPUT_DIR)
print("\nNo existing model or dataset files were modified.")
