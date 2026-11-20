import re
import pandas as pd
from pathlib import Path

BASE = Path("Models/lstm_world_model_v3_gpu_run2")
INPUT = BASE / "multi_step_feature_analysis_v1" / "feature_wise_metrics.csv"
OUT = BASE / "multi_step_feature_analysis_v1"

df = pd.read_csv(INPUT)
df = df[df["Target_Label"] == "ALL"].copy()

def classify(name):

    # Timing
    if (
        name.startswith(("Active ", "Bwd IAT ", "Flow IAT ",
                         "Fwd IAT ", "Idle "))
        or name == "Flow Duration"
    ):
        return "Timing"

    # Packet lengths and headers
    if (
        "Pkt Len" in name
        or name in {
            "Bwd Header Len", "Fwd Header Len",
            "Bwd Seg Size Avg", "Fwd Seg Size Avg",
            "Fwd Seg Size Min", "Pkt Size Avg",
            "TotLen Bwd Pkts", "TotLen Fwd Pkts"
        }
    ):
        return "Packet lengths and headers"

    # Packet and flow counts / volume
    if (
        name in {
            "Flow Count", "Fwd Act Data Pkts",
            "Subflow Bwd Byts", "Subflow Bwd Pkts",
            "Subflow Fwd Byts", "Subflow Fwd Pkts",
            "Tot Bwd Pkts", "Tot Fwd Pkts"
        }
    ):
        return "Packet and flow counts"

    # TCP flags
    if (
        "Flag" in name
        or name in {"CWE Flag Count", "Fwd PSH Flags"}
    ):
        return "TCP flags"

    # Rates and throughput
    if name in {
        "Bwd Pkts/s", "Flow Byts/s",
        "Flow Pkts/s", "Fwd Pkts/s"
    }:
        return "Rates and throughput"

    # Protocol and network-state indicators
    if name in {
        "Down/Up Ratio", "Dst Port", "Has Flows",
        "Init Bwd Win Byts", "Init Fwd Win Byts",
        "Protocol"
    }:
        return "Protocol and network-state indicators"

    return None


# Build explicit feature-to-family mapping.
df["Base_Feature"] = df["Feature"].str.replace(
    r"__(mean|std|max)$", "", regex=True
)

df["Feature_Family"] = df["Base_Feature"].map(classify)

# Fail safely if any feature is unclassified.
unknown = sorted(df.loc[df["Feature_Family"].isna(), "Base_Feature"].unique())

if unknown:
    raise ValueError(f"Unclassified features: {unknown}")

# Every model feature must occur exactly once per horizon.
for horizon, group in df.groupby("Horizon_Seconds"):
    if len(group) != 209 or group["Feature"].nunique() != 209:
        raise ValueError(
            f"Unexpected feature coverage at horizon {horizon}"
        )

# Verify all 71 original features are represented.
base_count = df["Base_Feature"].nunique()
if base_count != 71:
    raise ValueError(f"Expected 71 base features, found {base_count}")

# Verify family mapping is stable.
mapping = df[["Feature", "Base_Feature", "Feature_Family"]].drop_duplicates()

if len(mapping) != 209:
    raise ValueError(f"Expected 209 mapped features, found {len(mapping)}")

# Family-level macro averages.
# Each feature receives equal weight in its family.
rows = []

for (horizon, family), group in df.groupby(
    ["Horizon_Seconds", "Feature_Family"]
):

    lstm_mae = group["LSTM_MAE"].mean()
    persistence_mae = group["Persistence_MAE"].mean()

    lstm_rmse = group["LSTM_RMSE"].mean()
    persistence_rmse = group["Persistence_RMSE"].mean()

    lstm_huber = group["LSTM_Huber"].mean()
    persistence_huber = group["Persistence_Huber"].mean()

    mae_improvement = (
        (persistence_mae - lstm_mae) / persistence_mae * 100
        if persistence_mae != 0 else None
    )

    rmse_improvement = (
        (persistence_rmse - lstm_rmse) / persistence_rmse * 100
        if persistence_rmse != 0 else None
    )

    rows.append({
        "Horizon_Seconds": horizon,
        "Feature_Family": family,
        "Feature_Count": len(group),
        "LSTM_Better_Count": int(
            (group["LSTM_MAE"] < group["Persistence_MAE"]).sum()
        ),
        "Persistence_Better_Count": int(
            (group["LSTM_MAE"] > group["Persistence_MAE"]).sum()
        ),
        "Equal_Count": int(
            (group["LSTM_MAE"] == group["Persistence_MAE"]).sum()
        ),
        "Zero_Persistence_MAE_Count": int(
            (group["Persistence_MAE"] == 0).sum()
        ),
        "LSTM_Mean_MAE": lstm_mae,
        "Persistence_Mean_MAE": persistence_mae,
        "MAE_Improvement_Percent": mae_improvement,
        "LSTM_Mean_RMSE": lstm_rmse,
        "Persistence_Mean_RMSE": persistence_rmse,
        "RMSE_Improvement_Percent": rmse_improvement,
        "LSTM_Mean_Huber": lstm_huber,
        "Persistence_Mean_Huber": persistence_huber
    })

summary = pd.DataFrame(rows).sort_values(
    ["Horizon_Seconds", "Feature_Family"]
)

# Save artifacts.
mapping_path = OUT / "feature_family_mapping_v1.csv"
summary_path = OUT / "feature_family_summary_v1.csv"
report_path = OUT / "feature_family_report_v1.txt"

mapping.to_csv(mapping_path, index=False)
summary.to_csv(summary_path, index=False)

with open(report_path, "w", encoding="utf-8") as f:
    f.write("SENTINELX V3 — FEATURE-FAMILY FORECASTING ANALYSIS\n")
    f.write("=" * 75 + "\n")
    f.write(f"Model features: {len(mapping)}\n")
    f.write(f"Original features: {base_count}\n")
    f.write(f"Feature families: {mapping['Feature_Family'].nunique()}\n")
    f.write("Aggregation types: mean, std, max\n")
    f.write("Metric aggregation: macro-average across features in each family\n")
    f.write("Evaluation: recursive multi-step forecasting\n")
    f.write("Metrics: preprocessed model feature space\n")
    f.write("Test data was previously used in exploratory evaluation.\n")
    f.write("These results are not an untouched independent test evaluation.\n\n")

    for horizon in sorted(summary["Horizon_Seconds"].unique()):
        f.write("=" * 75 + "\n")
        f.write(f"HORIZON: {horizon} seconds\n")
        f.write("=" * 75 + "\n")

        part = summary[summary["Horizon_Seconds"] == horizon]

        f.write(part.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        f.write("\n\n")

print("=" * 75)
print("SENTINELX V3 — FEATURE-FAMILY ANALYSIS")
print("=" * 75)

print(f"Original features: {base_count}")
print(f"Model features: {len(mapping)}")
print(f"Families: {mapping['Feature_Family'].nunique()}")
print(f"Summary rows: {len(summary)}")

print("\nFEATURE COUNTS BY FAMILY:")
print(
    mapping.groupby("Feature_Family")["Feature"]
    .nunique()
    .sort_values(ascending=False)
    .to_string()
)

print("\nFAMILY-LEVEL FORECASTING RESULTS:")
for horizon in sorted(summary["Horizon_Seconds"].unique()):
    print(f"\n--- {horizon}-second horizon ---")
    part = summary[summary["Horizon_Seconds"] == horizon]

    print(
        part[[
            "Feature_Family",
            "Feature_Count",
            "LSTM_Better_Count",
            "Persistence_Better_Count",
            "LSTM_Mean_MAE",
            "Persistence_Mean_MAE",
            "MAE_Improvement_Percent"
        ]].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

print("\nSAVED:")
print(mapping_path)
print(summary_path)
print(report_path)

print("\nVALIDATION: PASS")
print("All 209 features assigned exactly once.")
print("All 71 original features classified.")
