from pathlib import Path
import pandas as pd

BASE = Path("Models/lstm_world_model_v3_gpu_run2")
ANALYSIS = BASE / "multi_step_feature_analysis_v1"
FORECAST = BASE / "multi_step_forecasting_v3_fixed"

OUT = Path("reports")
OUT.mkdir(parents=True, exist_ok=True)

feature = pd.read_csv(ANALYSIS / "feature_wise_metrics.csv")
mapping = pd.read_csv(ANALYSIS / "feature_family_mapping_v1.csv")
families = pd.read_csv(ANALYSIS / "feature_family_summary_v1.csv")
labels = pd.read_csv(ANALYSIS / "feature_family_label_summary_v1.csv")
comparison = pd.read_csv(
    ANALYSIS / "feature_family_benign_bot_comparison_v1.csv"
)
verification = pd.read_csv(
    ANALYSIS / "aggregate_verification.csv"
)
multi = pd.read_csv(FORECAST / "multi_step_metrics.csv")

# Integrity checks
checks = {
    "Feature-wise records": len(feature) == 4180,
    "Unique model features": feature["Feature"].nunique() == 209,
    "Feature-family mapping": len(mapping) == 209,
    "Original feature count": mapping["Base_Feature"].nunique() == 71,
    "Six feature families": mapping["Feature_Family"].nunique() == 6,
    "Family summary rows": len(families) == 30,
    "Benign/Bot summary rows": len(labels) == 60,
    "Benign/Bot comparison rows": len(comparison) == 30,
    "Aggregate verification": (
        verification["Verification"].eq("PASS").all()
    ),
}

print("REPORT INTEGRITY CHECKS")
for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}: {name}")

if not all(checks.values()):
    raise RuntimeError("Report generation stopped: integrity check failed.")

def table(df):
    return df.to_string(index=False, float_format=lambda x: f"{x:.5f}")

lines = []

def add(text=""):
    lines.append(str(text))

add("# SentinelX V3 — Predictive World Model Forecasting Analysis")
add()
add("## 1. Executive summary")
add()
add(
    "The SentinelX V3 LSTM World Model was evaluated for numerical "
    "future network-state prediction using recursive forecasts at "
    "1, 2, 5, 10 and 30 seconds."
)
add(
    "The model was compared with a persistence baseline, which predicts "
    "the future state by repeating the most recent observed state."
)
add(
    "The feature-wise analysis covered 209 model features derived from "
    "71 original network features, grouped into six feature families."
)
add(
    "All 20 aggregate comparisons between the feature-wise experiment "
    "and the original multi-step forecasting results passed verification."
)
add()
add("## 2. Model and evaluation configuration")
add()
add("- Model: SentinelX V3 LSTM World Model")
add("- Checkpoint: `best_model.pt`")
add("- Best checkpoint epoch: 11")
add("- Input sequence length: 10")
add("- Input features: 209")
add("- Hidden size: 128")
add("- LSTM layers: 1")
add("- Dropout: 0.2")
add("- Output features: 209")
add("- Training loss: Huber")
add("- Forecasting method: recursive prediction")
add("- Forecast horizons: 1, 2, 5, 10 and 30 seconds")
add("- Evaluation device for this analysis: CPU")
add("- Common forecast origins: 31,377")
add("- Test sequence count: 31,406")
add()
add("The test partition had previously been used for exploratory "
    "evaluation. These results are therefore not an untouched independent "
    "test evaluation.")
add()
add("## 3. Recursive multi-step aggregate metrics")
add()
add("Source: `multi_step_forecasting_v3_fixed/multi_step_metrics.csv`")
add()
add("```text")
add(table(multi))
add("```")
add()
add("These metrics are measured in preprocessed model feature space, "
    "not raw network units.")
add()
add("## 4. Feature-wise forecasting")
add()
add(f"- Feature-wise records: {len(feature):,}")
add(f"- Unique model features: {feature['Feature'].nunique()}")
add("- Forecast horizons: 5")
add("- Label groups in the source experiment: ALL, Benign, Bot, NO_FLOWS")
add()
add(
    "For three CWE Flag Count features, persistence MAE is exactly zero. "
    "Their percentage improvement is undefined, although their absolute "
    "errors remain available. They must not be assigned an artificial "
    "percentage."
)
add()
add("## 5. Feature-family organization")
add()
counts = (
    mapping.groupby("Feature_Family")["Feature"]
    .nunique()
    .reset_index(name="Feature_Count")
)
add(table(counts))
add()
add("The six families are:")
add("1. Timing")
add("2. Packet lengths and headers")
add("3. Packet and flow counts")
add("4. TCP flags")
add("5. Rates and throughput")
add("6. Protocol and network-state indicators")
add()
add("## 6. Overall feature-family forecasting results")
add()
cols = [
    "Horizon_Seconds", "Feature_Family", "Feature_Count",
    "LSTM_Better_Count", "Persistence_Better_Count",
    "LSTM_Mean_MAE", "Persistence_Mean_MAE",
    "MAE_Improvement_Percent"
]
add(table(families[cols]))
add()
add("Interpretation:")
add(
    "- Timing features show positive macro MAE improvement at all horizons."
)
add(
    "- Packet and flow counts show strong positive improvement at all horizons."
)
add(
    "- Rates and throughput improve at all horizons, with a smaller advantage "
    "at 30 seconds."
)
add(
    "- TCP flags are weaker than persistence at 1 and 2 seconds, then show "
    "positive family-level improvement at longer horizons."
)
add(
    "- Protocol and network-state indicators improve at all horizons, "
    "but their advantage is small at 30 seconds."
)
add()
add("## 7. Benign versus Bot feature-family comparison")
add()
add("Source: `feature_family_benign_bot_comparison_v1.csv`")
add()
cols = [
    "Horizon_Seconds", "Feature_Family",
    "MAE_Improvement_Percent_Benign",
    "MAE_Improvement_Percent_Bot",
    "Bot_Minus_Benign_Improvement"
]
add(table(comparison[cols]))
add()
add("Key observations:")
add(
    "- Packet and flow counts improve over persistence for both labels "
    "at every horizon."
)
add(
    "- Timing remains a strong family for both labels."
)
add(
    "- TCP-flag forecasting is substantially weaker for Bot-labelled "
    "samples at shorter horizons; its family-level improvement becomes "
    "positive at 30 seconds."
)
add(
    "- At 30 seconds, packet-length and header features show a larger "
    "relative improvement for Bot than for Benign samples."
)
add(
    "- Rates and throughput show a smaller advantage for Bot samples, "
    "particularly at 30 seconds."
)
add()
add("These are within-label forecasting comparisons. They do not establish "
    "that a traffic label caused a prediction difference or that a feature "
    "is an attack signature.")
add()
add("## 8. Methodology and verification")
add()
add(
    "- The model architecture and checkpoint were preserved."
)
add(
    "- Recursive predictions were fed into subsequent forecast windows."
)
add(
    "- A common set of forecast origins was used across horizons."
)
add(
    "- Paths crossing segment boundaries or timestamp gaps were excluded."
)
add(
    "- Feature order was taken from the trained checkpoint."
)
add(
    "- Feature-family assignment covered all 209 features exactly once."
)
add(
    "- Family metrics are macro-averages across features; each feature "
    "receives equal weight within its family."
)
add(
    "- The feature-wise aggregate metrics were checked against the original "
    "multi-step results."
)
add("- Aggregate verification: 20/20 PASS.")
add()
add("## 9. Limitations")
add()
add(
    "1. Metrics are in preprocessed model feature space, not raw bytes, "
    "packets, seconds or network rates."
)
add(
    "2. Test data was previously used for exploratory evaluation; results "
    "are not independent untouched test evidence."
)
add(
    "3. The test partition contains Benign and Bot labels. Comparisons "
    "between these groups may reflect traffic composition and capture "
    "conditions as well as label differences."
)
add(
    "4. Feature-family averages can conceal individual feature weaknesses."
)
add(
    "5. Recursive prediction errors can accumulate over longer horizons."
)
add(
    "6. Numerical state forecasting does not establish attack-detection "
    "accuracy, early-warning performance or attack-stage identification."
)
add(
    "7. SHAP explanations, residuals and feature errors describe model "
    "behavior; they are not causal evidence or definitive attack indicators."
)
add()
add("## 10. Conclusion")
add()
add(
    "The V3 LSTM World Model demonstrates lower aggregate forecasting "
    "errors than the persistence baseline across the evaluated recursive "
    "forecast horizons."
)
add(
    "Feature-level analysis shows that performance varies across network "
    "feature families. Timing and packet/flow-count features are "
    "consistently favorable, while TCP-flag forecasting is more "
    "challenging, particularly for Bot-labelled samples at short horizons."
)
add(
    "These findings characterize the current model's future-state "
    "prediction behavior. They do not, by themselves, demonstrate "
    "attack detection or prevention."
)
add()
add("## 11. Reproducibility and artifact locations")
add()
add(f"- Model checkpoint: `{BASE / 'best_model.pt'}`")
add(f"- Multi-step metrics: `{FORECAST / 'multi_step_metrics.csv'}`")
add(f"- Feature-wise metrics: `{ANALYSIS / 'feature_wise_metrics.csv'}`")
add(f"- Aggregate verification: `{ANALYSIS / 'aggregate_verification.csv'}`")
add(f"- Feature-family mapping: `{ANALYSIS / 'feature_family_mapping_v1.csv'}`")
add(f"- Feature-family summary: `{ANALYSIS / 'feature_family_summary_v1.csv'}`")
add(f"- Label-wise summary: `{ANALYSIS / 'feature_family_label_summary_v1.csv'}`")
add(f"- Benign/Bot comparison: `{ANALYSIS / 'feature_family_benign_bot_comparison_v1.csv'}`")
add()
add("Report generated from existing verified experiment artifacts. "
    "No model retraining or architecture modification was performed.")

output = OUT / "sentinelx_v3_forecasting_analysis_report.md"
output.write_text("\n".join(lines), encoding="utf-8")

print("\nREPORT GENERATED")
print(output)
print(f"Size: {output.stat().st_size:,} bytes")
