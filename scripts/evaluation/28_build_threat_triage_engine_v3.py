import json
import pandas as pd
from pathlib import Path

ROOT = Path(".")
MODEL_DIR = ROOT / "Models/lstm_world_model_v3_gpu_run2"

VALIDATION_INPUT = (
    MODEL_DIR / "validation_anomaly_analysis/validation_anomaly_scores.csv"
)
TEST_INPUT = (
    MODEL_DIR / "test_anomaly_evaluation/test_anomaly_scores.csv"
)

OUTPUT_DIR = MODEL_DIR / "triage_engine_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Frozen threshold: validation benign 99th percentile.
THRESHOLD = 1.88139677

def build_triage(input_path, split):
    df = pd.read_csv(input_path)

    required = [
        "Target State Row",
        "Target Timestamp",
        "Target CaptureDay",
        "Segment ID",
        "Anomaly Score"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Anomaly Score"] = pd.to_numeric(
        df["Anomaly Score"], errors="coerce"
    )

    if df["Anomaly Score"].isna().any():
        raise ValueError("Anomaly scores contain missing or invalid values.")

    # Ground-truth labels are deliberately not used in triage decisions.
    triage = df[required].copy()

    triage["Threshold"] = THRESHOLD
    triage["Triage Status"] = triage["Anomaly Score"].apply(
        lambda score:
            "REVIEW_REQUIRED" if score > THRESHOLD
            else "WITHIN_CALIBRATED_RANGE"
    )

    triage["Reason"] = triage["Anomaly Score"].apply(
        lambda score:
            "Prediction residual exceeds frozen validation threshold."
            if score > THRESHOLD
            else "Prediction residual does not exceed frozen validation threshold."
    )

    triage["Interpretation"] = triage["Anomaly Score"].apply(
        lambda score:
            "Unusual prediction error; investigate with additional network evidence."
            if score > THRESHOLD
            else "No residual-based review trigger. This does not guarantee benign activity."
    )

    output_csv = OUTPUT_DIR / f"{split}_triage.csv"
    triage.to_csv(output_csv, index=False)

    flagged = int(
        (triage["Triage Status"] == "REVIEW_REQUIRED").sum()
    )

    summary = {
        "split": split,
        "total_observations": len(triage),
        "review_required": flagged,
        "within_calibrated_range": len(triage) - flagged,
        "review_rate_percent": round(flagged / len(triage) * 100, 4),
        "frozen_threshold": THRESHOLD,
        "threshold_source": "Validation benign 99th percentile",
        "decision_input": "Mean absolute LSTM prediction residual",
        "ground_truth_labels_used_for_decisions": False,
        "limitation": (
            "Residual-based review is not proof of malicious activity. "
            "The threshold showed poor transfer to the held-out test split."
        ),
        "output_file": str(output_csv)
    }

    print(f"\n{split.upper()} TRIAGE")
    print("-" * 55)
    for key, value in summary.items():
        print(f"{key}: {value}")

    return summary


validation_summary = build_triage(VALIDATION_INPUT, "validation")
test_summary = build_triage(TEST_INPUT, "test")

combined = {
    "engine": "SentinelX Threat Triage Engine V1",
    "threshold": THRESHOLD,
    "validation": validation_summary,
    "test": test_summary
}

summary_path = OUTPUT_DIR / "triage_summary.json"

with open(summary_path, "w") as f:
    json.dump(combined, f, indent=4)

print("\n" + "=" * 55)
print("TRIAGE ENGINE BUILD COMPLETE")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Summary: {summary_path}")
print("=" * 55)
