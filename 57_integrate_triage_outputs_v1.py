from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(".")
MODEL = ROOT / "Models/lstm_world_model_v3_gpu_run2"

TRIAGE_DIR = MODEL / "triage_engine_v1"
EXPLAIN_DIR = MODEL / "explainable_triage_v2"
MITRE_FILE = MODEL / "mitre_attack_mapping_v1/mitre_attack_contextual_mapping.csv"

OUTPUT_DIR = MODEL / "integrated_triage_v1"

KEYS = [
    "Target State Row",
    "Target Timestamp",
    "Target CaptureDay",
    "Segment ID",
]


def require_columns(df, columns, name):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def main():
    print("=" * 72)
    print("SENTINELX — INTEGRATED THREAT TRIAGE V1")
    print("=" * 72)

    # Never overwrite previous experiment outputs.
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output directory already exists: {OUTPUT_DIR}\n"
            "Preserve it. Choose a new versioned output directory "
            "before running another build."
        )

    validation_path = TRIAGE_DIR / "validation_triage.csv"
    test_path = TRIAGE_DIR / "test_triage.csv"
    explanation_path = EXPLAIN_DIR / "explainable_triage.csv"

    for path in [
        validation_path,
        test_path,
        explanation_path,
        MITRE_FILE,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    validation = pd.read_csv(validation_path)
    test = pd.read_csv(test_path)
    explanations = pd.read_csv(explanation_path)
    mitre = pd.read_csv(MITRE_FILE)

    triage_columns = KEYS + [
        "Anomaly Score",
        "Threshold",
        "Triage Status",
        "Reason",
        "Interpretation",
    ]

    explanation_columns = KEYS + [
        "Prediction MAE (model space)",
        "Previous 5-step Median MAE",
        "5-step Rolling Mean MAE",
        "Change vs Previous Median",
    ]

    require_columns(validation, triage_columns, "Validation triage")
    require_columns(test, triage_columns, "Test triage")
    require_columns(explanations, explanation_columns, "Explanations")

    require_columns(
        mitre,
        ["Dataset_Label", "technique_id", "technique_name", "status",
         "confidence", "rationale", "evidence_needed"],
        "MITRE mapping",
    )

    # Verify key uniqueness before joining.
    for name, frame in [
        ("validation triage", validation),
        ("test triage", test),
        ("explanations", explanations),
    ]:
        if frame[KEYS].isna().any().any():
            raise ValueError(f"{name} contains missing observation keys.")

        if frame.duplicated(KEYS).any():
            raise ValueError(f"{name} contains duplicate observation keys.")

    # Verify exact one-to-one test alignment.
    alignment = test[KEYS].merge(
        explanations[KEYS],
        on=KEYS,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    unmatched = alignment["_merge"] != "both"
    if unmatched.any():
        raise ValueError(
            "Test triage and explanations do not align exactly: "
            f"{int(unmatched.sum())} unmatched records."
        )

    # Scores must describe the same residual calculation.
    if not np.allclose(
        test["Anomaly Score"].to_numpy(dtype=np.float64),
        explanations["Prediction MAE (model space)"].to_numpy(
            dtype=np.float64
        ),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError(
            "Anomaly Score and explanation MAE do not match."
        )

    # Join explanations to the residual-based triage records.
    integrated = test.merge(
        explanations.drop(
            columns=["Target CaptureDay"],
            errors="ignore",
        ),
        on=["Target State Row", "Target Timestamp", "Segment ID"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_explanation"),
    )

    if len(integrated) != len(test):
        raise ValueError("Integrated output row count changed.")

    if integrated["Prediction MAE (model space)"].isna().any():
        raise ValueError("Integrated output contains missing explanations.")

    # Explicitly distinguish forecast evidence from attack conclusions.
    integrated["Evidence Type"] = "LSTM next-state prediction residual"
    integrated["Security Interpretation"] = (
        "Forecast deviation only; not proof of malicious activity."
    )
    integrated["Analyst Action"] = np.where(
        integrated["Triage Status"] == "REVIEW_REQUIRED",
        "Review with independent network and host evidence",
        "No residual-based review trigger; other security controls remain necessary",
    )

    integrated["MITRE Mapping Policy"] = (
        "Dataset-label contextual reference only; not assigned "
        "to individual observations by this engine."
    )

    # Keep validation and test statistics separate.
    validation_counts = validation["Triage Status"].value_counts().to_dict()
    test_counts = test["Triage Status"].value_counts().to_dict()

    validation_flagged = int(
        (validation["Triage Status"] == "REVIEW_REQUIRED").sum()
    )
    test_flagged = int(
        (test["Triage Status"] == "REVIEW_REQUIRED").sum()
    )

    # Preserve the MITRE mapping as a separate reference artifact.
    # Do not join dataset labels to individual triage observations.
    mitre_reference = {
        "file": str(MITRE_FILE),
        "mapping_rows": int(len(mitre)),
        "policy": (
            "Contextual reference only. Dataset labels do not establish "
            "that a specific residual or observation corresponds to "
            "an ATT&CK technique."
        ),
    }

    summary = {
        "engine": "SentinelX Integrated Threat Triage V1",
        "status": "INTEGRATION_COMPLETE",
        "test_observations": int(len(integrated)),
        "test_alignment": {
            "exact_key_matches": int(len(alignment)),
            "unmatched_records": int(unmatched.sum()),
            "duplicate_keys": 0,
            "residual_score_match": True,
        },
        "validation": {
            "observations": int(len(validation)),
            "review_required": validation_flagged,
            "within_calibrated_range": (
                int(len(validation)) - validation_flagged
            ),
            "status_counts": {
                str(k): int(v) for k, v in validation_counts.items()
            },
        },
        "test": {
            "observations": int(len(test)),
            "review_required": test_flagged,
            "within_calibrated_range": int(len(test)) - test_flagged,
            "status_counts": {
                str(k): int(v) for k, v in test_counts.items()
            },
        },
        "decision_method": {
            "input": "Mean absolute next-state prediction residual",
            "threshold": float(test["Threshold"].iloc[0]),
            "threshold_source": "Validation benign 99th percentile",
            "ground_truth_labels_used_for_decisions": False,
            "model_changed": False,
        },
        "explainability": {
            "feature_contributors_per_observation": 5,
            "temporal_context": (
                "Previous five observations within the same segment."
            ),
            "comparison_space": "Preprocessed model feature space",
            "causal_explanation": False,
        },
        "mitre_context": mitre_reference,
        "limitations": [
            "A high prediction residual is not proof of malicious activity.",
            "A low residual does not guarantee benign activity.",
            "The frozen threshold showed poor transfer to the test split.",
            "The residual threshold is a review heuristic, not a validated attack detector.",
            "MITRE mappings are contextual dataset-label references, "
            "not per-observation attack classifications.",
            "Independent network or host evidence is required for security conclusions.",
        ],
        "outputs": {
            "integrated_test_triage": "integrated_test_triage.csv",
            "summary": "integrated_triage_summary.json",
            "mitre_reference": str(MITRE_FILE),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    integrated_path = OUTPUT_DIR / "integrated_test_triage.csv"
    summary_path = OUTPUT_DIR / "integrated_triage_summary.json"

    integrated.to_csv(integrated_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nINTEGRATION RESULTS")
    print("-" * 55)
    print("Test observations:", len(integrated))
    print("Exact matched records:", len(alignment))
    print("Unmatched records:", int(unmatched.sum()))
    print("Validation review required:", validation_flagged)
    print("Test review required:", test_flagged)
    print("Model/checkpoint modified: No")
    print("\nIntegrated output:", integrated_path)
    print("Summary:", summary_path)
    print("\nINTEGRATION COMPLETE")


if __name__ == "__main__":
    main()
