"""SentinelX domain services.

Every function here returns plain dictionaries that already carry provenance.
Routers stay thin; all interpretation lives here so it can be unit tested
without HTTP.
"""

from __future__ import annotations

import logging
from statistics import mean

from . import artifacts as A
from .config import (
    MODELS_DIR,
    MODEL_VERSIONS,
    RECORDED_ANOMALY_THRESHOLD,
    RECORDED_THRESHOLD_PERCENTILE,
)
from .provenance import DERIVED, RECORDED, block, unavailable
from .state import get_active_model, load_settings

log = logging.getLogger("sentinelx.services")

MODEL_ARTIFACTS = {
    "training_report": ("training_report.txt", "Training report"),
    "test_evaluation_report": ("test_evaluation_report.txt", "Held-out test evaluation"),
    "config": ("config.json", "Model configuration"),
    "multi_step": ("multi_step_forecasting_v3_fixed/multi_step_forecasting_report.txt",
                   "Recursive multi-step forecasting"),
    "feature_family": ("multi_step_feature_analysis_v1/feature_family_report_v1.txt",
                       "Feature-family forecasting"),
    "test_evaluation": ("test_evaluation/test_evaluation_summary.json", "Test evaluation summary"),
    "test_anomaly": ("test_anomaly_evaluation/test_anomaly_summary.json", "Test anomaly evaluation"),
    "validation_anomaly": ("validation_anomaly_analysis/anomaly_analysis_summary.json",
                           "Validation anomaly analysis"),
    "triage": ("triage_engine_v1/triage_summary.json", "Triage engine summary"),
    "integrated_triage": ("integrated_triage_v1/integrated_triage_summary.json",
                          "Integrated triage"),
    "shap_v3": ("shap_explainability_v3/summary.json", "SHAP (benign reference)"),
    "explanation_validation": ("explanation_validation_v1/summary.json",
                               "Explanation validation"),
    "explainable_triage": ("explainable_triage_v2/summary.json", "Explainable triage"),
    "attack_diagnostics": ("validation_attack_diagnostics_v1/summary.json",
                           "Validation attack diagnostics"),
    "run_structure": ("../lstm_world_model_v1/label_run_structure_v1/"
                      "label_run_structure_report.txt", "Label run structure"),
}

# Binary / dataset artifacts that are deliberately excluded from version control.
BINARY_ARTIFACTS = {
    "checkpoint": ("best_model.pt", "LSTM checkpoint (PyTorch)"),
    "scaler": ("scaler.joblib", "Fitted StandardScaler"),
    "sequences": ("sentinelx_test_X_seq10.npy", "Sequence arrays"),
    "triage_csv": ("triage_engine_v1/test_triage.csv", "Per-observation triage rows"),
}


def _artifact_path(model_dir: str, key: str) -> str:
    return f"Models/{model_dir}/{MODEL_ARTIFACTS[key][0]}"


# --------------------------------------------------------------------------
# Health / registry
# --------------------------------------------------------------------------

def artifact_registry() -> list[dict]:
    out: list[dict] = []
    for key, (rel_path, label) in MODEL_ARTIFACTS.items():
        for mid, meta in MODEL_VERSIONS.items():
            path = MODELS_DIR / meta["model_dir"] / rel_path
            out.append(
                {
                    "id": f"{mid}:{key}",
                    "label": label,
                    "path": A.rel(path),
                    "present": path.is_file(),
                    "required_for": mid.upper(),
                    "provenance_kind": "json" if rel_path.endswith(".json") else "text",
                }
            )
    # Binary artifacts are checked once against the active model folder.
    active = get_active_model()
    for key, (rel_path, label) in BINARY_ARTIFACTS.items():
        path = MODELS_DIR / MODEL_VERSIONS[active]["model_dir"] / rel_path
        out.append(
            {
                "id": f"{active}:{key}",
                "label": label,
                "path": A.rel(path),
                "present": path.is_file(),
                "required_for": "inference",
                "provenance_kind": "binary",
            }
        )
    return out


def model_statuses() -> list[dict]:
    out = []
    for mid, meta in MODEL_VERSIONS.items():
        ckpt = MODELS_DIR / meta["model_dir"] / meta["checkpoint"]
        present = ckpt.is_file()
        out.append(
            {
                "id": mid,
                "name": meta["name"],
                "input_features": meta["input_features"],
                "sequence_length": meta["sequence_length"],
                "status": meta["status"],
                "checkpoint_present": present,
                "checkpoint_path": A.rel(ckpt),
                "loadable": present,
                "note": meta["note"],
            }
        )
    return out


def health() -> dict:
    registry = artifact_registry()
    models = model_statuses()
    missing_binary = [a for a in registry if not a["present"] and a["provenance_kind"] == "binary"]
    warnings: list[str] = []
    if missing_binary:
        warnings.append(
            f"{len(missing_binary)} binary/dataset artifact(s) required for live inference "
            "are not present in this checkout (checkpoints, scaler, sequence arrays). "
            "Recorded research results are still fully available."
        )
    if not any(m["loadable"] for m in models):
        warnings.append(
            "No LSTM checkpoint is loadable, so on-demand inference is disabled. "
            "Every metric shown comes from recorded evaluation artifacts."
        )
    present = sum(1 for a in registry if a["present"])
    return {
        "status": "ok" if present else "degraded",
        "service": "sentinelx-api",
        "version": "4.0.0",
        "data_mode": "offline_dataset",
        "inference_available": any(m["loadable"] for m in models),
        "live_telemetry_available": False,
        "artifacts": registry,
        "models": models,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------
# Model metadata
# --------------------------------------------------------------------------

def _model_config(mid: str) -> tuple[dict | None, str | None]:
    meta = MODEL_VERSIONS[mid]
    cfg = A.model_json(meta["model_dir"], "config.json")
    if cfg is None:
        cfg = A.model_json(meta["model_dir"], "training_report.txt")
    return cfg, cfg and _artifact_path(meta["model_dir"], "config")


def training_report_parsed(mid: str) -> dict | None:
    """Extract headline training figures from a plain-text training report."""
    meta = MODEL_VERSIONS[mid]
    text = A.model_report(meta["model_dir"], "training_report.txt")
    if not text:
        return None
    import re

    def grab(pattern: str, cast=float):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    return {
        "input_features": grab(r"Input features:\s*(\d+)", int)
        or grab(r"Input sequence:\s*10\s*x\s*(\d+)", int),
        "sequence_length": grab(r"Sequence length:\s*(\d+)", int) or 10,
        "hidden_size": grab(r"Hidden size:\s*(\d+)", int),
        "num_layers": grab(r"LSTM layers:\s*(\d+)", int),
        "dropout": grab(r"Dropout:\s*([\d.]+)"),
        "training_samples": grab(r"Training (?:samples|sequences):\s*([\d,]+)", lambda v: int(v.replace(",", ""))),
        "validation_samples": grab(r"Validation (?:samples|sequences):\s*([\d,]+)", lambda v: int(v.replace(",", ""))),
        "best_epoch": grab(r"Best epoch:\s*(\d+)", int),
        "best_validation_huber": grab(r"Best validation Huber loss:\s*([\d.]+)"),
        "training_seconds": grab(r"Training time:\s*([\d.]+)"),
        "batch_size": grab(r"Batch size:\s*(\d+)", int),
        "optimizer": (re.search(r"Optimizer:\s*(\w+)", text) or [None, None])[1],
        "loss": (re.search(r"Loss:\s*(.+?)\s*$", text, re.MULTILINE) or [None, None])[1],
        "raw": text,
    }


def models_overview() -> dict:
    active = get_active_model()
    versions = []
    for mid, meta in MODEL_VERSIONS.items():
        cfg = A.model_json(meta["model_dir"], "config.json")
        rep = training_report_parsed(mid)
        test = A.model_json(meta["model_dir"], "test_evaluation/test_evaluation_summary.json")
        if test is None:
            # V2 stores its evaluation as text; parse the summary metrics.
            text = A.model_report(meta["model_dir"], "test_evaluation_report.txt")
            if text:
                import re
                test = {
                    "lstm_metrics": {
                        "huber_loss": float(m.group(1)),
                        "mae": float(m.group(2)),
                        "rmse": float(m.group(3)),
                    }
                    for m in [re.search(r"HUBER:\s*([\d.]+)\s*\nMAE:\s*([\d.]+)\s*\nRMSE:\s*([\d.]+)", text)]
                    if m
                }
                test["_source"] = A.rel(
                    MODELS_DIR / meta["model_dir"] / "test_evaluation_report.txt"
                )
        versions.append(
            {
                "id": mid,
                "name": meta["name"],
                "status": meta["status"],
                "is_active": mid == active,
                "input_features": meta["input_features"],
                "sequence_length": meta["sequence_length"],
                "note": meta["note"],
                "checkpoint_present": (MODELS_DIR / meta["model_dir"] / meta["checkpoint"]).is_file(),
                "config": cfg,
                "training": rep,
                "test_evaluation": test,
            }
        )
    return {
        "active_model": active,
        "versions": versions,
        "comparison_policy": (
            "V2 (212) and V3 (209) use different feature definitions, scalers and "
            "sequence arrays. Their numbers are reported side by side for reference "
            "only. No claim is made that either version is more accurate without a "
            "controlled, like-for-like evaluation."
        ),
        "provenance": RECORDED,
    }


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------

def _v3_test_metrics() -> dict | None:
    return A.model_json(MODEL_VERSIONS["v3"]["model_dir"],
                        "test_evaluation/test_evaluation_summary.json")


def _num(val):
    return None if val is None else float(val)


def dashboard() -> dict:
    active = get_active_model()
    v3 = _v3_test_metrics()
    settings = load_settings()
    lstm = (v3 or {}).get("lstm_metrics") or {}
    base = (v3 or {}).get("persistence_baseline_metrics") or {}
    imp = (v3 or {}).get("improvement_over_baseline_percent") or {}
    src = "Models/lstm_world_model_v3_gpu_run2/test_evaluation/test_evaluation_summary.json"

    metrics = [
        {"key": "test_mae", "label": "Test MAE", "value": _num(lstm.get("mae")),
         "unit": "model-space", "provenance": RECORDED if lstm else "unavailable",
         "source": src, "hint": "Mean absolute next-state error, held-out test split"},
        {"key": "test_rmse", "label": "Test RMSE", "value": _num(lstm.get("rmse")),
         "unit": "model-space", "provenance": RECORDED if lstm else "unavailable",
         "source": src, "hint": "Root mean squared next-state error"},
        {"key": "test_huber", "label": "Test Huber", "value": _num(lstm.get("huber_loss")),
         "unit": "model-space", "provenance": RECORDED if lstm else "unavailable",
         "source": src, "hint": "Huber loss used for training/selection"},
        {"key": "vs_persistence_mae", "label": "MAE vs persistence",
         "value": _num(imp.get("mae")), "unit": "%", "provenance": RECORDED if imp else "unavailable",
         "source": src, "hint": "Positive means the LSTM beats repeating the last state"},
        {"key": "test_samples", "label": "Test sequences", "value": (v3 or {}).get("test_samples"),
         "provenance": RECORDED if v3 else "unavailable", "source": src,
         "hint": "Evaluated sequences in the held-out test partition"},
        {"key": "input_features", "label": "Input features",
         "value": (v3 or {}).get("input_features"), "provenance": RECORDED if v3 else "unavailable",
         "source": src, "hint": "Feature width of the active model"},
    ]

    # Detection assessment is a first-class, deliberately negative result.
    ta = A.model_json(MODEL_VERSIONS["v3"]["model_dir"],
                      "test_anomaly_evaluation/test_anomaly_summary.json")
    va = A.model_json(MODEL_VERSIONS["v3"]["model_dir"],
                      "validation_anomaly_analysis/anomaly_analysis_summary.json")
    detection = {
        "validated_detector": False,
        "headline": "Residual threshold is a review heuristic, not a validated detector",
        "threshold": (ta or va or {}).get("threshold") or RECORDED_ANOMALY_THRESHOLD,
        "threshold_source": (ta or va or {}).get("threshold_source")
        or "Frozen 99th percentile of benign validation scores",
        "provenance": RECORDED if (ta or va) else "unavailable",
        "test_metrics": (ta or {}).get("detection_metrics_benign_vs_bot"),
        "validation_metrics": (va or {}).get("detection_metrics_benign_vs_infilteration"),
        "verdict": (
            "On the held-out test split the frozen threshold produced 0 true positives "
            "against 20,216 Bot-labelled sequences. It therefore cannot be described as "
            "an attack detector. It is surfaced as a triage review signal only."
        ),
        "source": "Models/lstm_world_model_v3_gpu_run2/test_anomaly_evaluation/test_anomaly_summary.json",
    }

    coverage_text = A.repo_report("mitre_attack_contextual_mapping_v1",
                                  "dataset_label_coverage_report.txt")
    coverage = A.parse_label_coverage(coverage_text) if coverage_text else {}
    label_distribution = []
    for partition in ("train", "validation", "test"):
        for label, count in sorted(coverage.get(partition, {}).items(),
                                   key=lambda kv: -kv[1]):
            label_distribution.append(
                {"partition": partition, "label": label, "count": count,
                 "provenance": RECORDED}
            )

    rs_text = A.model_report(MODEL_VERSIONS["v3"]["model_dir"],
                             "../lstm_world_model_v1/label_run_structure_v1/"
                             "label_run_structure_report.txt")
    runs = A.parse_run_structure(rs_text) if rs_text else {"runs": [], "total_runs": None}
    triage = A.model_json(MODEL_VERSIONS["v3"]["model_dir"],
                          "triage_engine_v1/triage_summary.json")

    recent_signals = []
    if triage:
        for split in ("validation", "test"):
            s = triage.get(split) or {}
            if s:
                recent_signals.append(
                    {
                        "kind": "triage_aggregate",
                        "split": split,
                        "review_required": s.get("review_required"),
                        "total": s.get("total_observations"),
                        "review_rate_percent": s.get("review_rate_percent"),
                        "provenance": RECORDED,
                    }
                )
    if runs.get("total_runs"):
        recent_signals.append(
            {
                "kind": "capture_run_structure",
                "split": "test",
                "total_runs": runs["total_runs"],
                "total_states": runs["total_states"],
                "window": runs.get("timestamp_range"),
                "provenance": RECORDED,
            }
        )

    posture = {
        "level": "review",
        "label": "Baseline established",
        "detail": (
            f"Active model {active.upper()} reproduces a documented predictive baseline. "
            "Detection is unvalidated and no live telemetry is connected."
        ),
        "provenance": DERIVED,
    }

    return {
        "active_model": active,
        "security_posture": posture,
        "metrics": metrics,
        "label_distribution": label_distribution,
        "detection_assessment": detection,
        "recent_signals": recent_signals,
        "system_health": {
            "inference_available": bool(v3) and any(m["loadable"] for m in model_statuses()),
            "live_telemetry": False,
            "data_mode": "offline_dataset",
            "settings": settings.model_dump(),
        },
        "limitations": [
            "Prediction error is not evidence of an attack.",
            "The frozen residual threshold failed to transfer to the test split.",
            "Feature attributions explain model behaviour, not causality.",
            "MITRE ATT&CK entries are contextual label references, not per-event classifications.",
            "No live network telemetry is ingested; all figures are offline dataset results.",
        ],
    }


# --------------------------------------------------------------------------
# Network monitoring
# --------------------------------------------------------------------------

def network_overview() -> dict:
    text = A.repo_report("mitre_attack_contextual_mapping_v1",
                         "dataset_label_coverage_report.txt")
    if not text:
        return {
            "available": False,
            "provenance": "unavailable",
            "labels": [],
            "partitions": {},
            "reason": "Dataset label coverage report is missing.",
        }
    coverage = A.parse_label_coverage(text)
    partitions: dict[str, dict] = {}
    for name, counts in coverage.items():
        total = sum(counts.values())
        labelled_attack = {k: v for k, v in counts.items()
                          if k not in {"Benign", "NO_FLOWS", "MIXED_ATTACK"}}
        attack = sum(labelled_attack.values())
        partitions[name] = {
            "total": total,
            "benign": counts.get("Benign", 0),
            "no_flows": counts.get("NO_FLOWS", 0),
            "labelled_attack": attack,
            "attack_share_percent": round(attack / total * 100, 2) if total else 0.0,
            "distinct_attack_labels": len(labelled_attack),
        }
    labels = []
    for name, counts in coverage.items():
        for label, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            labels.append({"partition": name, "label": label, "count": count})
    return {
        "available": True,
        "provenance": RECORDED,
        "source": "reports/mitre_attack_contextual_mapping_v1/dataset_label_coverage_report.txt",
        "partitions": partitions,
        "labels": labels,
        "flow_level_detail": unavailable(
            "Per-flow source/destination IP, port and protocol fields are not present "
            "in this checkout. The CICFlowMeter CSVs are excluded by .gitignore, so "
            "flow-level drill-down is unavailable.",
            requires=["Datasets/*.csv"],
        ),
    }


# --------------------------------------------------------------------------
# Predictive intelligence
# --------------------------------------------------------------------------

def predictions() -> dict:
    v3 = _v3_test_metrics()
    rel = A.repo_json("v3_reliability_audit_v1", "reliability_summary.json")
    ms_text = A.model_report(MODEL_VERSIONS["v3"]["model_dir"],
                             "multi_step_forecasting_v3_fixed/multi_step_forecasting_report.txt")
    ms_rows = A.parse_multistep_report(ms_text) if ms_text else []
    v2 = models_overview()["versions"][1].get("test_evaluation")

    return {
        "one_step": {
            "model": (v3 or {}).get("lstm_metrics"),
            "persistence": (v3 or {}).get("persistence_baseline_metrics"),
            "improvement_percent": (v3 or {}).get("improvement_over_baseline_percent"),
            "samples": (v3 or {}).get("test_samples"),
            "provenance": RECORDED if v3 else "unavailable",
            "source": "Models/lstm_world_model_v3_gpu_run2/test_evaluation/test_evaluation_summary.json",
        },
        "multi_step": {
            "rows": ms_rows,
            "horizons": sorted({r["horizon_seconds"] for r in ms_rows}),
            "provenance": RECORDED if ms_rows else "unavailable",
            "source": "Models/lstm_world_model_v3_gpu_run2/multi_step_forecasting_v3_fixed/"
                      "multi_step_forecasting_report.txt",
            "note": (
                "Recursive forecasting: each predicted state is fed back as input. "
                "The test partition was used in earlier exploratory analysis and is not "
                "an untouched independent test."
            ),
        },
        "error_distribution": {
            "quantiles": (rel or {}).get("overall_error_quantiles"),
            "samples": (rel or {}).get("samples"),
            "features": (rel or {}).get("features"),
            "provenance": RECORDED if rel else "unavailable",
            "source": "reports/v3_reliability_audit_v1/reliability_summary.json",
            "note": "Descriptive error quantiles only. They are not detection thresholds.",
        },
        "v2_reference": {
            "metrics": (v2 or {}).get("lstm_metrics") if isinstance(v2, dict) else None,
            "provenance": RECORDED if v2 else "unavailable",
            "note": "Different feature set (212) from V3 (209). Reference only.",
        },
        "live_forecast": unavailable(
            "On-demand next-state forecasting requires the trained checkpoint and the "
            "fitted 209-feature scaler, neither of which is present in this checkout.",
            requires=["Models/lstm_world_model_v3_gpu_run2/best_model.pt",
                      "sentinelx_standard_scaler_v1.joblib"],
        ),
    }


# --------------------------------------------------------------------------
# Triage
# --------------------------------------------------------------------------

def triage() -> dict:
    triage_summary = A.model_json(MODEL_VERSIONS["v3"]["model_dir"],
                                 "triage_engine_v1/triage_summary.json")
    integrated = A.model_json(MODEL_VERSIONS["v3"]["model_dir"],
                             "integrated_triage_v1/integrated_triage_summary.json")
    settings = load_settings()

    aggregates = []
    for split in ("validation", "test"):
        s = (triage_summary or {}).get(split)
        if s:
            aggregates.append(
                {
                    "split": split,
                    "total_observations": s.get("total_observations"),
                    "review_required": s.get("review_required"),
                    "within_calibrated_range": s.get("within_calibrated_range"),
                    "review_rate_percent": s.get("review_rate_percent"),
                    "frozen_threshold": s.get("frozen_threshold"),
                    "decision_input": s.get("decision_input"),
                    "labels_used": s.get("ground_truth_labels_used_for_decisions"),
                }
            )

    return {
        "engine": (integrated or {}).get("engine") or "SentinelX Threat Triage Engine V1",
        "status": (integrated or {}).get("status") or ("recorded" if aggregates else "unavailable"),
        "aggregates": aggregates,
        "provenance": RECORDED if aggregates else "unavailable",
        "source": "Models/lstm_world_model_v3_gpu_run2/triage_engine_v1/triage_summary.json",
        "calibration": {
            "threshold": RECORDED_ANOMALY_THRESHOLD,
            "percentile": RECORDED_THRESHOLD_PERCENTILE,
            "source_description": "Frozen 99th percentile of benign validation scores",
            "recalibrated_on_test": False,
            "user_threshold_override": settings.anomaly_review_threshold,
            "note": "The user override is a UI preference; recorded results keep the frozen value.",
        },
        "decision_policy": {
            "input": "Mean absolute next-state prediction residual across 209 features",
            "ground_truth_labels_used": False,
            "threshold_based_attack_decision": False,
        },
        "integrity": (integrated or {}).get("test_alignment"),
        "per_event_rows": unavailable(
            "Per-observation triage rows (triage CSV) are generated by the research "
            "pipeline and excluded from version control. Aggregate review counts are "
            "recorded and shown above.",
            requires=["Models/lstm_world_model_v3_gpu_run2/triage_engine_v1/test_triage.csv"],
        ),
        "limitations": (integrated or {}).get("limitations") or [
            "A high prediction residual is not proof of malicious activity.",
            "A low residual does not guarantee benign activity.",
            "The residual threshold is a review heuristic, not a validated attack detector.",
        ],
        "recommended_actions": [
            "Correlate the review window with authentication, DNS and proxy logs.",
            "Check whether the same hosts show repeated failed logins or unusual fan-out.",
            "Compare the current window against the same time-of-day baseline.",
            "Record analyst notes and escalate only with corroborating evidence.",
        ],
    }


# --------------------------------------------------------------------------
# Explainability
# --------------------------------------------------------------------------

def explainability() -> dict:
    v3 = A.model_json(MODEL_VERSIONS["v3"]["model_dir"],
                      "shap_explainability_v3/summary.json")
    v2 = A.model_json(MODEL_VERSIONS["v3"]["model_dir"],
                      "shap_explainability_v2/summary.json")
    v1 = A.model_json(MODEL_VERSIONS["v3"]["model_dir"],
                      "shap_explainability_v1/shap_experiment_summary.json")
    ev = A.model_json(MODEL_VERSIONS["v3"]["model_dir"],
                      "explanation_validation_v1/summary.json")
    et = A.model_json(MODEL_VERSIONS["v3"]["model_dir"],
                      "explainable_triage_v2/summary.json")
    fam_text = A.model_report(MODEL_VERSIONS["v3"]["model_dir"],
                              "multi_step_feature_analysis_v1/feature_family_report_v1.txt")
    families = A.parse_feature_family_report(fam_text) if fam_text else []

    experiments = []
    for label, payload in (("SHAP V1", v1), ("SHAP V2", v2), ("SHAP V3", v3)):
        if payload:
            experiments.append({"name": label, "summary": payload, "provenance": RECORDED})

    return {
        "method": (v3 or v2 or v1 or {}).get("method"),
        "interpretation": (v3 or {}).get("interpretation"),
        "experiments": experiments,
        "output_features": (v3 or v2 or v1 or {}).get("output_features") or [],
        "feature_families": families,
        "error_quantiles": (ev or {}).get("split_summary"),
        "local_explanations": {
            "feature_contributors_per_observation": (et or {}).get("top_features_per_observation"),
            "temporal_history_window": (et or {}).get("temporal_history_window"),
            "temporal_comparison": (et or {}).get("temporal_comparison"),
            "provenance": RECORDED if et else "unavailable",
        },
        "provenance": RECORDED if experiments else "unavailable",
        "shap_values": unavailable(
            "SHAP value tensors were written as .npz/.csv outputs which are excluded "
            "from version control. The recorded experiment summaries and their "
            "methodology are shown instead.",
            requires=["Models/lstm_world_model_v3_gpu_run2/shap_explainability_v3/*.npz"],
        ),
        "caveats": [
            "SHAP values attribute model output to inputs; they do not establish causality.",
            "Attributions are relative to a benign training background, not attack probability.",
            "Explanations describe behaviour in preprocessed model feature space.",
        ],
    }


# --------------------------------------------------------------------------
# Attack timeline
# --------------------------------------------------------------------------

def timeline() -> dict:
    text = A.model_report(MODEL_VERSIONS["v3"]["model_dir"],
                          "../lstm_world_model_v1/label_run_structure_v1/"
                          "label_run_structure_report.txt")
    if not text:
        return {
            "available": False,
            "reason": "Label run structure report is missing.",
            "provenance": "unavailable",
            "window": None,
            "runs": [],
            "histogram": [],
        }
    parsed = A.parse_run_structure(text)
    transitions = A.model_report("lstm_world_model_v1",
                                "transition_forecast_analysis_v1/transition_forecast_report.txt")
    continuity = A.model_report("lstm_world_model_v1",
                                "label_persistence_baseline_v1/label_persistence_report.txt")
    return {
        "available": True,
        "provenance": RECORDED,
        "source": "Models/lstm_world_model_v1/label_run_structure_v1/"
                  "label_run_structure_report.txt",
        "window": parsed.get("timestamp_range"),
        "total_states": parsed.get("total_states"),
        "total_runs": parsed.get("total_runs"),
        "runs": parsed.get("runs", []),
        "histogram": parsed.get("histogram", []),
        "transition_analysis_available": bool(transitions),
        "continuity_analysis_available": bool(continuity),
        "note": (
            "Runs are contiguous stretches of a single dataset label in the test "
            "capture. They show observed label persistence, not confirmed attack "
            "progression. Transition and continuity analyses are present as "
            "recorded reports where available."
        ),
    }


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------

DATASET_EXPECTED_COLUMNS = 80
DATASET_SOURCE_ROWS = 1_048_575


def datasets() -> dict:
    coverage_text = A.repo_report("mitre_attack_contextual_mapping_v1",
                                  "dataset_label_coverage_report.txt")
    coverage = A.parse_label_coverage(coverage_text) if coverage_text else {}
    config = A.model_json(MODEL_VERSIONS["v3"]["model_dir"], "config.json")
    return {
        "sources": [
            {
                "id": "cse-cic-ids2018",
                "name": "CSE-CIC-IDS2018",
                "role": "primary",
                "description": "Network-flow dataset (CICFlowMeter) used by the main pipeline.",
                "files": ["02-14-2018.csv", "02-15-2018.csv", "03-02-2018.csv"],
                "original_rows": DATASET_SOURCE_ROWS,
                "original_columns": DATASET_EXPECTED_COLUMNS,
                "present_in_checkout": (MODELS_DIR.parent / "Datasets").is_dir(),
            },
            {
                "id": "ctu-13",
                "name": "CTU-13",
                "role": "secondary",
                "description": "Botnet capture corpus used for additional context.",
                "files": [],
                "present_in_checkout": False,
            },
        ],
        "partitions": coverage,
        "feature_version": {
            "v3_features": (config or {}).get("input_features"),
            "sequence_length": (config or {}).get("sequence_length"),
            "training_samples": (config or {}).get("training_samples"),
            "validation_samples": (config or {}).get("validation_samples"),
            "provenance": RECORDED if config else "unavailable",
        },
        "preprocessing_contract": [
            "Median imputation fitted on training states only.",
            "Log transform for highly skewed non-negative features (threshold 2.0).",
            "StandardScaler fitted on training states, frozen for validation and test.",
            "Sequence length 10, generated within segment boundaries only.",
            "No cross-segment or cross-timestamp-gap sequences.",
        ],
        "upload": {
            "supported": False,
            "reason": (
                "Dataset upload is intentionally disabled in offline mode. A new "
                "dataset must go through the audited preprocessing pipeline so it "
                "cannot silently replace the training artifacts."
            ),
        },
        "raw_files_present": any(
            (MODELS_DIR.parent / "Datasets").glob("*.csv")
        ) if (MODELS_DIR.parent / "Datasets").is_dir() else False,
        "provenance": RECORDED if coverage else "unavailable",
    }


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------

REPORT_CATALOG = [
    ("v3_test_evaluation", "V3 held-out test evaluation",
     "Models/lstm_world_model_v3_gpu_run2/test_evaluation/test_evaluation_summary.json"),
    ("v3_multi_step", "V3 recursive multi-step forecasting",
     "Models/lstm_world_model_v3_gpu_run2/multi_step_forecasting_v3_fixed/multi_step_forecasting_report.txt"),
    ("v3_anomaly_test", "V3 test residual anomaly evaluation",
     "Models/lstm_world_model_v3_gpu_run2/test_anomaly_evaluation/test_anomaly_summary.json"),
    ("v3_anomaly_validation", "V3 validation residual anomaly analysis",
     "Models/lstm_world_model_v3_gpu_run2/validation_anomaly_analysis/anomaly_analysis_summary.json"),
    ("v3_triage", "Threat triage engine summary",
     "Models/lstm_world_model_v3_gpu_run2/triage_engine_v1/triage_summary.json"),
    ("v3_shap", "SHAP explainability experiment",
     "Models/lstm_world_model_v3_gpu_run2/shap_explainability_v3/summary.json"),
    ("v3_reliability", "Rare-error reliability audit",
     "reports/v3_reliability_audit_v1/reliability_summary.json"),
    ("v2_training", "V2 training report", "Models/lstm_world_model_v2/training_report.txt"),
    ("v2_test", "V2 test evaluation", "Models/lstm_world_model_v2/test_evaluation_report.txt"),
    ("v1_runs", "Label run structure",
     "Models/lstm_world_model_v1/label_run_structure_v1/label_run_structure_report.txt"),
]


def report_catalog() -> list[dict]:
    from .config import REPO_ROOT

    out = []
    for rid, title, rel_path in REPORT_CATALOG:
        path = REPO_ROOT / rel_path
        out.append(
            {
                "id": rid,
                "title": title,
                "path": rel_path,
                "present": path.is_file(),
                "kind": "json" if rel_path.endswith(".json") else "text",
            }
        )
    return out


def build_report_markdown(title: str, sections: list[dict]) -> str:
    lines = [f"# {title}", ""]
    lines.append("Generated by SentinelX from recorded research artifacts.")
    lines.append("All numeric values are copied from committed artifacts; nothing is simulated.")
    lines.append("")
    for sec in sections:
        lines.append(f"## {sec['title']}")
        lines.append("")
        if "prose" in sec:
            lines.append(sec["prose"])
            lines.append("")
        table = sec.get("table")
        if table and table.get("columns"):
            cols = table["columns"]
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("|" + "|".join(["---"] * len(cols)) + "|")
            for row in table["rows"]:
                lines.append("| " + " | ".join(str(c) for c in row) + " |")
            lines.append("")
        if sec.get("source"):
            lines.append(f"_Source: {sec['source']}_")
            lines.append("")
    return "\n".join(lines)


def generate_summary_report() -> dict:
    import datetime as _dt

    dash = dashboard()
    pred = predictions()
    tri = triage()
    v3 = _v3_test_metrics() or {}
    lstm = v3.get("lstm_metrics") or {}
    base = v3.get("persistence_baseline_metrics") or {}
    imp = v3.get("improvement_over_baseline_percent") or {}

    sections = [
        {
            "title": "Predictive model performance (one-step, held-out test)",
            "table": {
                "columns": ["Metric", "LSTM", "Persistence", "Improvement %"],
                "rows": [
                    ["Huber", lstm.get("huber_loss"), base.get("huber_loss"), imp.get("huber_loss")],
                    ["MAE", lstm.get("mae"), base.get("mae"), imp.get("mae")],
                    ["RMSE", lstm.get("rmse"), base.get("rmse"), imp.get("rmse")],
                ],
            },
            "source": "Models/lstm_world_model_v3_gpu_run2/test_evaluation/test_evaluation_summary.json",
        },
        {
            "title": "Multi-step recursive forecasting (ALL labels)",
            "table": {
                "columns": ["Horizon (s)", "LSTM MAE", "Persistence MAE", "MAE improvement %"],
                "rows": [
                    [r["horizon_seconds"], r["lstm_mae"], r["persistence_mae"],
                     r["mae_improvement_percent"]]
                    for r in pred["multi_step"]["rows"] if r["target_label"] == "ALL"
                ],
            },
            "source": pred["multi_step"]["source"],
        },
        {
            "title": "Residual review triage",
            "table": {
                "columns": ["Split", "Observations", "Review required", "Review rate %"],
                "rows": [
                    [a["split"], a["total_observations"], a["review_required"],
                     a["review_rate_percent"]]
                    for a in tri["aggregates"]
                ],
            },
            "prose": (
                "The frozen residual threshold is a review heuristic. On the held-out "
                "test split it produced zero true positives against 20,216 Bot-labelled "
                "sequences, so it must not be presented as an attack detector."
            ),
            "source": tri.get("source"),
        },
        {
            "title": "Detection assessment",
            "prose": dash["detection_assessment"]["verdict"],
            "source": dash["detection_assessment"]["source"],
        },
        {
            "title": "Limitations",
            "prose": "\n".join(f"- {item}" for item in dash["limitations"]),
        },
    ]
    md = build_report_markdown(
        f"SentinelX Security Analytics Summary — {_dt.date.today().isoformat()}", sections
    )
    return {
        "title": "Security Analytics Summary",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "markdown": md,
        "sections": sections,
        "provenance": DERIVED,
    }
