"""Test fixtures.

Artifacts created here are synthetic *test fixtures* used to exercise the API
contracts. They are never served by a real deployment and are never presented as
network telemetry.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

RUN_ID = "test_run_v3"
THRESHOLD = 1.88139677


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@pytest.fixture()
def workspace(tmp_path: Path) -> Settings:
    models = tmp_path / "Models"
    reports = tmp_path / "reports"
    datasets = tmp_path / "Datasets"
    run = models / RUN_ID
    var = tmp_path / "var"
    reports.mkdir(parents=True, exist_ok=True)
    datasets.mkdir(parents=True, exist_ok=True)

    _write_json(
        run / "config.json",
        {
            "model": "SentinelXLSTM V3",
            "sequence_length": 10,
            "input_features": 209,
            "hidden_size": 128,
            "num_layers": 1,
            "dropout": 0.2,
            "batch_size": 128,
            "learning_rate": 0.001,
            "optimizer": "AdamW",
            "loss": "HuberLoss",
            "max_epochs": 30,
            "patience": 6,
            "seed": 42,
            "device": "cuda:0",
            "gpu_name": "Test GPU",
            "training_samples": 1000,
            "validation_samples": 250,
            "best_epoch": 11,
            "best_validation_huber": 0.43,
            "training_seconds": 12.5,
            "test_data_used": False,
        },
    )
    _write_json(
        run / "test_evaluation" / "test_evaluation_summary.json",
        {
            "model": "SentinelXLSTM V3",
            "best_epoch": 11,
            "test_samples": 30,
            "input_features": 209,
            "sequence_length": 10,
            "lstm_metrics": {"huber_loss": 0.12, "mae": 0.32, "rmse": 0.53},
            "persistence_baseline_metrics": {
                "huber_loss": 0.17,
                "mae": 0.35,
                "rmse": 0.67,
            },
            "improvement_over_baseline_percent": {
                "huber_loss": 29.4,
                "mae": 8.6,
                "rmse": 20.9,
            },
            "test_data_used_for_training": False,
            "test_data_used_for_model_selection": False,
        },
    )
    _write_json(
        run / "triage_engine_v1" / "triage_summary.json",
        {
            "engine": "SentinelX Threat Triage Engine V1",
            "threshold": THRESHOLD,
            "test": {
                "split": "test",
                "total_observations": 30,
                "review_required": 1,
                "within_calibrated_range": 29,
                "frozen_threshold": THRESHOLD,
                "threshold_source": "Validation benign 99th percentile",
            },
        },
    )
    _write_json(
        run / "test_anomaly_evaluation" / "test_anomaly_summary.json",
        {
            "checkpoint_epoch": 11,
            "test_samples": 30,
            "threshold": THRESHOLD,
            "score_definition": "Mean absolute residual across 209 features",
            "detection_metrics_benign_vs_bot": {"TP": 1, "FN": 9, "FP": 0, "TN": 20, "precision": 1.0, "recall": 0.1},
        },
    )
    _write_json(
        run / "shap_explainability_v3" / "summary.json",
        {
            "experiment": "Test SHAP",
            "method": "SHAP GradientExplainer",
            "checkpoint_epoch": 11,
            "sequence_length": 10,
            "input_features": 209,
            "background_source": "Benign training sequences",
            "background_size": 64,
            "benign_explanation_samples": 2,
            "bot_explanation_samples": 2,
            "output_features": ["Flow Count", "ACK Flag Cnt__mean"],
            "interpretation": "Test fixture summary.",
        },
    )
    (run / "multi_step_forecasting_v3_fixed").mkdir(parents=True, exist_ok=True)
    (run / "multi_step_forecasting_v3_fixed" / "multi_step_forecasting_report.txt").write_text(
        "SENTINELX V3 — RECURSIVE MULTI-STEP FORECASTING\n"
        "========================================================================\n\n"
        "RESULTS\n"
        "------------------------------------------------------------------------\n"
        " Horizon_Seconds Target_Label  Valid_Origins  LSTM_Huber  Persistence_Huber  LSTM_MAE  Persistence_MAE  MAE_Improvement_Percent  LSTM_RMSE  Persistence_RMSE  RMSE_Improvement_Percent\n"
        "               1          ALL            100        0.120       0.177      0.320        0.356                 9.990      0.531          0.679                 21.749\n"
        "               1       Benign             50        0.125       0.186      0.324        0.365                11.115      0.546          0.703                 22.379\n"
        "               1          Bot             50        0.117       0.171      0.318        0.351                 9.338      0.522          0.664                 21.359\n"
        "               5          ALL             96        0.143       0.207      0.351        0.404                13.038      0.582          0.735                 20.789\n",
        encoding="utf-8",
    )

    rows = []
    labels = ["Benign"] * 20 + ["Bot"] * 10
    for index in range(30):
        mae = THRESHOLD + 0.01 if index == 7 else 0.2 + index * 0.001
        rows.append(
            {
                "Replay Step": index + 1,
                "Step": index + 1,
                "Target Timestamp": f"2018-03-02 10:00:{index:02d}",
                "Target CaptureDay": "2018-03-02",
                "Target Label": labels[index],
                "Segment ID": 1 if index < 15 else 2,
                "Actual Flow Count": 10.0 + index,
                "Predicted Flow Count": 9.5 + index,
                "Prediction MAE": mae,
                "Top Contributor 1": "Flow IAT Mean__mean",
                "Top Contributor 2": "ACK Flag Cnt__mean",
            }
        )
    triage_dir = run / "explainable_triage_v2"
    triage_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(triage_dir / "explainable_triage.csv", index=False)

    # Per-state replay arrays: the ten input states, the observed next state and the
    # model's forecast. Synthetic fixtures; never served by a deployment.
    feature_names = ["Flow Count"] + [f"feature_{i}" for i in range(1, 208)] + ["ACK Flag Cnt__mean"]
    actual = np.zeros((30, len(feature_names)), dtype=np.float64)
    predicted = np.zeros((30, len(feature_names)), dtype=np.float64)
    for index in range(30):
        actual[index, 0] = 10.0 + index
        predicted[index, 0] = 9.5 + index
        actual[index, 1:-1] = np.linspace(0.0, 1.0, 207) + index * 0.01
        predicted[index, 1:-1] = np.linspace(0.0, 0.9, 207) + index * 0.01
        actual[index, -1] = 0.5 + index * 0.001
        predicted[index, -1] = 0.45 + index * 0.001
    sequences = np.repeat(actual[:, None, :], 10, axis=1).astype(np.float32)
    replay_dir = run / "historical_replay_v1"
    replay_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        replay_dir / "replay_arrays.npz",
        predictions=predicted,
        actual=actual,
        feature_names=np.array(feature_names),
    )
    np.save(datasets / "sentinelx_test_sequences_v3.npy", sequences)
    _write_json(
        replay_dir / "replay_manifest.json",
        {
            "input_features": len(feature_names),
            "sequence_length": 10,
            "checkpoint_epoch": 11,
            "replay_observations": 30,
        },
    )

    importance_rows = []
    for label in ("Benign", "Bot"):
        for output in ("Flow Count", "ACK Flag Cnt__mean"):
            for rank in range(1, 6):
                importance_rows.append(
                    {
                        "Target Label": label,
                        "Predicted Output Feature": output,
                        "Input Feature": f"feature_{rank}",
                        "Mean Absolute SHAP": 1.0 / rank,
                        "Mean Signed SHAP": 0.1 * (1 if rank % 2 else -1),
                        "Feature Rank": rank,
                    }
                )
    shap_dir = run / "shap_explainability_v3"
    pd.DataFrame(importance_rows).to_csv(shap_dir / "shap_feature_importance_by_label.csv", index=False)
    pd.DataFrame(
        [
            {"Target Label": label, "Predicted Output Feature": output, "Input Time Position": position, "Mean Absolute SHAP": 0.01 * (position + 1)}
            for label in ("Benign", "Bot")
            for output in ("Flow Count", "ACK Flag Cnt__mean")
            for position in range(10)
        ]
    ).to_csv(shap_dir / "shap_temporal_importance_by_label.csv", index=False)
    pd.DataFrame(
        [
            {
                "Predicted Output Feature": "Flow Count",
                "Input Feature": "feature_1",
                "Benign": 0.1,
                "Bot": 0.3,
                "Bot_minus_Benign_SHAP": 0.2,
                "Absolute_Difference": 0.2,
            }
        ]
    ).to_csv(shap_dir / "shap_benign_bot_comparison.csv", index=False)
    pd.DataFrame(
        [
            {"Label": label, "Output": "Flow Count", "Top10 Overlap": 8, "Jaccard": 0.66, "Spearman": 0.94}
            for label in ("Benign", "Bot")
        ]
    ).to_csv(shap_dir / "stability_comparison_v2_v3.csv", index=False)

    _write_json(
        reports / "v3_reliability_audit_v1" / "reliability_summary.json",
        {
            "samples": 30,
            "overall_error_quantiles": {
                "sample_mae": {"0.5": 0.29, "0.9": 0.47, "0.95": 0.53, "0.99": 0.70, "0.999": 0.95}
            },
        },
    )

    return Settings(
        root=tmp_path,
        models_dir=models,
        reports_dir=reports,
        datasets_dir=datasets,
        var_dir=var,
        default_run_id=RUN_ID,
        api_key=None,
        log_level="WARNING",
        cors_origins=("http://localhost:5173",),
        max_upload_mb=2,
        default_page_size=25,
    )


@pytest.fixture()
def client(workspace: Settings) -> TestClient:
    return TestClient(create_app(workspace))


@pytest.fixture()
def empty_client(tmp_path: Path) -> TestClient:
    (tmp_path / "Models").mkdir()
    settings = Settings(
        root=tmp_path,
        models_dir=tmp_path / "Models",
        reports_dir=tmp_path / "reports",
        datasets_dir=tmp_path / "Datasets",
        var_dir=tmp_path / "var",
        default_run_id=RUN_ID,
        api_key=None,
        log_level="WARNING",
        cors_origins=(),
        max_upload_mb=2,
        default_page_size=25,
    )
    return TestClient(create_app(settings))
