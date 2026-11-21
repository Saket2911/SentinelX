from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.services.anomaly import Calibration
from app.services.datasets import analyse_file
from app.services.replay import resolve_columns
from app.services.research_text import (
    parse_feature_analysis_report,
    parse_label_coverage,
    parse_multi_step_report,
    parse_training_report,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_parse_label_coverage_from_committed_report() -> None:
    path = REPO_ROOT / "reports" / "mitre_attack_contextual_mapping_v1" / "dataset_label_coverage_report.txt"
    if not path.is_file():
        pytest.skip("committed label coverage report not present")
    coverage = parse_label_coverage(path.read_text(encoding="utf-8"))
    assert coverage["train"]["Benign"] == 163960
    assert coverage["validation"]["Infilteration"] == 12644
    assert coverage["test"]["Bot"] == 20216
    assert coverage["test"]["NO_FLOWS"] == 1


def test_parse_multi_step_report_from_committed_artifact() -> None:
    path = (
        REPO_ROOT
        / "Models"
        / "lstm_world_model_v3_gpu_run2"
        / "multi_step_forecasting_v3_fixed"
        / "multi_step_forecasting_report.txt"
    )
    if not path.is_file():
        pytest.skip("committed multi-step report not present")
    rows = parse_multi_step_report(path.read_text(encoding="utf-8"))
    assert rows
    first = rows[0]
    assert first["horizon_seconds"] == 1
    assert first["label"] == "ALL"
    assert first["valid_origins"] == 31377
    assert first["lstm_mae"] == pytest.approx(0.320655)
    assert {row["horizon_seconds"] for row in rows} >= {1, 2, 5, 10, 30}


def test_parse_feature_analysis_report() -> None:
    text = (
        "AGGREGATE VERIFICATION\n"
        "All checks passed: True\n\n"
        "HORIZON: 1 seconds\n"
        "Features improved over persistence: 165/209\n"
        "Features worse than persistence: 41/209\n"
        "Features equal to persistence: 3/209\n\n"
        "TOP 5 FEATURES BY MAE IMPROVEMENT:\n"
        "  Fwd IAT Min__mean: 31.677%\n"
        "  Flow IAT Min__mean: 31.655%\n"
    )
    parsed = parse_feature_analysis_report(text)
    assert parsed["verification_passed"] is True
    horizon = parsed["horizons"][0]
    assert horizon["improved"] == 165
    assert horizon["total"] == 209
    assert len(horizon["top_improvements"]) == 2
    assert horizon["top_improvements"][0]["feature"] == "Fwd IAT Min__mean"


def test_parse_training_report() -> None:
    text = "Input features: 209\nBest epoch: 11\nGPU confidence: 0.5\n"
    parsed = parse_training_report(text)
    assert parsed["input_features"] == 209
    assert parsed["best_epoch"] == 11
    assert parsed["gpu_confidence"] == 0.5


def test_priority_bands_follow_recorded_quantiles() -> None:
    calibration = Calibration(
        threshold=1.88,
        threshold_source="validation benign p99",
        quantiles={"0.5": 0.29, "0.9": 0.47, "0.95": 0.53, "0.99": 0.70},
        sample_count=100,
        method="mean absolute residual",
    )
    assert calibration.priority_for(None) == "unknown"
    assert calibration.priority_for(0.1) == "baseline"
    assert calibration.priority_for(0.49) == "within_range"
    assert calibration.priority_for(0.55) == "elevated"
    assert calibration.priority_for(0.75) == "high"
    assert calibration.priority_for(1.9) == "review_required"
    assert "not a validated intrusion-detection threshold" in calibration.as_dict()["priority_semantics"]


def test_replay_column_resolution_is_defensive() -> None:
    frame = pd.DataFrame(
        {
            "Step": [1, 2],
            "Target Timestamp": ["2018-03-02 10:00:00", "2018-03-02 10:00:01"],
            "Dataset Label": ["Benign", "Bot"],
            "Segment": ["1", "1"],
            "actual flow count": [1.0, 2.0],
            "predicted flow count": [0.5, 2.5],
        }
    )
    columns = resolve_columns(frame)
    assert columns.step == "Step"
    assert columns.label == "Dataset Label"
    assert columns.segment == "Segment"
    assert columns.pairs == {
        "flow count": ("actual flow count", "predicted flow count")
    }

    bare = resolve_columns(pd.DataFrame({"a": [1], "b": [2]}))
    assert bare.step is None and bare.label is None and not bare.pairs


def test_dataset_analysis_flags_missing_contract(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("foo,bar\n1,2\n", encoding="utf-8")
    result = analyse_file(path, model_features=None)
    assert result["valid"] is False
    checks = {check["id"]: check for check in result["checks"]}
    assert checks["required_columns"]["status"] == "failed"
    assert checks["feature_contract"]["status"] == "warning"

    good = tmp_path / "good.csv"
    pd.DataFrame(
        {
            "Timestamp": ["2018-03-02 10:00:00", "2018-03-02 10:00:01"],
            "Label": ["Benign", "Bot"],
            "Fwd Pkts": [1, 2],
            "Bwd Pkts": [2, 3],
            "Duration": [1.0, 2.0],
            "Bytes": [10, 20],
        }
    ).to_csv(good, index=False)
    with_contract = analyse_file(good, model_features=["Fwd Pkts", "Bwd Pkts", "Duration", "Bytes"])
    assert with_contract["valid"] is True
    assert with_contract["missing_features"] == 0
    assert with_contract["row_count"] == 2
