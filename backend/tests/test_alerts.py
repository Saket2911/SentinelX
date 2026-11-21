"""Prediction scoring, alert rules, persistence and playback alert APIs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.services import alerts as alerts_service
from app.services.playback import reset_cache

RUN = "test_run_v3"


def _rewrite_predictions(workspace, deviations: dict[int, float]) -> None:
    """Add a constant deviation to selected target states in the test fixture.

    These are controlled synthetic fixtures used only to exercise the alert
    logic; they are never presented as real dataset results.
    """

    path = workspace.models_dir / RUN / "historical_replay_v1" / "replay_arrays.npz"
    with np.load(path) as archive:
        predictions = np.array(archive["predictions"], dtype=np.float64)
        actual = np.array(archive["actual"], dtype=np.float64)
        feature_names = np.array(archive["feature_names"])
    for index, deviation in deviations.items():
        predictions[index, :] = actual[index, :] + deviation
    np.savez(path, predictions=predictions, actual=actual, feature_names=feature_names)
    reset_cache()


def _evaluate(client: TestClient, step: int):
    return client.post("/api/playback/evaluate", json={"step": step})


def test_score_state_uses_mean_absolute_residual_in_feature_order() -> None:
    names = ["flow", "packets", "duration"]
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([0.5, 2.5, 1.0])
    score = alerts_service.score_state(actual, predicted, names, limit=3)

    assert score["method"] == "mean_absolute_residual"
    assert score["anomaly_score"] == ((0.5 + 0.5 + 2.0) / 3)
    assert score["mse"] == ((0.25 + 0.25 + 4.0) / 3)
    assert score["rmse"] == ((0.25 + 0.25 + 4.0) / 3) ** 0.5
    assert score["max_absolute_error"] == 2.0
    assert [item["feature"] for item in score["feature_residuals"]] == [
        "duration",
        "flow",
        "packets",
    ]
    assert score["feature_residuals"][0]["signed_residual"] == 2.0


def test_score_state_rejects_shape_mismatch_and_non_finite() -> None:
    try:
        alerts_service.score_state(np.zeros(2), np.zeros(3), ["a", "b", "c"])
    except ValueError as exc:
        assert "shape" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("shape mismatch should be rejected")
    try:
        alerts_service.score_state(np.array([np.nan]), np.zeros(1), ["a"])
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("non-finite values should be rejected")


def test_evaluate_conditions_threshold_and_persistence_rules() -> None:
    config = {
        "persistence_window": 3,
        "elevated_ratio": 0.8,
        "high_ratio": 1.5,
        "critical_ratio": 2.0,
    }
    quiet = alerts_service.evaluate_conditions([0.2, 0.3, 0.4], ["1", "1", "1"], 2.0, config)
    assert quiet["triggered"] is False
    assert quiet["severity"] is None

    single = alerts_service.evaluate_conditions([0.2, 0.3, 2.1], ["1", "1", "1"], 2.0, config)
    assert single["single_threshold"] is True
    assert single["persistence"] is False
    assert single["severity"] == "medium"

    high = alerts_service.evaluate_conditions([0.2, 0.3, 3.1], ["1", "1", "1"], 2.0, config)
    assert high["severity"] == "high"

    critical = alerts_service.evaluate_conditions([0.2, 0.3, 4.1], ["1", "1", "1"], 2.0, config)
    assert critical["severity"] == "critical"

    persistent = alerts_service.evaluate_conditions(
        [1.7, 1.8, 1.9], ["1", "1", "1"], 2.0, config
    )
    assert persistent["persistence"] is True
    assert persistent["single_threshold"] is False
    assert persistent["severity"] == "high"
    assert persistent["trigger"] == "sustained_elevated"


def test_persistence_never_spans_a_segment_boundary() -> None:
    config = {
        "persistence_window": 3,
        "elevated_ratio": 0.8,
        "high_ratio": 1.5,
        "critical_ratio": 2.0,
    }
    result = alerts_service.evaluate_conditions(
        [1.7, 1.8, 1.7], ["1", "1", "2"], 2.0, config
    )
    assert result["persistence"] is False
    assert result["same_segment"] is False
    assert result["triggered"] is False


def test_conditions_unavailable_without_threshold() -> None:
    result = alerts_service.evaluate_conditions([9.0], ["1"], None)
    assert result["available"] is False
    assert result["triggered"] is False
    assert result["reason"]


def test_sequence_validation_rejects_insufficient_history() -> None:
    try:
        alerts_service._validate_sequence(np.zeros((9, 209)), 209, 10)
    except alerts_service.InsufficientContextError as exc:
        assert "(9, 209)" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a nine-state window must be rejected")

    non_finite = np.zeros((10, 209))
    non_finite[3, 4] = np.nan
    try:
        alerts_service._validate_sequence(non_finite, 209, 10)
    except alerts_service.InsufficientContextError as exc:
        assert "non-finite" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("non-finite history must be rejected")


def test_evaluate_returns_prediction_alignment_and_context(client: TestClient) -> None:
    payload = _evaluate(client, 2).json()

    assert payload["state"]["step"] == 2
    assert payload["state"]["index"] == 1
    assert payload["state"]["timestamp"].startswith("2018-03-02")
    assert payload["prediction"]["source"] == alerts_service.PREDICTION_SOURCE
    assert payload["prediction"]["sequence_length"] == 10
    assert payload["prediction"]["model_version"].startswith(RUN)
    assert payload["context"]["states_used"] == 10
    assert payload["context"]["target_step"] == 2
    assert payload["context"]["segment_consistent"] is True

    flow = next(
        item for item in payload["score"]["feature_residuals"] if item["feature"] == "Flow Count"
    )
    assert flow["actual"] == 11.0
    assert flow["predicted"] == 10.5
    assert flow["absolute_error"] == 0.5
    assert payload["score"]["feature_count"] == 209
    # Fixture scores stay far below the calibrated threshold: no alert is invented.
    assert payload["conditions"]["triggered"] is False
    assert payload["alert"] is None


def test_alert_is_generated_at_the_target_timestamp(workspace, client: TestClient) -> None:
    _rewrite_predictions(workspace, {7: 4.0})
    payload = _evaluate(client, 8).json()

    assert payload["conditions"]["triggered"] is True
    assert payload["conditions"]["single_threshold"] is True
    assert payload["conditions"]["severity"] == "critical"
    alert = payload["alert"]
    assert alert is not None
    assert alert["target_step"] == 8
    assert alert["timestamp"] == payload["state"]["timestamp"]
    assert alert["anomaly_score"] >= payload["state"]["threshold"]
    assert alert["threshold"] == payload["state"]["threshold"]
    assert alert["severity"] == "critical"
    assert alert["trigger_reason"]
    assert alert["status"] == "new"
    assert alert["model_version"].startswith(RUN)
    assert alert["features"]
    assert alert["predicted_ref"].startswith("replay_arrays")
    assert alert["context_ref"].startswith("sentinelx_test_X_seq10_v3.npy")
    assert payload["deduplicated"] is False


def test_alert_persistence_and_duplicate_prevention(workspace, client: TestClient) -> None:
    _rewrite_predictions(workspace, {7: 4.0})
    first = _evaluate(client, 8).json()
    second = _evaluate(client, 8).json()
    reset = _evaluate(client, 8).json()

    assert first["alert"]["alert_id"] == second["alert"]["alert_id"] == reset["alert"]["alert_id"]
    assert second["deduplicated"] is True
    assert reset["deduplicated"] is True

    listing = client.get("/api/playback/alerts").json()
    assert listing["total"] == 1
    assert listing["alerts"][0]["alert_id"] == first["alert"]["alert_id"]

    detail = client.get(f"/api/playback/alerts/{first['alert']['alert_id']}").json()
    assert detail["alert"]["alert_id"] == first["alert"]["alert_id"]
    assert detail["explainability"]["shap"]["note"]


def test_seeking_and_replaying_do_not_duplicate_alerts(workspace, client: TestClient) -> None:
    _rewrite_predictions(workspace, {7: 4.0})
    _evaluate(client, 1)
    _evaluate(client, 20)
    _evaluate(client, 8)
    _evaluate(client, 3)
    _evaluate(client, 8)

    listing = client.get("/api/playback/alerts").json()
    assert [alert["target_step"] for alert in listing["alerts"]] == [8]
    assert listing["total"] == 1


def test_persistence_alert_from_consecutive_states(workspace, client: TestClient) -> None:
    # 0.85 x threshold on three consecutive states in the same segment.
    _rewrite_predictions(workspace, {4: 1.6, 5: 1.6, 6: 1.6})
    payload = _evaluate(client, 7).json()

    assert payload["conditions"]["persistence"] is True
    assert payload["conditions"]["single_threshold"] is False
    assert payload["alert"] is not None
    assert payload["alert"]["severity"] == "high"
    assert "consecutive elevated states" in payload["alert"]["trigger_reason"]


def test_segment_boundary_blocks_persistence_alert(workspace, client: TestClient) -> None:
    # Positions 13/14 are segment 1, position 15 is segment 2.
    _rewrite_predictions(workspace, {13: 1.6, 14: 1.6, 15: 1.6})
    payload = _evaluate(client, 16).json()

    assert payload["conditions"]["same_segment"] is False
    assert payload["conditions"]["persistence"] is False
    assert payload["conditions"]["single_threshold"] is False
    assert payload["alert"] is None


def test_alert_status_update_is_audited(workspace, client: TestClient) -> None:
    _rewrite_predictions(workspace, {7: 4.0})
    alert_id = _evaluate(client, 8).json()["alert"]["alert_id"]

    response = client.patch(
        f"/api/playback/alerts/{alert_id}",
        json={"status": "investigating", "notes": "Correlating flows", "assignee": "tier-1"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["alert"]["status"] == "investigating"
    assert payload["alert"]["notes"] == "Correlating flows"
    assert payload["history"][0]["action"] == "status"
    assert payload["history"][0]["to_value"] == "investigating"

    reopened = client.get(f"/api/playback/alerts/{alert_id}").json()
    assert reopened["alert"]["status"] == "investigating"


def test_anomaly_series_reports_threshold_and_flagged_states(
    workspace, client: TestClient
) -> None:
    _rewrite_predictions(workspace, {7: 4.0})
    _evaluate(client, 8)
    payload = client.get("/api/playback/anomalies").json()

    assert payload["threshold"] == 1.88139677
    assert payload["flagged_steps"] == [8]
    assert payload["alert_steps"] == [8]
    assert payload["points"][7]["flagged"] is True
    assert payload["points"][7]["alert_id"]
    assert payload["summary"]["states"] == 30
    assert payload["method"] == "mean_absolute_residual"


def test_dashboard_includes_real_alert_metrics(workspace, client: TestClient) -> None:
    baseline = client.get("/api/dashboard").json()
    assert baseline["alerts"]["total"] == 0
    assert baseline["alerts"]["unresolved"] == 0

    _rewrite_predictions(workspace, {7: 4.0})
    _evaluate(client, 8)
    dashboard = client.get("/api/dashboard").json()
    assert dashboard["alerts"]["total"] == 1
    assert dashboard["alerts"]["high_priority"] == 1
    assert dashboard["alerts"]["unresolved"] == 1
    assert dashboard["alerts"]["recent"][0]["target_step"] == 8
    metrics = {metric["id"]: metric for metric in dashboard["metrics"]}
    assert metrics["playback_alerts"]["value"] == 1
    assert metrics["high_priority_alerts"]["value"] == 1
    assert metrics["unresolved_alerts"]["value"] == 1


def test_alert_filters_by_step_time_and_dataset(workspace, client: TestClient) -> None:
    _rewrite_predictions(workspace, {7: 4.0})
    alert = _evaluate(client, 8).json()["alert"]

    assert client.get(
        "/api/playback/alerts", params={"target_step": 8}
    ).json()["total"] == 1
    assert client.get(
        "/api/playback/alerts", params={"target_step": 9}
    ).json()["total"] == 0
    assert client.get(
        "/api/playback/alerts", params={"dataset_id": alert["dataset_id"]}
    ).json()["total"] == 1
    assert client.get(
        "/api/playback/alerts", params={"since": "2018-03-02T10:00:00"}
    ).json()["total"] == 1
    assert client.get(
        "/api/playback/alerts", params={"until": "2018-03-02T10:00:00"}
    ).json()["total"] == 0


def test_model_version_contract_is_enforced(workspace, client: TestClient) -> None:
    config_path: Path = workspace.models_dir / RUN / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["input_features"] = 212  # V1/V2 contract must never be mixed with V3 arrays
    config_path.write_text(json.dumps(config), encoding="utf-8")
    reset_cache()

    response = _evaluate(client, 2)
    assert response.status_code == 409
    body = response.json()["detail"]
    assert body["error"] == "model_contract_mismatch"
    assert "209" in body["detail"]
