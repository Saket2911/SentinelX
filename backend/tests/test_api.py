from __future__ import annotations

import io

from fastapi.testclient import TestClient

from .conftest import RUN_ID, THRESHOLD


def test_health_reports_checks(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert {check["id"] for check in body["checks"]} >= {"models_dir", "runs", "state", "torch"}
    assert body["runtime"]["auth_enabled"] is False


def test_capabilities_are_honest(client: TestClient) -> None:
    body = client.get("/api/capabilities").json()
    states = {capability["id"]: capability["state"] for capability in body["capabilities"]}
    assert states["live_telemetry"] == "planned"
    assert states["analyst_workflow"] == "implemented"
    assert any("not implemented" in limitation.lower() for limitation in body["limitations"])


def test_dashboard_composes_fixture_artifacts(client: TestClient) -> None:
    body = client.get("/api/dashboard").json()
    assert body["availability"]["available"] is True
    assert body["status"]["posture"] == "review_required"
    assert body["status"]["threshold"] == THRESHOLD
    metrics = {metric["id"]: metric for metric in body["metrics"]}
    assert metrics["observations"]["value"] == 30
    assert metrics["review_candidates"]["value"] == 1
    assert metrics["benign"]["value"] == 20
    assert metrics["attack_labels"]["value"] == 10
    assert body["label_distribution"]
    assert body["recent_events"]
    assert body["model_health"]["run_id"] == RUN_ID


def test_empty_deployment_reports_unavailable_not_zero(empty_client: TestClient) -> None:
    response = empty_client.get("/api/dashboard")
    assert response.status_code == 503
    assert response.json()["error"] == "artifact_unavailable"


def test_prediction_summary_and_multi_step(client: TestClient) -> None:
    body = client.get("/api/predictions/summary").json()
    assert body["availability"]["available"] is True
    metrics = {metric["id"]: metric for metric in body["metrics"]}
    assert metrics["mae"]["value"] == 0.32
    assert metrics["improvement"]["status"] == "good"
    assert len(body["multi_step"]) == 4
    horizon_one = [row for row in body["multi_step"] if row["horizon_seconds"] == 1]
    assert {row["label"] for row in horizon_one} == {"ALL", "Benign", "Bot"}
    assert body["error_distribution"]
    assert body["calibration"]["threshold"] == THRESHOLD


def test_prediction_series_returns_actual_and_predicted(client: TestClient) -> None:
    body = client.get(
        "/api/predictions/series", params={"feature": "Flow Count", "limit": 5}
    ).json()
    assert body["availability"]["available"] is True
    assert body["feature"] == "Flow Count"
    assert "Flow Count" in body["features"]
    assert len(body["points"]) == 5
    first = body["points"][0]
    assert first["actual"] is not None and first["predicted"] is not None
    assert first["priority"] in {"baseline", "within_range", "elevated", "high", "review_required"}
    assert body["summary"]["total_points"] == 30


def test_unknown_feature_is_reported_not_invented(client: TestClient) -> None:
    body = client.get(
        "/api/predictions/series", params={"feature": "not_a_feature"}
    ).json()
    assert body["feature"] is None
    assert "feature_note" in body["summary"]


def test_events_filter_update_and_audit_trail(client: TestClient) -> None:
    listing = client.get("/api/events").json()
    assert listing["availability"]["available"] is True
    assert listing["total"] == 1
    event = listing["events"][0]
    assert event["event_id"] == "step-8"
    assert event["priority"] == "review_required"
    assert event["dataset_label"] == "Benign"
    assert event["contributors"][0]["feature"] == "Flow IAT Mean__mean"
    assert event["recommended_actions"]
    assert listing["filters"]["selection"] == "threshold"

    updated = client.patch(
        f"/api/events/{event['event_id']}",
        json={"status": "investigating", "assignee": "soc-tier-1", "notes": "checked flows"},
    )
    assert updated.status_code == 200
    body = updated.json()
    assert body["event"]["status"] == "investigating"
    assert body["event"]["assignee"] == "soc-tier-1"
    assert any(entry["action"] == "status" for entry in body["history"])

    filtered = client.get("/api/events", params={"status": "investigating"}).json()
    assert filtered["total"] == 1
    assert filtered["status_counts"] == {"investigating": 1}

    missing = client.get("/api/events/step-9999")
    assert missing.status_code == 404


def test_top_residuals_are_labelled_as_review_context(client: TestClient) -> None:
    body = client.get("/api/predictions/top-residuals", params={"limit": 3}).json()
    assert len(body["residuals"]) == 3
    assert body["residuals"][0]["step"] == 8
    assert "not incidents" in body["note"].lower()


def test_timeline_phases_and_buckets(client: TestClient) -> None:
    body = client.get("/api/timeline").json()
    assert body["availability"]["available"] is True
    assert body["buckets"]
    assert body["phases"]
    assert {phase["label"] for phase in body["phases"]} == {"Benign", "Bot"}
    assert any("not reconstructed attack stages" in limitation for limitation in body["limitations"])


def test_explainability_global_and_temporal(client: TestClient) -> None:
    body = client.get("/api/explainability", params={"label": "Bot", "output": "Flow Count"}).json()
    assert body["availability"]["available"] is True
    assert body["method"]["local_available"] is False
    assert len(body["global_importance"]) == 5
    assert body["global_importance"][0]["feature"] == "feature_1"
    assert len(body["temporal_importance"]) == 10
    assert body["temporal_importance"][-1]["lag_seconds"] == 0
    assert body["local"] is None
    assert any("not present" in limitation for limitation in body["limitations"])


def test_model_registry_and_inference_guardrails(client: TestClient) -> None:
    listing = client.get("/api/models").json()
    assert listing["active_run_id"] == RUN_ID
    assert len(listing["models"]) == 1
    model = listing["models"][0]
    assert model["architecture"]["input_features"] == 209
    assert model["checkpoint"]["present"] is False
    assert model["test_metrics"]["lstm"]["mae"] == 0.32

    detail = client.get(f"/api/models/{RUN_ID}").json()
    assert detail["health"]
    assert detail["multi_step"]
    assert detail["evaluation"]["calibration"]["threshold"] == THRESHOLD

    status = client.get(f"/api/models/{RUN_ID}/inference-status").json()
    assert status["available"] is False
    assert status["reasons"]

    blocked = client.post(
        f"/api/models/{RUN_ID}/activate", json={"run_id": RUN_ID, "confirm": True}
    )
    assert blocked.status_code == 409
    assert "checkpoint" in blocked.json()["detail"]["detail"]

    unconfirmed = client.post(
        f"/api/models/{RUN_ID}/activate", json={"run_id": RUN_ID, "confirm": False}
    )
    assert unconfirmed.status_code == 400

    prediction = client.post(
        f"/api/models/{RUN_ID}/predict",
        json={"sequence": [[0.0] * 209 for _ in range(10)]},
    )
    assert prediction.status_code == 503


def test_prediction_rejects_wrong_shape(client: TestClient) -> None:
    response = client.post(
        f"/api/models/{RUN_ID}/predict", json={"sequence": [[0.0] * 10 for _ in range(10)]}
    )
    assert response.status_code == 422
    assert "209" in response.json()["detail"]["detail"]


def test_dataset_upload_and_validation(client: TestClient) -> None:
    listing = client.get("/api/datasets").json()
    assert listing["model_features"] is None
    assert listing["artifacts"]

    csv = "Timestamp,Label,Fwd Pkts,Bwd Pkts\n2018-03-02 10:00:00,Benign,1,2\n"
    upload = client.post(
        "/api/datasets/upload",
        files={"file": ("sample.csv", io.BytesIO(csv.encode()), "text/csv")},
    )
    assert upload.status_code == 200
    body = upload.json()
    assert body["dataset_id"].startswith("upload:")
    assert body["validation"]["valid"] is False  # 209-feature contract unavailable/not met
    assert any(check["id"] == "feature_contract" for check in body["validation"]["checks"])

    validated = client.get("/api/datasets/validate", params={"dataset_id": body["dataset_id"]})
    assert validated.status_code == 200
    assert validated.json()["row_count"] == 1

    too_large = client.post(
        "/api/datasets/upload",
        files={"file": ("big.csv", io.BytesIO(b"a" * (3 * 1024 * 1024)), "text/csv")},
    )
    assert too_large.status_code == 413

    traversal = client.get("/api/datasets/validate", params={"dataset_id": "upload:../../app/main.py"})
    assert traversal.status_code == 404


def test_reports_are_generated_from_service_data(client: TestClient) -> None:
    created = client.post(
        "/api/reports",
        json={
            "title": "Nightly SOC Review",
            "sections": ["summary", "model_evaluation", "triage", "explainability"],
            "author": "soc-tier-1",
        },
    )
    assert created.status_code == 200
    metadata = created.json()
    assert metadata["run_id"] == RUN_ID

    listing = client.get("/api/reports").json()
    assert [report["report_id"] for report in listing["reports"]] == [metadata["report_id"]]

    content = client.get(f"/api/reports/{metadata['report_id']}").json()
    assert "# Nightly SOC Review" in content["markdown"]
    assert "Model evaluation" in content["markdown"]
    assert "not confirmed incidents" in content["markdown"]

    download = client.get(f"/api/reports/{metadata['report_id']}/download")
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("text/markdown")

    invalid = client.post("/api/reports", json={"title": "x", "sections": ["unknown"]})
    assert invalid.status_code == 422


def test_settings_round_trip_and_calibration_provenance(client: TestClient) -> None:
    body = client.get("/api/settings").json()
    assert body["threshold_source"]["calibrated_threshold"] == THRESHOLD
    assert body["auth_enabled"] is False

    updated = client.put(
        "/api/settings",
        json={"review_threshold": 1.5, "page_size": 50, "analyst_name": "tier-2"},
    )
    assert updated.status_code == 200
    assert updated.json()["settings"]["review_threshold"] == 1.5
    assert updated.json()["threshold_source"]["analyst_override_in_use"] is True
    assert updated.json()["threshold_source"]["calibrated_threshold"] == THRESHOLD

    invalid = client.put("/api/settings", json={"page_size": 5000})
    assert invalid.status_code == 422


def test_system_status_lists_artifacts(client: TestClient) -> None:
    body = client.get("/api/system/status").json()
    assert body["active_run_id"] == RUN_ID
    keys = {artifact["key"] for artifact in body["artifacts"]}
    assert {"config", "test_evaluation", "replay"} <= keys
    replay = next(artifact for artifact in body["artifacts"] if artifact["key"] == "explainable_triage")
    assert replay["present"] is True
    assert body["runtime"]["artifacts_present"] > 0
