"""API and service tests for the SentinelX backend.

These pin the contract that matters for credibility:
  * recorded numbers are the real recorded numbers,
  * provenance is always declared,
  * absent artifacts are reported as absent, never invented,
  * mutations validate input.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.artifacts import (
    parse_label_coverage,
    parse_multistep_report,
    parse_run_structure,
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


# ------------------------------------------------------------------ health --

def test_health_declares_data_mode_and_gaps(client: TestClient):
    body = client.get("/api/health").json()
    assert body["service"] == "sentinelx-api"
    assert body["data_mode"] == "offline_dataset"
    assert body["live_telemetry_available"] is False
    # No checkpoint is committed, so inference must be reported unavailable.
    assert body["inference_available"] is False
    assert any("checkpoint is loadable" in w for w in body["warnings"])


def test_artifact_registry_marks_binaries_absent(client: TestClient):
    body = client.get("/api/health").json()
    binaries = [a for a in body["artifacts"] if a["provenance_kind"] == "binary"]
    assert binaries, "expected binary artifacts to be registered"
    assert all(not a["present"] for a in binaries)


# --------------------------------------------------------------- dashboard --

def test_dashboard_matches_recorded_test_metrics(client: TestClient):
    body = client.get("/api/dashboard").json()
    by_key = {m["key"]: m for m in body["metrics"]}
    assert by_key["test_mae"]["value"] == pytest.approx(0.3206269443035126, rel=1e-9)
    assert by_key["test_rmse"]["value"] == pytest.approx(0.531328022480011, rel=1e-9)
    assert by_key["test_samples"]["value"] == 31406
    assert by_key["input_features"]["value"] == 209
    assert all(m["provenance"] == "recorded_artifact" for m in body["metrics"])


def test_dashboard_does_not_claim_a_validated_detector(client: TestClient):
    det = client.get("/api/dashboard").json()["detection_assessment"]
    assert det["validated_detector"] is False
    assert det["test_metrics"]["TP"] == 0
    assert det["test_metrics"]["FN"] == 20216
    assert det["test_metrics"]["recall"] == 0.0


def test_dashboard_states_limitations(client: TestClient):
    body = client.get("/api/dashboard").json()
    assert len(body["limitations"]) >= 4
    joined = " ".join(body["limitations"]).lower()
    assert "not evidence of an attack" in joined
    assert "no live network telemetry" in joined


# ----------------------------------------------------------------- network --

def test_partitions_totals_are_recorded(client: TestClient):
    body = client.get("/api/network/overview").json()
    assert body["available"] is True
    assert body["partitions"]["train"]["total"] == 197389
    assert body["partitions"]["validation"]["total"] == 50870
    assert body["partitions"]["test"]["total"] == 31406


def test_flow_level_detail_is_declared_unavailable(client: TestClient):
    body = client.get("/api/network/overview").json()
    assert body["flow_level_detail"]["provenance"] == "unavailable"
    assert "Datasets/*.csv" in body["flow_level_detail"]["requires"]


# ------------------------------------------------------------- predictions --

def test_multi_step_has_five_horizons_and_four_labels(client: TestClient):
    body = client.get("/api/predictions").json()["multi_step"]
    assert body["horizons"] == [1, 2, 5, 10, 30]
    assert len(body["rows"]) == 20
    for row in body["rows"]:
        assert row["lstm_mae"] > 0
        assert row["persistence_mae"] > 0


def test_multi_step_first_horizon_values_match_report(client: TestClient):
    rows = client.get("/api/predictions").json()["multi_step"]["rows"]
    all_row = next(r for r in rows if r["horizon_seconds"] == 1 and r["target_label"] == "ALL")
    assert all_row["valid_origins"] == 31377
    assert all_row["lstm_mae"] == pytest.approx(0.320655, rel=1e-6)
    assert all_row["persistence_mae"] == pytest.approx(0.356244, rel=1e-6)


def test_multi_step_filter(client: TestClient):
    body = client.get("/api/predictions/multi-step?horizon=30&label=Benign").json()
    assert body["filtered_count"] == 1
    assert body["rows"][0]["horizon_seconds"] == 30
    assert body["rows"][0]["target_label"] == "Benign"


def test_improvement_at_long_horizon_is_positive(client: TestClient):
    rows = client.get("/api/predictions").json()["multi_step"]["rows"]
    h30 = next(r for r in rows if r["horizon_seconds"] == 30 and r["target_label"] == "ALL")
    assert h30["mae_improvement_percent"] > 0


def test_live_forecast_is_declared_unavailable(client: TestClient):
    live = client.get("/api/predictions").json()["live_forecast"]
    assert live["provenance"] == "unavailable"
    assert any("best_model.pt" in r for r in live["requires"])


# ------------------------------------------------------------------ triage --

def test_triage_uses_frozen_threshold(client: TestClient):
    body = client.get("/api/triage").json()
    assert body["calibration"]["threshold"] == pytest.approx(1.881396770477295, rel=1e-12)
    assert body["calibration"]["recalibrated_on_test"] is False
    assert body["decision_policy"]["ground_truth_labels_used"] is False
    splits = {a["split"]: a for a in body["aggregates"]}
    assert splits["validation"]["review_required"] == 2332
    assert splits["test"]["review_required"] == 1


def test_triage_reports_event_rows_unavailable(client: TestClient):
    assert client.get("/api/triage").json()["per_event_rows"]["provenance"] == "unavailable"


# ---------------------------------------------------------- explainability --

def test_explainability_has_experiments_and_caveats(client: TestClient):
    body = client.get("/api/explainability").json()
    assert body["experiments"]
    assert "causal" in " ".join(body["caveats"]).lower()
    assert body["shap_values"]["provenance"] == "unavailable"


def test_feature_families_parsed_with_negative_gain_preserved(client: TestClient):
    families = client.get("/api/explainability").json()["feature_families"]
    assert len(families) >= 6
    tcp = [f for f in families if f["feature_family"] == "TCP flags" and f["horizon_seconds"] == 1]
    assert tcp and tcp[0]["mae_improvement_percent"] < 0


# ---------------------------------------------------------------- timeline --

def test_timeline_run_structure(client: TestClient):
    body = client.get("/api/timeline").json()
    assert body["available"] is True
    assert body["total_runs"] == 390
    assert body["total_states"] == 14722
    assert body["window"]["start"] == "2018-03-02 08:54:38"


# ---------------------------------------------------------------- datasets --

def test_datasets_upload_is_disabled(client: TestClient):
    body = client.get("/api/datasets").json()
    assert body["upload"]["supported"] is False
    assert body["feature_version"]["v3_features"] == 209
    assert body["feature_version"]["sequence_length"] == 10


# ------------------------------------------------------------------ models --

def test_models_expose_three_versions_without_superiority_claim(client: TestClient):
    body = client.get("/api/models").json()
    ids = [v["id"] for v in body["versions"]]
    assert ids == ["v1", "v2", "v3"]
    assert "no claim" in body["comparison_policy"].lower()
    v2 = next(v for v in body["versions"] if v["id"] == "v2")
    assert v2["input_features"] == 212
    assert v2["test_evaluation"]["lstm_metrics"]["mae"] == pytest.approx(0.3153945693)


def test_activate_unknown_model_is_404(client: TestClient):
    assert client.post("/api/models/v9/activate").status_code == 404


def test_activate_known_model_persists(client: TestClient):
    res = client.post("/api/models/v2/activate").json()
    assert res["status"] == "activated"
    assert res["inference_available"] is False
    assert res["warning"]
    # restore
    client.post("/api/models/v3/activate")
    assert client.get("/api/models").json()["active_model"] == "v3"


# ----------------------------------------------------------------- reports --

def test_report_catalog_entries_exist(client: TestClient):
    body = client.get("/api/reports/catalog").json()
    assert len(body["artifacts"]) >= 10
    assert all(a["present"] for a in body["artifacts"])


def test_artifact_unknown_id_is_404(client: TestClient):
    assert client.get("/api/reports/artifact/nope").status_code == 404


def test_generate_and_download_report(client: TestClient):
    gen = client.post("/api/reports/generate").json()
    assert "SentinelX Security Analytics Summary" in gen["markdown"]
    assert "not evidence of an attack" in gen["markdown"].lower() or "review heuristic" in gen["markdown"].lower()
    dl = client.get(f"/api/reports/generated/{gen['id']}")
    assert dl.status_code == 200
    assert dl.headers["content-type"].startswith("text/markdown")


def test_download_rejects_path_traversal(client: TestClient):
    assert client.get("/api/reports/generated/..%2F..%2Fetc%2Fpasswd").status_code in (400, 404)


# ---------------------------------------------------------------- settings --

def test_settings_roundtrip_and_validation(client: TestClient):
    original = client.get("/api/settings").json()["settings"]
    payload = {**original, "table_page_size": 40}
    assert client.put("/api/settings", json=payload).json()["settings"]["table_page_size"] == 40

    bad = {**original, "anomaly_review_threshold": 0}
    assert client.put("/api/settings", json=bad).status_code == 422

    client.put("/api/settings", json=original)


# ----------------------------------------------------------------- parsers --

def test_parse_label_coverage_shape():
    text = "TRAIN\n---\nBenign: 10\nBot: 5\n\nTEST\n---\nBenign: 2\n"
    parsed = parse_label_coverage(text)
    assert parsed["train"] == {"Benign": 10, "Bot": 5}
    assert parsed["test"] == {"Benign": 2}


def test_parse_multistep_ignores_non_rows():
    assert parse_multistep_report("no table here") == []


def test_parse_run_structure_handles_missing_report():
    out = parse_run_structure("")
    assert out["total_runs"] is None and out["runs"] == []
