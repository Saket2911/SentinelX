"""Per-state playback contracts."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_meta_describes_replay_timeline(client: TestClient) -> None:
    response = client.get("/api/replay/meta")
    assert response.status_code == 200
    payload = response.json()

    assert payload["availability"]["available"] is True
    assert payload["total_states"] == 30
    assert payload["feature_count"] == 209
    assert payload["sequence_length"] == 10
    assert payload["step_min"] == 1 and payload["step_max"] == 30
    assert payload["threshold"] == 1.88139677
    assert len(payload["segments"]) == 2
    assert payload["segments"][0]["states"] == 15
    assert payload["manifest"]["checkpoint_epoch"] == 11
    assert "not live monitoring" in " ".join(payload["limitations"])

    features = payload["features"]
    assert features[0] == "Flow Count"
    assert "ACK Flag Cnt__mean" in features
    assert len(features) == 209


def test_state_returns_observed_and_predicted_values(client: TestClient) -> None:
    response = client.get("/api/replay/state", params={"step": 1})
    assert response.status_code == 200
    payload = response.json()

    state = payload["state"]
    assert state["step"] == 1
    assert state["index"] == 0
    assert state["total_states"] == 30
    assert state["timestamp"].startswith("2018-03-02")
    assert state["capture_day"] == "2018-03-02"
    assert state["segment_id"] == "1"
    assert state["dataset_label"] == "Benign"
    assert state["sequence_length"] == 10
    assert state["feature_count"] == 209
    assert state["threshold"] == 1.88139677

    flow = next(entry for entry in payload["features"] if entry["feature"] == "Flow Count")
    assert flow["actual"] == 10.0
    assert flow["predicted"] == 9.5
    assert flow["residual"] == 0.5
    assert flow["absolute_error"] == 0.5
    assert len(flow["history"]) == 10
    assert all(value == 10.0 for value in flow["history"])

    assert payload["contributors"][0]["feature"] == "Flow IAT Mean__mean"
    assert payload["contributors"][0]["rank"] == 1
    assert len(payload["available_features"]) == 209


def test_state_accepts_selected_features(client: TestClient) -> None:
    response = client.get(
        "/api/replay/state",
        params={"step": 12, "features": "feature_1, ACK Flag Cnt__mean, nope"},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["state"]["segment_id"] == "1"
    returned = [entry["feature"] for entry in payload["features"]]
    assert returned == ["feature_1", "ACK Flag Cnt__mean"]
    ack = payload["features"][1]
    assert len(ack["history"]) == 10
    assert ack["actual"] != ack["predicted"]


def test_state_rejects_unknown_step(client: TestClient) -> None:
    response = client.get("/api/replay/state", params={"step": 999})
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_series_returns_feature_history_and_stats(client: TestClient) -> None:
    response = client.get(
        "/api/replay/series", params={"feature": "Flow Count", "start": 0, "limit": 30}
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["feature"] == "Flow Count"
    assert len(payload["points"]) == 30
    first = payload["points"][0]
    assert first["step"] == 1
    assert first["actual"] == 10.0
    assert first["predicted"] == 9.5
    assert first["residual"] == 0.5

    assert payload["stats"]["samples"] == 30
    assert payload["stats"]["mae"] == 0.5
    assert payload["stats"]["max_absolute_error"] == 0.5


def test_series_includes_state_mean_error_and_top_errors(client: TestClient) -> None:
    payload = client.get(
        "/api/replay/series", params={"feature": "Flow Count", "limit": 3}
    ).json()
    # Flow Count is off by 0.5 in the fixture; other features contribute to the mean.
    assert payload["points"][0]["state_mae"] > 0
    assert payload["state_mae"] > 0
    assert len(payload["top_errors"]) == 10
    assert payload["top_errors"][0]["absolute_error"] == 0.5
    assert payload["top_errors"][0]["step"] == 1


def test_state_can_include_the_full_feature_table(client: TestClient) -> None:
    payload = client.get(
        "/api/replay/state", params={"step": 3, "include_all": "true"}
    ).json()
    table = payload["feature_table"]
    assert len(table) == 209
    assert table[0]["absolute_error"] >= table[-1]["absolute_error"]
    assert table[0]["feature"] in payload["available_features"]


def test_contributors_fall_back_to_replay_columns(client: TestClient) -> None:
    payload = client.get("/api/replay/contributors", params={"limit": 5}).json()
    assert payload["total_states"] == 30
    assert payload["source"] == "computed from replay contributor columns"
    assert payload["contributors"][0]["feature"] == "Flow IAT Mean__mean"
    assert payload["contributors"][0]["appearance_percent"] == 100.0


def test_series_window_is_bounded(client: TestClient) -> None:
    response = client.get(
        "/api/replay/series", params={"feature": "Flow Count", "start": 10, "limit": 5}
    )
    payload = response.json()
    assert [point["step"] for point in payload["points"]] == [11, 12, 13, 14, 15]
    assert payload["window"] == {"start": 10, "end": 15, "total_states": 30}


def test_series_falls_back_to_first_feature(client: TestClient) -> None:
    response = client.get("/api/replay/series", params={"feature": "does-not-exist", "limit": 2})
    assert response.status_code == 200
    assert response.json()["feature"] == "Flow Count"


def test_playback_reports_unavailable_without_artifacts(empty_client: TestClient) -> None:
    response = empty_client.get("/api/replay/meta")
    assert response.status_code == 503
    payload = response.json()
    assert payload["error"] == "artifact_unavailable"
    assert payload["detail"]
