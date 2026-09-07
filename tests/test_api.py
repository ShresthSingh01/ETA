"""
test_api.py - Integration tests for FastAPI endpoints, replay states, and event injection.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_get_trains(client):
    res = client.get("/api/trains")
    assert res.status_code == 200
    data = res.json()
    assert "trains" in data
    assert len(data["trains"]) == 4
    train_nums = [t["train_number"] for t in data["trains"]]
    assert 12303 in train_nums
    assert 12951 in train_nums


def test_get_replay_state(client):
    res = client.get("/api/replay/state")
    assert res.status_code == 200
    data = res.json()
    assert "current_station" in data
    assert "comparison_table" in data
    assert "summary" in data
    assert len(data["comparison_table"]) > 0


def test_step_replay(client):
    res = client.post("/api/replay/step", json={"step": 2})
    assert res.status_code == 200
    data = res.json()
    assert data["current_step"] == 2


def test_inject_and_clear_event(client):
    # Inject TSR
    res = client.post("/api/events/inject", json={
        "event_type": "SPEED_RESTRICTION",
        "from_station": "HWH",
        "to_station": "BWN",
        "restricted_speed_kmh": 30.0,
        "affected_km": 20.0,
        "halt_duration_minutes": 0.0
    })
    assert res.status_code == 200
    data = res.json()
    assert len(data["active_events"]) >= 1
    assert data["active_events"][0]["restricted_speed_kmh"] == 30.0
    
    # Clear
    res_clear = client.post("/api/events/clear")
    assert res_clear.status_code == 200
    data_clear = res_clear.json()
    assert len(data_clear["active_events"]) == 0


def test_benchmarks_endpoint(client):
    res = client.get("/api/benchmarks")
    assert res.status_code == 200
    data = res.json()
    assert "Test Set - Main LightGBM" in data
    assert data["Test Set - Main LightGBM"]["MAE"] < 7.0


def test_inject_caution_order_and_unscheduled_stop(client):
    # Inject Caution Order
    res = client.post("/api/events/inject", json={
        "event_type": "CAUTION_ORDER",
        "from_station": "HWH",
        "to_station": "BWN",
        "restricted_speed_kmh": 40.0,
        "affected_km": 10.0,
        "halt_duration_minutes": 0.0,
        "source_type": "CAUTION_ORDER"
    })
    assert res.status_code == 200
    data = res.json()
    assert len(data["active_events"]) >= 1
    assert data["active_events"][0]["event_type"] == "CAUTION_ORDER"

    # Inject Unscheduled Stop
    res_stop = client.post("/api/events/inject", json={
        "event_type": "UNSCHEDULED_STOP",
        "from_station": "BWN",
        "to_station": "ASN",
        "halt_duration_minutes": 15.0,
        "source_type": "MANUAL_ENTRY"
    })
    assert res_stop.status_code == 200
    data_stop = res_stop.json()
    assert any(e["event_type"] == "UNSCHEDULED_STOP" for e in data_stop["active_events"])

    # Clear
    client.post("/api/events/clear")


def test_stratified_benchmarks_endpoints(client):
    res_h = client.get("/api/benchmarks/horizon")
    assert res_h.status_code == 200
    assert "1 hop" in res_h.json()

    res_s = client.get("/api/benchmarks/scenarios")
    assert res_s.status_code == 200
    assert "On-time (<5m)" in res_s.json()

    res_r = client.get("/api/benchmarks/rules")
    assert res_r.status_code == 200
    assert "mps_floor_clamps" in res_r.json()


def test_get_provider_status(client):
    res = client.get("/api/live/provider/status")
    assert res.status_code == 200
    data = res.json()
    assert "active_mode" in data
    assert "provider_name" in data
    assert "is_fallback_active" in data
    assert "telemetry_metrics" in data
