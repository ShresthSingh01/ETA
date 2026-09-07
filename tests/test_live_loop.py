"""
test_live_loop.py - End-to-End Closed-Loop Real-Live Telemetry & Evaluation Tests.

Smart India Hackathon 2026 • Problem Statement 26028 (Ministry of Railways)
Verifies:
1. Live Observation -> Dynamic Network Injection -> Forward ETA -> Next Observation -> Self-Evaluation -> New ETA
2. Zero synthetic data policy enforcement
3. Real-time REST demonstration endpoint /api/demo/live-loop
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.replay.simulator import ReplaySimulator
from src.engine.prediction_logger import LivePredictionLogger


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_closed_loop_step_by_step():
    """Verifies that advancing steps updates network state, evaluates prior predictions, and computes rolling metrics."""
    sim = ReplaySimulator()
    sim.load_journey(12303, '2024-09-28')
    logger = LivePredictionLogger()
    
    # Step 0: Origin Departure
    st0 = sim.get_state(step=0)
    origin_code = st0['current_station']['station_code']
    assert origin_code in sim.calculator.network_engine.live_station_delay
    assert 'closed_loop_trace' in st0
    assert st0['closed_loop_trace']['latest_observation']['is_synthetic'] is False
    
    # Log initial forward predictions
    for hop in st0['comparison_table']:
        logger.log_prediction(
            train_id="12303",
            station_code=hop["station_code"],
            station_name=hop["station_name"],
            predicted_arrival=hop["our_predicted_eta"],
            predicted_delay_mins=hop["our_predicted_delay"],
            confidence_pct=hop["confidence_pct"]
        )
        
    # Step 1: Arrival at Barddhaman (BWN)
    st1 = sim.get_state(step=1)
    stn1_code = st1['current_station']['station_code']
    delay1 = st1['current_delay_mins']
    
    # Network engine must now have stn1 live delay
    assert stn1_code in sim.calculator.network_engine.live_station_delay
    assert sim.calculator.network_engine.live_station_delay[stn1_code] == delay1
    
    # Closed-loop evaluation must match prior prediction
    err1 = logger.record_arrival("12303", stn1_code, delay1)
    assert err1 is not None
    assert err1 >= 0.0
    
    # Step 2: Arrival at Durgapur (DGR)
    st2 = sim.get_state(step=2)
    stn2_code = st2['current_station']['station_code']
    delay2 = st2['current_delay_mins']
    
    assert stn2_code in sim.calculator.network_engine.live_station_delay
    
    # Log step 1 predictions then evaluate step 2
    for hop in st1['comparison_table']:
        logger.log_prediction(
            train_id="12303",
            station_code=hop["station_code"],
            station_name=hop["station_name"],
            predicted_arrival=hop["our_predicted_eta"],
            predicted_delay_mins=hop["our_predicted_delay"],
            confidence_pct=hop["confidence_pct"]
        )
    err2 = logger.record_arrival("12303", stn2_code, delay2)
    assert err2 is not None
    
    metrics = logger.get_metrics()
    assert metrics['evaluated_count'] >= 2
    assert metrics['rolling_mae_mins'] >= 0.0


def test_dynamic_network_delay_injection_effect():
    """Verifies that injecting live station delay alters downstream features and shifts ETAs."""
    sim = ReplaySimulator()
    sim.load_journey(12303, '2024-09-28')
    
    # 1. Normal state (0m delay at ASN)
    sim.calculator.network_engine.update_live_delay('ASN', 0.0)
    feat_normal = sim.calculator.network_engine.compute_journey_downstream_features(
        remaining_sections=sim.journey_sections[:3],
        current_hour_of_day=8,
        day_of_month=28
    )
    
    # 2. Severe delay state (45m delay at ASN)
    sim.calculator.network_engine.update_live_delay('ASN', 45.0)
    feat_congested = sim.calculator.network_engine.compute_journey_downstream_features(
        remaining_sections=sim.journey_sections[:3],
        current_hour_of_day=8,
        day_of_month=28
    )
    
    # Feature vectors must reflect the dynamic injection (not identical)
    assert not (feat_normal == feat_congested).all()
    # 1-hop or weighted delay pressure must be higher in congested state
    assert feat_congested[:, 3].max() >= feat_normal[:, 3].max()


def test_closed_loop_trace_structure():
    """Verifies that closed_loop_trace strictly complies with zero-synthetic policy."""
    sim = ReplaySimulator()
    st = sim.get_state(step=2)
    
    trace = st.get('closed_loop_trace')
    assert trace is not None
    assert 'latest_observation' in trace
    assert 'dynamic_network_update' in trace
    assert 'recomputed_next_eta' in trace
    
    obs = trace['latest_observation']
    assert obs['is_synthetic'] is False
    assert "NTES" in obs['source'] or "RailRadar" in obs['source']


def test_api_demo_live_loop_endpoint(client):
    """Verifies that GET /api/demo/live-loop executes end-to-end and returns complete trace."""
    res = client.get("/api/demo/live-loop?train_number=12303")
    assert res.status_code == 200
    data = res.json()
    
    assert data["status"] == "success"
    assert data["train_number"] == 12303
    assert data["total_stops_evaluated"] > 0
    assert data["journey_rolling_mae_mins"] >= 0.0
    assert len(data["trace"]) > 0
    
    # Verify first step is origin departure
    first_step = data["trace"][0]
    assert first_step["step"] == 0
    assert first_step["event"] == "ORIGIN_DEPARTURE"
    
    # Verify subsequent step has arrival and error
    second_step = data["trace"][1]
    assert second_step["step"] == 1
    assert second_step["event"] == "STATION_ARRIVAL"
    assert "prediction_error_mins" in second_step
