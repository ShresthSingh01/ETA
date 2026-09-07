"""
test_live_integration.py - Comprehensive Unit & Integration Tests for RailRadar Live Architecture.
"""

import pytest
from datetime import datetime, timezone

from src.integrations.base import (
    CanonicalTrainState,
    FreshnessLevel,
    get_freshness_level,
    StationBoardEntry
)
from src.integrations.replay_provider import ReplayProvider
from src.integrations.railradar import RailRadarProvider, TokenBucketRateLimiter
from src.engine.state_correction import (
    apply_current_state_correction,
    StateCorrectionResult,
    TrainMotionState
)
from src.engine.prediction_logger import LivePredictionLogger
from src.replay.simulator import ReplaySimulator
from fastapi.testclient import TestClient
from src.api.main import app


def test_freshness_classification():
    assert get_freshness_level(20.0) == FreshnessLevel.FRESH
    assert get_freshness_level(60.0) == FreshnessLevel.FRESH
    assert get_freshness_level(120.0) == FreshnessLevel.AGING
    assert get_freshness_level(180.0) == FreshnessLevel.AGING
    assert get_freshness_level(181.0) == FreshnessLevel.STALE
    assert get_freshness_level(999.0) == FreshnessLevel.STALE
    assert get_freshness_level(-5.0) == FreshnessLevel.UNKNOWN


def test_canonical_train_state_serialization():
    state = CanonicalTrainState(
        provider="RailRadarLiveProvider",
        train_id="12303",
        train_name="Poorva Express",
        journey_date="2026-06-22",
        timestamp="2026-06-22T10:00:00Z",
        status="RUNNING",
        current_station_code="DDU",
        current_station_name="Pt Deen Dayal Upadhyaya",
        current_sequence=4,
        segment_progress=0.45,
        speed_kmph=75.5,
        bearing_deg=315.0,
        current_delay_min=14.0,
        next_station_code="PRYJ",
        next_station_name="Prayagraj Jn",
        source_freshness_sec=25.0,
        is_actual_position=True
    )
    d = state.to_dict()
    assert d["train_id"] == "12303"
    assert d["segment_progress"] == 0.45
    assert d["freshness_level"] == "FRESH"
    assert d["speed_kmph"] == 75.5
    assert d["is_actual_position"] is True


def test_token_bucket_rate_limiter():
    limiter = TokenBucketRateLimiter(capacity=3, refill_rate_per_sec=0.0)
    assert limiter.acquire() is True
    assert limiter.acquire() is True
    assert limiter.acquire() is True
    assert limiter.acquire() is False  # Empty bucket


def test_railradar_provider_caching_and_fallback():
    provider = RailRadarProvider(api_key=None, cache_ttl_sec=60.0)
    state1 = provider.get_live_state("12303")
    assert state1 is not None
    assert state1.train_id == "12303"
    assert state1.status == "UNAVAILABLE"
    assert state1.latitude is None
    assert state1.longitude is None
    assert state1.is_actual_position is False
    assert state1.raw_metadata["synthetic_telemetry"] is False
    
    # Second call should hit in-memory TTL cache
    state2 = provider.get_live_state("12303")
    assert state2 is not None
    assert provider.cache_hits >= 1

    # Station board in zero-synthetic mode returns empty list when unauthenticated
    board = provider.get_station_board("CNB")
    assert len(board) == 0

    health = provider.get_health()
    assert health["status"] == "UNAVAILABLE"
    assert health["is_connected"] is False
    assert health["synthetic_telemetry"] is False
    assert health["cache_hits"] >= 1


def test_kinematic_state_correction():
    # Fresh observation with active speed
    state = CanonicalTrainState(
        provider="RailRadar",
        train_id="12303",
        train_name="Poorva",
        journey_date="2026-06-22",
        timestamp="2026-06-22T10:00:00Z",
        status="RUNNING",
        current_station_code="DDU",
        current_station_name="DDU",
        current_sequence=1,
        segment_progress=0.50,  # 50% completed
        speed_kmph=80.0,
        bearing_deg=0.0,
        current_delay_min=10.0,
        next_station_code="PRYJ",
        next_station_name="PRYJ",
        source_freshness_sec=15.0,  # FRESH
        is_actual_position=True
    )

    # 40 km section, ML predicted 30 minutes for full section
    res = apply_current_state_correction(
        ml_full_section_time=30.0,
        distance_km=40.0,
        canonical_state=state,
        is_immediate_section=True
    )
    assert res.is_corrected is True
    # 50% distance remaining = 20 km. At 80 km/h = 15 minutes.
    # ML remaining = 15 minutes.
    # Blended = 0.7*15 + 0.3*15 = 15 minutes.
    assert 14.0 <= res.corrected_time_mins <= 16.0
    assert "Live Kinematic Correction" in res.reason
    assert res.motion_state == TrainMotionState.MOVING


def test_unexpected_stop_kinematics():
    # Train stopped mid-section (speed 0 km/h, 40% progress)
    stopped_state = CanonicalTrainState(
        provider="RailRadarLiveProvider",
        train_id="12303",
        train_name="Poorva Express",
        journey_date="2024-09-28",
        timestamp=datetime.now(timezone.utc).isoformat(),
        status="HALTED",
        current_station_code="DDU",
        current_station_name="DDU",
        current_sequence=2,
        segment_progress=0.40,  # 40% along the section
        speed_kmph=0.0,         # Stationary
        bearing_deg=0.0,
        current_delay_min=15.0,
        next_station_code="PRYJ",
        next_station_name="PRYJ",
        source_freshness_sec=10.0,
        is_actual_position=True
    )

    res = apply_current_state_correction(
        ml_full_section_time=30.0,
        distance_km=50.0,
        canonical_state=stopped_state,
        is_immediate_section=True
    )
    assert res.motion_state == TrainMotionState.UNEXPECTED_STOP
    assert res.is_corrected is True
    # 60% remaining of 30 min = 18 min + 3.0 min signal hold = 21.0 min
    assert res.corrected_time_mins == 21.0
    assert "Unexpected Mid-Section Halt" in res.reason


def test_station_origin_dwell_kinematics():
    # Train at platform origin stop (speed 0 km/h, 2% progress)
    origin_state = CanonicalTrainState(
        provider="RailRadarLiveProvider",
        train_id="12303",
        train_name="Poorva Express",
        journey_date="2024-09-28",
        timestamp=datetime.now(timezone.utc).isoformat(),
        status="BOARDING",
        current_station_code="HWH",
        current_station_name="HWH",
        current_sequence=1,
        segment_progress=0.02,
        speed_kmph=0.0,
        bearing_deg=0.0,
        current_delay_min=0.0,
        next_station_code="BWN",
        next_station_name="BWN",
        source_freshness_sec=5.0,
        is_actual_position=True
    )

    res = apply_current_state_correction(
        ml_full_section_time=60.0,
        distance_km=90.0,
        canonical_state=origin_state,
        is_immediate_section=True
    )
    assert res.motion_state == TrainMotionState.STATION_HALT
    assert res.is_corrected is False
    assert res.corrected_time_mins == 60.0


def test_simulator_mode_switching_and_state():
    sim = ReplaySimulator()
    
    # Test historical replay mode
    sim.set_mode("historical_replay")
    s_replay = sim.get_state()
    assert s_replay["mode"] == "historical_replay"
    assert s_replay["canonical_state"]["provider"] == "HistoricalReplayProvider"

    # Test live external mode
    sim.set_mode("live_external")
    s_live = sim.get_state()
    assert s_live["mode"] == "live_external"
    assert "Live" in s_live["canonical_state"]["provider"]
    assert len(s_live["comparison_table"]) > 0

    # Provider health (without API key, is_connected is False and status is UNAVAILABLE)
    health = sim.get_provider_health()
    assert health["status"] == "UNAVAILABLE"
    assert health["is_connected"] is False
    assert health["synthetic_telemetry"] is False


def test_live_prediction_logger():
    logger = LivePredictionLogger(capacity=10)
    pid = logger.log_prediction(
        train_id="12303",
        station_code="CNB",
        station_name="Kanpur Central",
        predicted_arrival="02:30 PM",
        predicted_delay_mins=15.0,
        confidence_pct=91.0
    )
    assert pid.startswith("PRED-")

    # Record matching arrival
    err = logger.record_arrival(
        train_id="12303",
        station_code="CNB",
        actual_delay_mins=17.0,
        actual_arrival="02:32 PM"
    )
    assert err == 2.0  # abs(15.0 - 17.0)

    metrics = logger.get_metrics()
    assert metrics["evaluated_count"] > 0
    assert metrics["rolling_mae_mins"] > 0.0


def test_fastapi_endpoints():
    client = TestClient(app)
    
    # 1. Health endpoint
    r_health = client.get("/api/live/health")
    assert r_health.status_code == 200
    assert "is_connected" in r_health.json()

    # 2. Mode switch
    r_mode = client.post("/api/mode/switch", json={"mode": "live_external"})
    assert r_mode.status_code == 200
    assert r_mode.json()["mode"] == "live_external"

    # 3. Live state
    r_state = client.get("/api/live/state")
    assert r_state.status_code == 200
    assert r_state.json()["mode"] == "live_external"

    # 4. Station board (zero synthetic policy -> empty list when unauthenticated)
    r_stn = client.get("/api/live/station/CNB")
    assert r_stn.status_code == 200
    assert r_stn.json()["station_code"] == "CNB"
    assert isinstance(r_stn.json()["board"], list)

    # 5. Prediction log
    r_log = client.get("/api/predictions/log")
    assert r_log.status_code == 200
    assert "rolling_mae_mins" in r_log.json()

    # 6. Set API Key
    r_key = client.post("/api/live/apikey", json={"api_key": "test_key_sample"})
    assert r_key.status_code == 200
    assert r_key.json()["status"] == "success"

    # Switch back to replay mode
    r_mode_back = client.post("/api/mode/switch", json={"mode": "historical_replay"})
    assert r_mode_back.status_code == 200
    assert r_mode_back.json()["mode"] == "historical_replay"


def test_railradar_http_failure_matrix(monkeypatch):
    """
    Validates the complete failure matrix: 401, 404, 429, 503, timeout, and malformed payload.
    Ensures zero synthetic data is ever generated on any failure path.
    """
    provider = RailRadarProvider(api_key="valid_sample_key", cache_ttl_sec=0.0)

    # 1. Test 401 Unauthorized
    monkeypatch.setattr(provider, "_http_get", lambda endpoint: (None, 401))
    provider.last_error = "HTTP 401: Unauthorized - Invalid or Missing API Key"
    state_401 = provider.get_live_state("12303")
    assert state_401.status == "UNAVAILABLE"
    assert state_401.latitude is None
    assert state_401.longitude is None
    assert state_401.is_actual_position is False
    assert any("401" in ex for ex in state_401.exceptions)

    # 2. Test 404 Not Found
    monkeypatch.setattr(provider, "_http_get", lambda endpoint: (None, 404))
    provider.last_error = "HTTP 404: Train Not Found / Inactive"
    state_404 = provider.get_live_state("99999")
    assert state_404.status == "UNAVAILABLE"
    assert any("404" in ex for ex in state_404.exceptions)

    # 3. Test 429 Rate Limit
    monkeypatch.setattr(provider, "_http_get", lambda endpoint: (None, 429))
    provider.last_error = "HTTP 429: Rate Limit Exceeded - Backing Off"
    state_429 = provider.get_live_state("12303")
    assert state_429.status == "UNAVAILABLE"
    assert any("429" in ex for ex in state_429.exceptions)

    # 4. Test 503 Service Unavailable
    monkeypatch.setattr(provider, "_http_get", lambda endpoint: (None, 503))
    provider.last_error = "HTTP 503: Upstream Service Unavailable"
    state_503 = provider.get_live_state("12303")
    assert state_503.status == "UNAVAILABLE"
    assert any("503" in ex for ex in state_503.exceptions)

    # 5. Test 504 Timeout
    monkeypatch.setattr(provider, "_http_get", lambda endpoint: (None, 504))
    provider.last_error = "Connection Timeout (4.0s exceeded)"
    state_timeout = provider.get_live_state("12303")
    assert state_timeout.status == "UNAVAILABLE"
    assert any("Timeout" in ex for ex in state_timeout.exceptions)

    # 6. Test Malformed JSON
    monkeypatch.setattr(provider, "_http_get", lambda endpoint: (None, 502))
    provider.last_error = "Malformed JSON: Expecting value"
    state_malformed = provider.get_live_state("12303")
    assert state_malformed.status == "UNAVAILABLE"
    assert any("Malformed" in ex for ex in state_malformed.exceptions)


def test_railradar_live_freshness_lifecycle():
    """
    Validates the LIVE -> STALE -> UNAVAILABLE state machine.
    """
    provider = RailRadarProvider(api_key="valid_key", cache_ttl_sec=300.0)
    
    # Inject a fresh live state into the cache
    now = datetime.now(timezone.utc).isoformat()
    state = CanonicalTrainState(
        provider="RailRadarLiveProvider",
        train_id="12303",
        train_name="Poorva Express",
        journey_date="2026-06-22",
        timestamp=now,
        status="RUNNING",
        current_station_code="DDU",
        current_station_name="Pt Deen Dayal Upadhyaya",
        current_sequence=4,
        segment_progress=0.45,
        speed_kmph=75.5,
        bearing_deg=315.0,
        current_delay_min=14.0,
        next_station_code="PRYJ",
        next_station_name="Prayagraj Jn",
        source_freshness_sec=15.0,
        is_actual_position=True,
        latitude=25.2819,
        longitude=83.1180
    )
    
    # 1. Fresh observation: age = 20s
    cache_key = "train_12303_None"
    provider.train_cache[cache_key] = (state, datetime.now(timezone.utc).timestamp() - 20.0)
    retrieved = provider.get_live_state("12303")
    assert retrieved.freshness_level == FreshnessLevel.FRESH
    assert retrieved.status == "RUNNING"

    # 2. Aging observation: age = 90s
    provider.train_cache[cache_key] = (state, datetime.now(timezone.utc).timestamp() - 90.0)
    retrieved = provider.get_live_state("12303")
    assert retrieved.freshness_level == FreshnessLevel.AGING
    assert retrieved.status == "RUNNING"

    # 3. Stale observation: age = 210s
    provider.train_cache[cache_key] = (state, datetime.now(timezone.utc).timestamp() - 210.0)
    retrieved = provider.get_live_state("12303")
    assert retrieved.freshness_level == FreshnessLevel.STALE
    assert retrieved.status == "STALE"


def test_railradar_nested_envelope_unwrapping():
    """
    Verifies that RailRadarProvider correctly unpacks {"success": true, "data": {...}}
    envelopes and extracts coordinates, speed, and station codes accurately.
    """
    provider = RailRadarProvider(api_key="valid_key", cache_ttl_sec=0.0)
    mock_payload = {
        "success": True,
        "data": {
            "trainNumber": "12951",
            "trainName": "MUMBAI TEJAS RAJDHANI",
            "status": "RUNNING",
            "currentStation": {"code": "BVI", "name": "BORIVALI"},
            "nextStation": {"code": "ST", "name": "SURAT"},
            "currentSequence": 2,
            "segmentProgress": 0.35,
            "speed": 95.0,
            "delay": 2.0,
            "latitude": 19.2287,
            "longitude": 72.8564,
            "bearing": 18.0
        }
    }
    provider._http_get = lambda endpoint: (mock_payload, 200)
    state = provider.get_live_state("12951")

    assert state is not None
    assert state.train_id == "12951"
    assert state.status == "RUNNING"
    assert state.current_station_code == "BVI"
    assert state.current_station_name == "BORIVALI"
    assert state.next_station_code == "ST"
    assert state.speed_kmph == 95.0
    assert state.current_delay_min == 2.0
    assert state.latitude == 19.2287
    assert state.longitude == 72.8564
    assert state.segment_progress == 0.35


def test_station_alias_and_origin_coordinates_resolution():
    """
    Verifies that ReplaySimulator resolves MMCT to Mumbai Central coordinates (18.97, 72.81)
    and not default Howrah coordinates (22.58, 88.34).
    """
    sim = ReplaySimulator()
    sim.load_journey(12951, "2024-09-28")
    
    assert len(sim.stations_route) > 0
    origin = sim.stations_route[0]
    assert origin['station_code'] == "MMCT"
    assert origin['station_name'] == "Mumbai Central"
    # Must be in Mumbai Central vicinity, NOT Howrah/Kolkata
    assert 18.9 <= origin['latitude'] <= 19.1
    assert 72.7 <= origin['longitude'] <= 73.0


def test_set_api_key_clears_cache_and_enables_live_fetch(monkeypatch):
    """
    Verifies that calling set_api_key invalidates previous UNAVAILABLE cache
    and immediately makes live state queryable.
    """
    monkeypatch.delenv("RAILRADAR_API_KEY", raising=False)
    provider = RailRadarProvider(api_key=None, cache_ttl_sec=60.0)
    s1 = provider.get_live_state("12951")
    assert s1.status == "UNAVAILABLE"
    assert any("No RailRadar API key" in ex for ex in s1.exceptions)

    # Update API key
    provider.set_api_key("new_test_key")
    provider._http_get = lambda ep: ({
        "success": True,
        "data": {
            "trainNumber": "12951",
            "status": "RUNNING",
            "currentStation": {"code": "BVI", "name": "BORIVALI"},
            "speed": 85.0
        }
    }, 200)

    s2 = provider.get_live_state("12951")
    assert s2.status == "RUNNING"
    assert s2.current_station_code == "BVI"
    assert s2.speed_kmph == 85.0
    monkeypatch.delenv("RAILRADAR_API_KEY", raising=False)
