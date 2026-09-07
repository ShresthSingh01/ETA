"""
test_network_state.py - Unit and Integration Tests for Downstream Network State Engine.

Tests:
1. DownstreamNetworkStateEngine station state queries and boundary conditions.
2. Trajectory downstream multi-hop feature calculations.
3. Feature registry tier definitions (M0, M1, M2, M3).
4. API endpoints for network-pressure and ablation-ladder benchmarks.
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient

from src.engine.network_state import DownstreamNetworkStateEngine, NETWORK_STATE_FEATURE_NAMES
from src.model.features import get_feature_names, NETWORK_M1_FEATURES, NETWORK_M2_FEATURES, NETWORK_M3_FEATURES
from src.engine.eta_calculator import ETACalculator
from src.api.main import app


@pytest.fixture
def network_engine():
    return DownstreamNetworkStateEngine(grid_path='data/processed/station_network_grid.npz')


@pytest.fixture
def client():
    return TestClient(app)


def test_feature_registry_tiers():
    m0 = get_feature_names(model_tier='M0')
    m1 = get_feature_names(model_tier='M1')
    m2 = get_feature_names(model_tier='M2')
    m3 = get_feature_names(model_tier='M3')

    assert len(m0) == 23, "M0 must have 23 features"
    assert len(m1) == 27, "M1 must have 27 features"
    assert len(m2) == 31, "M2 must have 31 features"
    assert len(m3) == 34, "M3 must have 34 features"

    # Verify additive hierarchy
    for f in m0:
        assert f in m1 and f in m2 and f in m3
    for f in NETWORK_M1_FEATURES:
        assert f in m1 and f in m2 and f in m3
    for f in NETWORK_M2_FEATURES:
        assert f not in m1 and f in m2 and f in m3
    for f in NETWORK_M3_FEATURES:
        assert f not in m1 and f not in m2 and f in m3


def test_network_state_engine_query(network_engine):
    # Valid station query
    mean_delay, delayed_count, active_count = network_engine.query_station_state('BWN', global_hour=200)
    assert isinstance(mean_delay, float)
    assert isinstance(delayed_count, float)
    assert isinstance(active_count, float)
    assert mean_delay >= 0.0

    # Unknown station fallback
    d_unk, c_unk, a_unk = network_engine.query_station_state('NONEXISTENT_STN', global_hour=200)
    assert d_unk == 0.0
    assert c_unk == 0.0
    assert a_unk == 0.0


def test_trajectory_multi_hop_features(network_engine):
    sample_secs = [
        {'to_station': 'BWN', 'distance_km': 100.0},
        {'to_station': 'DGR', 'distance_km': 70.0},
        {'to_station': 'ASN', 'distance_km': 40.0}
    ]
    feats = network_engine.compute_journey_downstream_features(
        sample_secs,
        current_hour_of_day=10,
        day_of_month=2
    )

    assert feats.shape == (3, 11)
    assert not np.isnan(feats).any(), "Network features must not contain NaN"
    # 1-hop delay must be non-negative
    assert feats[0, 0] >= 0.0
    # Weighted delay must be non-negative
    assert feats[0, 3] >= 0.0


def test_network_pressure_api(client):
    res = client.get('/api/benchmarks/network-pressure')
    assert res.status_code == 200
    data = res.json()
    assert 'Normal (<5m)' in data
    assert 'High (>=30m)' in data
    assert 'Model_M0_Baseline' in data['Normal (<5m)']
    assert 'Model_M3_DownstreamNetwork' in data['Normal (<5m)']


def test_ablation_ladder_api(client):
    res = client.get('/api/benchmarks/ablation-ladder')
    assert res.status_code == 200
    data = res.json()
    assert any('M0' in k for k in data.keys())
    assert any('M1' in k for k in data.keys())
    assert any('M2' in k for k in data.keys())
    assert any('M3' in k for k in data.keys())


def test_station_network_state_api(client):
    res = client.get('/api/network-state/BWN?hour=200')
    assert res.status_code == 200
    data = res.json()
    assert data['station_code'] == 'BWN'
    assert 'mean_delay_mins' in data
    assert 'congestion_pressure' in data
    assert data['congestion_pressure'] in ['NORMAL', 'LOW', 'MEDIUM', 'HIGH']
