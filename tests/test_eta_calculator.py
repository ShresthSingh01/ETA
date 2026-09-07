"""
test_eta_calculator.py - Unit tests for cumulative ETA calculation, confidence scoring, and event injection.
"""

import pytest
from src.engine.eta_calculator import ETACalculator, StationETA
from src.engine.rule_engine import OperationalEvent


@pytest.fixture(scope="module")
def calculator():
    return ETACalculator(model_path='models/lightgbm_eta.txt')


def test_time_formatting():
    assert ETACalculator.minutes_to_ampm(480) == "08:00 AM"
    assert ETACalculator.minutes_to_ampm(735) == "12:15 PM"
    assert ETACalculator.minutes_to_ampm(1425) == "11:45 PM"
    assert ETACalculator.minutes_to_ampm(1442) == "12:02 AM (+1d)"
    assert ETACalculator.minutes_to_ampm(1500) == "01:00 AM (+1d)"


def test_confidence_decay():
    conf_0, lvl_0 = ETACalculator.compute_confidence(hop_index=0)
    conf_5, lvl_5 = ETACalculator.compute_confidence(hop_index=5)
    conf_20, lvl_20 = ETACalculator.compute_confidence(hop_index=20)
    
    assert conf_0 > conf_5 > conf_20, "Confidence must decay over distance"
    assert lvl_0 == "HIGH"
    assert lvl_20 in ["MEDIUM", "LOW"]


def test_cumulative_eta_forward_progression(calculator):
    # Mock remaining sections for Train 12303: HWH -> BWN -> DGR -> ASN
    sections = [
        {
            'from_station': 'HWH',
            'to_station': 'BWN',
            'to_station_name': 'Barddhaman',
            'distance_km': 95.0,
            'scheduled_section_time': 65.0,
            'scheduled_dwell_to': 3.0,
            'scheduled_arr_to_mins': 545.0,  # 09:05 AM
            'scheduled_dep_to_mins': 548.0,  # 09:08 AM
            'section_median_time': 68.0,
            'section_p90_time': 80.0,
            'section_min_time': 55.0,
            'zone': 'ER'
        },
        {
            'from_station': 'BWN',
            'to_station': 'DGR',
            'to_station_name': 'Durgapur',
            'distance_km': 62.0,
            'scheduled_section_time': 49.0,
            'scheduled_dwell_to': 2.0,
            'scheduled_arr_to_mins': 597.0,  # 09:57 AM
            'scheduled_dep_to_mins': 599.0,  # 09:59 AM
            'section_median_time': 50.0,
            'section_p90_time': 60.0,
            'section_min_time': 42.0,
            'zone': 'ER'
        },
        {
            'from_station': 'DGR',
            'to_station': 'ASN',
            'to_station_name': 'Asansol',
            'distance_km': 42.0,
            'scheduled_section_time': 33.0,
            'scheduled_dwell_to': 5.0,
            'scheduled_arr_to_mins': 632.0,  # 10:32 AM
            'scheduled_dep_to_mins': 637.0,  # 10:37 AM
            'section_median_time': 34.0,
            'section_p90_time': 45.0,
            'section_min_time': 28.0,
            'zone': 'ER'
        }
    ]
    
    # Train departs HWH at 08:00 AM (480 mins) on time
    etas = calculator.predict_journey_etas(
        train_number=12303,
        current_station='HWH',
        current_clock_mins=480.0,
        current_dep_delay_mins=0.0,
        remaining_sections=sections
    )
    
    assert len(etas) == 3
    assert etas[0].station_code == 'BWN'
    assert etas[1].station_code == 'DGR'
    assert etas[2].station_code == 'ASN'
    
    # Times must strictly advance forward
    assert etas[0].predicted_section_time_mins > 0
    assert etas[1].predicted_section_time_mins > 0
    assert etas[2].predicted_section_time_mins > 0
    assert etas[0].confidence_pct > etas[2].confidence_pct


def test_event_injection_impact(calculator):
    sections = [
        {
            'from_station': 'HWH',
            'to_station': 'BWN',
            'to_station_name': 'Barddhaman',
            'distance_km': 95.0,
            'scheduled_section_time': 65.0,
            'scheduled_dwell_to': 3.0,
            'scheduled_arr_to_mins': 545.0,
            'scheduled_dep_to_mins': 548.0,
            'section_median_time': 65.0,
            'section_p90_time': 80.0,
            'section_min_time': 55.0,
            'zone': 'ER'
        }
    ]
    
    # Normal run
    etas_normal = calculator.predict_journey_etas(
        train_number=12303,
        current_station='HWH',
        current_clock_mins=480.0,
        current_dep_delay_mins=0.0,
        remaining_sections=sections
    )
    
    # Event run: TSR of 30 km/h over 20 km on HWH-BWN
    tsr_event = OperationalEvent(
        event_type='SPEED_RESTRICTION',
        from_station='HWH',
        to_station='BWN',
        affected_km=20.0,
        restricted_speed_kmh=30.0
    )
    etas_event = calculator.predict_journey_etas(
        train_number=12303,
        current_station='HWH',
        current_clock_mins=480.0,
        current_dep_delay_mins=0.0,
        remaining_sections=sections,
        active_events=[tsr_event]
    )
    
    # Predicted travel time under TSR must be strictly higher
    assert etas_event[0].predicted_section_time_mins > etas_normal[0].predicted_section_time_mins
    assert etas_event[0].is_rule_adjusted
    assert "TEMPORARY_SPEED_RESTRICTION" in etas_event[0].explanation
