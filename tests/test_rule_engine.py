"""
test_rule_engine.py - The 5 Proof Tests for the Deterministic Railway Rule Engine.
"""

import pytest
from src.engine.rule_engine import (
    SectionInfo,
    OperationalEvent,
    apply_railway_rules
)


def test_proof_1_ml_within_bounds_passes_through():
    """Test 1: ML predicts 18 min on a 25 km section where MPS is 100 km/h (min time = 15m).
    ML is within bounds -> passed through unmodified."""
    sec = SectionInfo(
        from_station="STN_A",
        to_station="STN_B",
        distance_km=25.0,
        scheduled_section_time=20.0,
        min_historical_time=15.0,
        p90_time=28.0,
        max_permissible_speed_kmh=100.0
    )
    result = apply_railway_rules(ml_time=18.0, section=sec)
    
    assert result.final_time == 18.0
    assert not result.is_clamped
    assert not result.is_adjusted
    assert len(result.audit_trail) == 0


def test_proof_2_impossible_speed_clamped_to_floor():
    """Test 2: ML predicts 10 min on same section (requires 150 km/h on a 100 km/h track).
    Minimum possible is 15 min -> rule engine clamps to 15 min."""
    sec = SectionInfo(
        from_station="STN_A",
        to_station="STN_B",
        distance_km=25.0,
        scheduled_section_time=20.0,
        min_historical_time=15.0,
        p90_time=28.0,
        max_permissible_speed_kmh=100.0
    )
    result = apply_railway_rules(ml_time=10.0, section=sec)
    
    assert result.final_time == 15.0
    assert result.is_clamped
    assert result.audit_trail[0].rule_name == "MINIMUM_PHYSICAL_RUNNING_TIME"
    assert result.audit_trail[0].delta_minutes == 5.0


def test_proof_3_physics_derived_speed_restriction():
    """Test 3: Active speed restriction of 30 km/h over 15 km on a 100 km/h track.
    Time at 100 km/h for 15 km = 9 min.
    Time at 30 km/h for 15 km = 30 min.
    Additional time added = exactly 21.0 minutes."""
    sec = SectionInfo(
        from_station="ASN",
        to_station="CRJ",
        distance_km=25.0,
        scheduled_section_time=15.0,  # normal operational speed = 100 km/h
        min_historical_time=15.0,
        p90_time=25.0,
        max_permissible_speed_kmh=100.0
    )
    event = OperationalEvent(
        event_type='SPEED_RESTRICTION',
        from_station="ASN",
        to_station="CRJ",
        affected_km=15.0,
        restricted_speed_kmh=30.0
    )
    result = apply_railway_rules(ml_time=16.0, section=sec, active_events=[event])
    
    # 16.0 ml_time + 21.0 tsr_delay = 37.0 min
    assert result.final_time == 37.0
    assert result.is_adjusted
    tsr_log = [log for log in result.audit_trail if log.rule_name == "TEMPORARY_SPEED_RESTRICTION"][0]
    assert tsr_log.delta_minutes == 21.0


def test_proof_4_recovery_capped_at_timetable_allowance():
    """Test 4: Train is 60 min late; scheduled time is 40 min; ML predicts 25 min (37.5% recovery).
    IR operating rule caps recovery at 15% of scheduled time (6 min max recovery).
    Clamped to 40 - 6 = 34 min."""
    sec = SectionInfo(
        from_station="CNB",
        to_station="ETW",
        distance_km=60.0,
        scheduled_section_time=40.0,
        min_historical_time=20.0,
        p90_time=55.0,
        max_permissible_speed_kmh=160.0  # High speed trunk line MPS
    )
    result = apply_railway_rules(
        ml_time=25.0,
        section=sec,
        current_dep_delay=60.0
    )
    
    assert result.final_time == 34.0
    assert result.is_clamped
    rec_log = [log for log in result.audit_trail if log.rule_name == "RECOVERY_MARGIN_CAP"][0]
    assert rec_log.delta_minutes == 9.0  # 34.0 - 25.0 = 9.0 min clamped


def test_proof_5_simultaneous_rules_audit_trail():
    """Test 5: Multiple rules active simultaneously (TSR + minimum bound + recovery cap).
    Engine applies rules deterministically and generates complete structured audit trail."""
    sec = SectionInfo(
        from_station="DDU",
        to_station="PRYJ",
        distance_km=150.0,
        scheduled_section_time=120.0,  # 75 km/h
        min_historical_time=90.0,
        p90_time=160.0,
        max_permissible_speed_kmh=110.0
    )
    event = OperationalEvent(
        event_type='SPEED_RESTRICTION',
        from_station="DDU",
        to_station="PRYJ",
        affected_km=30.0,
        restricted_speed_kmh=45.0  # 30/45*60 - 30/75*60 = 40m - 24m = +16 min
    )
    # ML predicts impossible fast time 70 min (MPS min is 81.8 min, min historical is 90m)
    result = apply_railway_rules(
        ml_time=70.0,
        section=sec,
        active_events=[event],
        current_dep_delay=45.0
    )
    
    # 1. Base clamped by recovery cap (120m - 15% = 102m)
    # 2. TSR adds 16.0 min -> 102.0 + 16.0 = 118.0 min
    assert result.final_time == 118.0
    assert result.is_clamped
    assert result.is_adjusted
    assert len(result.audit_trail) >= 2
    rule_names = [log.rule_name for log in result.audit_trail]
    assert "RECOVERY_MARGIN_CAP" in rule_names
    assert "TEMPORARY_SPEED_RESTRICTION" in rule_names


def test_proof_6_caution_order_and_unscheduled_stop():
    """Test 6: Caution order speed restriction and unscheduled precedence stop."""
    sec = SectionInfo(
        from_station="CNB",
        to_station="PRYJ",
        distance_km=60.0,
        scheduled_section_time=40.0,  # 90 km/h normal speed
        min_historical_time=30.0,
        p90_time=50.0,
        max_permissible_speed_kmh=110.0
    )
    caution_event = OperationalEvent(
        event_type='CAUTION_ORDER',
        from_station="CNB",
        to_station="PRYJ",
        affected_km=15.0,
        restricted_speed_kmh=30.0,  # 15/30*60 - 15/90*60 = 30m - 10m = +20 min
        source_type="CAUTION_ORDER"
    )
    stop_event = OperationalEvent(
        event_type='UNSCHEDULED_STOP',
        from_station="CNB",
        to_station="PRYJ",
        halt_duration_minutes=12.0,
        source_type="MANUAL_ENTRY"
    )
    result = apply_railway_rules(
        ml_time=40.0,
        section=sec,
        active_events=[caution_event, stop_event]
    )
    # 40.0 ml_time + 20.0 caution order + 12.0 unscheduled stop = 72.0 min
    assert result.final_time == 72.0
    assert result.is_adjusted
    log_names = [log.rule_name for log in result.audit_trail]
    assert "CAUTION_ORDER" in log_names
    assert "UNSCHEDULED_STOP" in log_names


def test_proof_7_rule_provenance_metadata():
    """Test 7: Every audit log entry has a valid RuleType and cited source document."""
    sec = SectionInfo(
        from_station="STN_X",
        to_station="STN_Y",
        distance_km=25.0,
        scheduled_section_time=20.0,
        min_historical_time=15.0,
        p90_time=28.0,
        max_permissible_speed_kmh=100.0
    )
    # Trigger MPS floor clamp
    result = apply_railway_rules(ml_time=5.0, section=sec)
    assert len(result.audit_trail) > 0
    log = result.audit_trail[0]
    assert log.rule_type == "PHYSICAL_SAFETY"
    assert "G&SR" in log.source_document or "Rule" in log.source_document


def test_proof_8_signal_hold_operational_event():
    """Test 8: SIGNAL_HOLD adds signal clearance wait time and is audited properly."""
    sec = SectionInfo(
        from_station="CNB",
        to_station="PRYJ",
        distance_km=195.0,
        scheduled_section_time=130.0,
        min_historical_time=110.0,
        p90_time=160.0
    )
    sig_event = OperationalEvent(
        event_type="SIGNAL_HOLD",
        from_station="CNB",
        to_station="PRYJ",
        halt_duration_minutes=8.5,
        source_type="MANUAL_ENTRY"
    )
    res = apply_railway_rules(ml_time=125.0, section=sec, active_events=[sig_event])
    assert res.final_time == 133.5  # 125.0 + 8.5
    assert res.is_adjusted is True
    log_names = [l.rule_name for l in res.audit_trail]
    assert "SIGNAL_HOLD" in log_names
    sig_log = [l for l in res.audit_trail if l.rule_name == "SIGNAL_HOLD"][0]
    assert "signal hold" in sig_log.reason.lower()
    assert sig_log.rule_type == "OPERATIONAL_EVENT"


def test_negative_zero_distance_and_zero_speed():
    """Edge Case: Zero distance and unphysical zero/negative speed must not crash with ZeroDivisionError."""
    sec = SectionInfo(
        from_station="A",
        to_station="B",
        distance_km=0.0,  # Zero distance
        scheduled_section_time=0.0,
        min_historical_time=0.0,
        p90_time=10.0
    )
    event = OperationalEvent(
        event_type="SPEED_RESTRICTION",
        from_station="A",
        to_station="B",
        affected_km=10.0,
        restricted_speed_kmh=-15.0  # Unphysical negative speed
    )
    res = apply_railway_rules(ml_time=10.0, section=sec, active_events=[event])
    assert res.final_time >= 1.0  # Sanity floor preserved without crash


def test_negative_affected_km_isolation():
    """Edge Case: Negative affected km must be clamped to zero without creating negative delays."""
    sec = SectionInfo(
        from_station="A",
        to_station="B",
        distance_km=40.0,
        scheduled_section_time=30.0,
        min_historical_time=25.0,
        p90_time=45.0
    )
    event = OperationalEvent(
        event_type="SPEED_RESTRICTION",
        from_station="A",
        to_station="B",
        affected_km=-10.0,  # Negative distance
        restricted_speed_kmh=20.0
    )
    res = apply_railway_rules(ml_time=30.0, section=sec, active_events=[event])
    # No delay should be added from a negative affected distance
    assert res.final_time == 30.0


def test_event_wrong_section_isolation():
    """Edge Case: Operational event for Section A->B must NOT affect Section B->C."""
    sec_bc = SectionInfo(
        from_station="B",
        to_station="C",
        distance_km=50.0,
        scheduled_section_time=40.0,
        min_historical_time=35.0,
        p90_time=55.0
    )
    event_ab = OperationalEvent(
        event_type="MAINTENANCE_BLOCK",
        from_station="A",
        to_station="B",
        halt_duration_minutes=45.0
    )
    res = apply_railway_rules(ml_time=38.0, section=sec_bc, active_events=[event_ab])
    assert res.final_time == 38.0
    assert res.is_adjusted is False
    assert len(res.audit_trail) == 0


def test_duplicate_additive_events_on_same_section():
    """Edge Case: Multiple active events on the same section accumulate additively."""
    sec = SectionInfo(
        from_station="A",
        to_station="B",
        distance_km=60.0,
        scheduled_section_time=45.0,
        min_historical_time=38.0,
        p90_time=60.0
    )
    ev1 = OperationalEvent(
        event_type="UNSCHEDULED_STOP",
        from_station="A",
        to_station="B",
        halt_duration_minutes=10.0
    )
    ev2 = OperationalEvent(
        event_type="SIGNAL_HOLD",
        from_station="A",
        to_station="B",
        halt_duration_minutes=5.0
    )
    res = apply_railway_rules(ml_time=42.0, section=sec, active_events=[ev1, ev2])
    assert res.final_time == 57.0  # 42 + 10 + 5
    assert len(res.audit_trail) == 2


def test_inactive_event_ignored():
    """Negative Test: Event with active=False must be completely ignored even if stations match."""
    sec = SectionInfo(
        from_station="DELHI",
        to_station="AGRA",
        distance_km=200.0,
        scheduled_section_time=120.0,
        min_historical_time=100.0,
        p90_time=150.0
    )
    inactive_event = OperationalEvent(
        event_type="MAINTENANCE_BLOCK",
        from_station="DELHI",
        to_station="AGRA",
        halt_duration_minutes=60.0,
        active=False
    )
    res = apply_railway_rules(ml_time=115.0, section=sec, active_events=[inactive_event])
    assert res.final_time == 115.0
    assert not res.is_adjusted
    assert len(res.audit_trail) == 0


def test_tsr_zero_affected_distance():
    """Boundary Test: TSR with affected_km=0.0 must add exactly zero delay."""
    sec = SectionInfo(
        from_station="HWH",
        to_station="BWN",
        distance_km=100.0,
        scheduled_section_time=60.0,
        min_historical_time=50.0,
        p90_time=75.0,
        max_permissible_speed_kmh=110.0
    )
    zero_dist_tsr = OperationalEvent(
        event_type="SPEED_RESTRICTION",
        from_station="HWH",
        to_station="BWN",
        affected_km=0.0,
        restricted_speed_kmh=30.0
    )
    res = apply_railway_rules(ml_time=58.0, section=sec, active_events=[zero_dist_tsr])
    assert res.final_time == 58.0
    assert not res.is_adjusted
    assert len(res.audit_trail) == 0


def test_tsr_speed_higher_than_normal():
    """Boundary Test: TSR speed higher than normal operating speed must not reduce running time."""
    sec = SectionInfo(
        from_station="CNB",
        to_station="ALJN",
        distance_km=300.0,
        scheduled_section_time=180.0,  # normal speed = 100 km/h
        min_historical_time=150.0,
        p90_time=220.0,
        max_permissible_speed_kmh=110.0
    )
    fast_tsr = OperationalEvent(
        event_type="SPEED_RESTRICTION",
        from_station="CNB",
        to_station="ALJN",
        affected_km=50.0,
        restricted_speed_kmh=130.0  # Faster than normal speed 100 km/h
    )
    res = apply_railway_rules(ml_time=175.0, section=sec, active_events=[fast_tsr])
    assert res.final_time == 175.0
    assert not res.is_adjusted


def test_early_train_negative_delay_no_recovery_clamp():
    """Edge Case: Train running early (negative delay) should not trigger recovery margin clamping."""
    sec = SectionInfo(
        from_station="MUMBAI",
        to_station="SURAT",
        distance_km=260.0,
        scheduled_section_time=180.0,
        min_historical_time=140.0,
        p90_time=210.0,
        max_permissible_speed_kmh=130.0
    )
    # ML predicts fast running (150 min), train is 10 min early (delay = -10)
    res = apply_railway_rules(ml_time=150.0, section=sec, current_dep_delay=-10.0)
    assert res.final_time == 150.0
    recovery_logs = [l for l in res.audit_trail if l.rule_name == "RECOVERY_MARGIN_CAP"]
    assert len(recovery_logs) == 0


def test_tsr_affected_distance_clamped_to_section_length():
    """Boundary Test: TSR affected_km exceeding section distance must be clamped to section length."""
    sec = SectionInfo(
        from_station="PUNE",
        to_station="LONAVALA",
        distance_km=60.0,
        scheduled_section_time=45.0,  # 80 km/h
        min_historical_time=40.0,
        p90_time=55.0,
        max_permissible_speed_kmh=100.0
    )
    excess_dist_tsr = OperationalEvent(
        event_type="SPEED_RESTRICTION",
        from_station="PUNE",
        to_station="LONAVALA",
        affected_km=150.0,  # Exceeds section distance of 60.0 km
        restricted_speed_kmh=30.0
    )
    res = apply_railway_rules(ml_time=42.0, section=sec, active_events=[excess_dist_tsr])
    # Expected: clamped to 60 km: (60/30 - 60/80) * 60 = (2.0 - 0.75) * 60 = 75.0 min delay
    assert res.final_time == 42.0 + 75.0
    tsr_log = [l for l in res.audit_trail if l.rule_name == "TEMPORARY_SPEED_RESTRICTION"][0]
    assert tsr_log.delta_minutes == 75.0


def test_rule_classification_registry_provenance():
    """Validates 7-class rule categorization and explicit heuristic classification for recovery cap."""
    from src.engine.rule_engine import RULE_METADATA_REGISTRY, RuleClassification, RuleType

    # Check recovery cap is explicitly an ENGINEERING_HEURISTIC (WTT slack practice, not G&SR statutory law)
    rec_meta = RULE_METADATA_REGISTRY["RECOVERY_MARGIN_CAP"]
    assert rec_meta.classification == RuleClassification.ENGINEERING_HEURISTIC
    assert rec_meta.rule_type == RuleType.ENGINEERING_HEURISTIC
    assert "Working Time Table" in rec_meta.source_document

    # Check minimum physical running time is an OFFICIAL_RULE
    mps_meta = RULE_METADATA_REGISTRY["MINIMUM_PHYSICAL_RUNNING_TIME"]
    assert mps_meta.classification == RuleClassification.OFFICIAL_RULE
    assert mps_meta.rule_type == RuleType.PHYSICAL_SAFETY

    # Check outlier ceiling is a MODEL_ASSUMPTION
    ceil_meta = RULE_METADATA_REGISTRY["MAXIMUM_OUTLIER_CEILING"]
    assert ceil_meta.classification == RuleClassification.MODEL_ASSUMPTION

    # Verify every registered rule has complete metadata fields
    for r_id, meta in RULE_METADATA_REGISTRY.items():
        assert meta.rule_id is not None
        assert meta.source_document != ""
        assert meta.classification in RuleClassification


def test_negative_rule_scoping_and_boundaries():
    """Negative and Boundary Testing: verifies that invalid, out-of-scope, expired, or future rules are safely ignored."""
    sec = SectionInfo(
        from_station="DDU",
        to_station="PRYJ",
        distance_km=150.0,
        scheduled_section_time=120.0,
        min_historical_time=90.0,
        p90_time=150.0,
        max_permissible_speed_kmh=130.0,
        zone="NCR",
        division="PRYJ"
    )

    # 1. Inactive Event
    inactive_event = OperationalEvent(
        event_type="MAINTENANCE_BLOCK",
        from_station="DDU",
        to_station="PRYJ",
        halt_duration_minutes=30.0,
        active=False
    )
    assert not inactive_event.affects(sec)

    # 2. Wrong Zone Event
    wrong_zone = OperationalEvent(
        event_type="CAUTION_ORDER",
        from_station="DDU",
        to_station="PRYJ",
        zone="WR",  # Sec is NCR
        affected_km=20.0,
        restricted_speed_kmh=40.0
    )
    assert not wrong_zone.affects(sec)

    # 3. Wrong Division Event
    wrong_div = OperationalEvent(
        event_type="CAUTION_ORDER",
        from_station="DDU",
        to_station="PRYJ",
        division="DHN",  # Sec is PRYJ
        affected_km=20.0,
        restricted_speed_kmh=40.0
    )
    assert not wrong_div.affects(sec)

    # 4. Future Event (Not yet active on current date)
    future_event = OperationalEvent(
        event_type="MAINTENANCE_BLOCK",
        from_station="DDU",
        to_station="PRYJ",
        effective_from="2026-10-01",
        halt_duration_minutes=45.0
    )
    assert not future_event.affects(sec, current_date="2026-09-06")

    # 5. Expired Event
    expired_event = OperationalEvent(
        event_type="MAINTENANCE_BLOCK",
        from_station="DDU",
        to_station="PRYJ",
        effective_to="2026-08-31",
        halt_duration_minutes=45.0
    )
    assert not expired_event.affects(sec, current_date="2026-09-06")

    # 6. Invalid parameters (Negative/Zero speed or affected distance)
    invalid_speed = OperationalEvent(
        event_type="SPEED_RESTRICTION",
        from_station="DDU",
        to_station="PRYJ",
        affected_km=20.0,
        restricted_speed_kmh=0.0  # Invalid
    )
    assert not invalid_speed.affects(sec)

    invalid_dist = OperationalEvent(
        event_type="SPEED_RESTRICTION",
        from_station="DDU",
        to_station="PRYJ",
        affected_km=-10.0,  # Invalid
        restricted_speed_kmh=30.0
    )
    assert not invalid_dist.affects(sec)


def test_operational_event_cancellation():
    """Event Lifecycle: Cancellation marks section final time as 0.0 with cancellation audit reason."""
    sec = SectionInfo(
        from_station="HWH",
        to_station="BWN",
        distance_km=90.0,
        scheduled_section_time=60.0,
        min_historical_time=50.0,
        p90_time=75.0,
        max_permissible_speed_kmh=110.0
    )
    cancel_event = OperationalEvent(
        event_type="CANCELLATION",
        from_station="HWH",
        to_station="BWN"
    )
    res = apply_railway_rules(ml_time=58.0, section=sec, active_events=[cancel_event])
    assert res.final_time == 0.0
    assert res.is_adjusted
    assert any(log.rule_name == "CANCELLATION" for log in res.audit_trail)


def test_operational_event_reschedule_and_diversion():
    """Event Lifecycle: Rescheduling and Diversions add explicit audit buffers."""
    sec = SectionInfo(
        from_station="CNB",
        to_station="NDLS",
        distance_km=440.0,
        scheduled_section_time=300.0,
        min_historical_time=260.0,
        p90_time=360.0,
        max_permissible_speed_kmh=130.0
    )

    # 1. Reschedule by 45 minutes
    resched_event = OperationalEvent(
        event_type="RESCHEDULE",
        from_station="CNB",
        to_station="NDLS",
        halt_duration_minutes=45.0
    )
    res_sched = apply_railway_rules(ml_time=290.0, section=sec, active_events=[resched_event])
    assert res_sched.final_time == 290.0 + 45.0
    assert any(log.rule_name == "RESCHEDULE" for log in res_sched.audit_trail)

    # 2. Diversion via alternate loop (+25 mins)
    divert_event = OperationalEvent(
        event_type="DIVERSION",
        from_station="CNB",
        to_station="NDLS",
        halt_duration_minutes=25.0
    )
    res_div = apply_railway_rules(ml_time=290.0, section=sec, active_events=[divert_event])
    assert res_div.final_time == 290.0 + 25.0
    assert any(log.rule_name == "DIVERSION" for log in res_div.audit_trail)




