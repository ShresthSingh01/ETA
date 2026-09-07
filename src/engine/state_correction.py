"""
state_correction.py - Post-ML Kinematic State Correction & Motion State Classifier.

Applies real-time speed, segment progress, motion state classification, and observation
freshness to the active section's ML prediction without retraining or corrupting LightGBM weights.

Distinguishes four distinct operational motion states:
1. MOVING: Normal cruising traversal (v >= 15 km/h) -> Freshness-weighted kinematic blend.
2. SLOW_MOVING: Yard / caution crawl (5 <= v < 15 km/h) -> Conservative caution blend.
3. STATION_HALT: Expected origin/platform dwell (v < 5 km/h, progress <= 5%) -> ML timetable baseline.
4. UNEXPECTED_STOP: Mid-section halt (v < 5 km/h, progress > 5%) -> Signal hold buffer addition (+3.0m).
"""

from enum import Enum
from typing import Optional
from dataclasses import dataclass

from src.integrations.base import CanonicalTrainState, FreshnessLevel


class TrainMotionState(str, Enum):
    MOVING = "MOVING"
    SLOW_MOVING = "SLOW_MOVING"
    STATION_HALT = "STATION_HALT"
    UNEXPECTED_STOP = "UNEXPECTED_STOP"


@dataclass
class StateCorrectionResult:
    corrected_time_mins: float
    original_ml_time_mins: float
    is_corrected: bool
    adjustment_minutes: float
    reason: str
    motion_state: TrainMotionState = TrainMotionState.MOVING


def classify_motion_state(speed_kmph: float, segment_progress: float) -> TrainMotionState:
    """
    Classifies train kinematic state to distinguish normal running from unexpected signal holds.
    """
    progress = min(max(segment_progress, 0.0), 1.0)
    speed = max(speed_kmph, 0.0)

    if speed >= 15.0:
        return TrainMotionState.MOVING
    elif speed >= 5.0:
        return TrainMotionState.SLOW_MOVING
    else:
        # speed < 5.0 km/h: stationary or virtually stopped
        if progress <= 0.05 or progress >= 0.98:
            return TrainMotionState.STATION_HALT
        else:
            return TrainMotionState.UNEXPECTED_STOP


def apply_current_state_correction(
    ml_full_section_time: float,
    distance_km: float,
    canonical_state: Optional[CanonicalTrainState],
    is_immediate_section: bool = True
) -> StateCorrectionResult:
    """
    Blends real-time kinematic observation (speed + progress) with ML baseline:
    - Only applies to the active, immediate downstream section.
    - Calculates remaining distance = distance * (1 - segment_progress).
    - Classifies motion state to avoid naive speed division anomalies.
    - If train is unexpectedly halted mid-section, applies signal clearance buffer.
    - If live speed is fresh, blends kinematic traversal time with ML residual.
    """
    if not is_immediate_section or canonical_state is None or canonical_state.status == "UNAVAILABLE":
        return StateCorrectionResult(
            corrected_time_mins=ml_full_section_time,
            original_ml_time_mins=ml_full_section_time,
            is_corrected=False,
            adjustment_minutes=0.0,
            reason="Downstream section or provider unavailable: standard ML + rule trajectory applies",
            motion_state=TrainMotionState.MOVING
        )

    progress = min(max(canonical_state.segment_progress, 0.0), 0.99)
    freshness = canonical_state.freshness_level
    speed = canonical_state.speed_kmph
    motion_state = classify_motion_state(speed, progress)

    # 1. Compute ML portion for remaining fraction of section
    remaining_fraction = 1.0 - progress
    ml_remaining_time = ml_full_section_time * remaining_fraction
    remaining_distance = distance_km * remaining_fraction

    # -------------------------------------------------------------
    # State 1: STATION_HALT (At origin platform, speed < 5 km/h)
    # -------------------------------------------------------------
    if motion_state == TrainMotionState.STATION_HALT:
        return StateCorrectionResult(
            corrected_time_mins=round(ml_full_section_time, 2),
            original_ml_time_mins=round(ml_full_section_time, 2),
            is_corrected=False,
            adjustment_minutes=0.0,
            reason="Station Platform Halt: train at origin stop, using full ML baseline",
            motion_state=motion_state
        )

    # -------------------------------------------------------------
    # State 2: UNEXPECTED_STOP (Mid-section halt, speed < 5 km/h)
    # -------------------------------------------------------------
    if motion_state == TrainMotionState.UNEXPECTED_STOP:
        # Typical Indian Railways signal hold / loop precedence clearance allowance: +3.0 min
        signal_clearance_hold = 3.0
        adjusted_time = ml_remaining_time + signal_clearance_hold
        delta = adjusted_time - ml_full_section_time
        reason = (
            f"Unexpected Mid-Section Halt: train stationary ({speed:.1f} km/h at {progress*100:.0f}% section progress). "
            f"Added +{signal_clearance_hold:.1f}m signal/precedence clearance hold to ML remaining time."
        )
        return StateCorrectionResult(
            corrected_time_mins=round(adjusted_time, 2),
            original_ml_time_mins=round(ml_full_section_time, 2),
            is_corrected=True,
            adjustment_minutes=round(delta, 2),
            reason=reason,
            motion_state=motion_state
        )

    # -------------------------------------------------------------
    # State 3: SLOW_MOVING (Caution crawl, 5 <= speed < 15 km/h)
    # -------------------------------------------------------------
    if motion_state == TrainMotionState.SLOW_MOVING:
        # Conservative caution blend: 85% ML, 15% crawl speed (clamped floor 8 km/h)
        eff_speed = max(speed, 8.0)
        kinematic_remaining = (remaining_distance / eff_speed) * 60.0
        blended_time = (0.85 * ml_remaining_time) + (0.15 * kinematic_remaining)
        delta = blended_time - ml_full_section_time
        reason = (
            f"Caution Crawl Observation: speed {speed:.1f} km/h at {progress*100:.0f}% progress. "
            f"Applied 85/15 conservative caution blend."
        )
        return StateCorrectionResult(
            corrected_time_mins=round(blended_time, 2),
            original_ml_time_mins=round(ml_full_section_time, 2),
            is_corrected=True,
            adjustment_minutes=round(delta, 2),
            reason=reason,
            motion_state=motion_state
        )

    # -------------------------------------------------------------
    # State 4: MOVING (Active cruising speed >= 15 km/h)
    # -------------------------------------------------------------
    if freshness in (FreshnessLevel.FRESH, FreshnessLevel.AGING):
        kinematic_remaining_time = (remaining_distance / speed) * 60.0

        if freshness == FreshnessLevel.FRESH:
            alpha_ml, alpha_speed = 0.70, 0.30
        else:  # AGING
            alpha_ml, alpha_speed = 0.85, 0.15

        blended_time = (alpha_ml * ml_remaining_time) + (alpha_speed * kinematic_remaining_time)
        delta = blended_time - ml_full_section_time

        reason = (
            f"Live Kinematic Correction: {progress*100:.0f}% completed, "
            f"current speed {speed:.1f} km/h blended with ML baseline ({freshness.value} feed)"
        )
        return StateCorrectionResult(
            corrected_time_mins=round(blended_time, 2),
            original_ml_time_mins=round(ml_full_section_time, 2),
            is_corrected=True,
            adjustment_minutes=round(delta, 2),
            reason=reason,
            motion_state=motion_state
        )

    # If telemetry is STALE (>300s old), scale remaining ML time proportionally
    delta = ml_remaining_time - ml_full_section_time
    return StateCorrectionResult(
        corrected_time_mins=round(ml_remaining_time, 2),
        original_ml_time_mins=round(ml_full_section_time, 2),
        is_corrected=True,
        adjustment_minutes=round(delta, 2),
        reason=f"Stale telemetry: Section {progress*100:.0f}% completed, remaining distance {remaining_distance:.1f} km",
        motion_state=motion_state
    )
