"""
rule_engine.py - Post-ML Deterministic Railway Operating Constraint Engine.

Applies Indian Railways General & Subsidiary Rules (G&SR) and physical track limits:
- BOUND: Minimum running time (track MPS limit) and maximum outlier ceiling.
- ADJUST: Physics-derived Temporary Speed Restriction (TSR) delays, maintenance blocks, and recovery caps.
- VALIDATE: Physical sanity checks.
- EXPLAIN: Full audit trail of every adjustment made with reason and numerical delta.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


class RuleType(str, Enum):
    PHYSICAL_SAFETY = "PHYSICAL_SAFETY"
    OPERATIONAL_EVENT = "OPERATIONAL_EVENT"
    ENGINEERING_HEURISTIC = "ENGINEERING_HEURISTIC"


class RuleClassification(str, Enum):
    OFFICIAL_RULE = "OFFICIAL_RULE"                  # Statutory IR General & Subsidiary Rules (G&SR)
    WTT_DATA = "WTT_DATA"                            # Working Time Table scheduled runtimes / allowances
    OPERATIONAL_SOURCE = "OPERATIONAL_SOURCE"        # Real-time control office / COIS / Caution notices
    DERIVED_PHYSICS = "DERIVED_PHYSICS"              # Kinematic equations (v=d/t, acceleration, braking)
    MODEL_ASSUMPTION = "MODEL_ASSUMPTION"            # Mathematical boundary assumptions (e.g. 3x P90 ceiling)
    ENGINEERING_HEURISTIC = "ENGINEERING_HEURISTIC"  # Operational rules of thumb (e.g. 15% slack recovery)
    SIMULATED_EVENT = "SIMULATED_EVENT"              # What-If manual injections


@dataclass
class RuleMetadata:
    rule_id: str
    rule_name: str
    rule_type: RuleType
    classification: RuleClassification
    source_document: str
    source_reference: str
    description: str
    zone: str = "ALL"
    division: str = "ALL"
    section: str = "ALL"
    effective_from: str = "2024-01-01"
    effective_to: str = "2099-12-31"
    priority: int = 100
    parameters: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"


RULE_METADATA_REGISTRY: Dict[str, RuleMetadata] = {
    "MINIMUM_PHYSICAL_RUNNING_TIME": RuleMetadata(
        rule_id="RULE-GSR-408",
        rule_name="MINIMUM_PHYSICAL_RUNNING_TIME",
        rule_type=RuleType.PHYSICAL_SAFETY,
        classification=RuleClassification.OFFICIAL_RULE,
        source_document="IR General & Subsidiary Rules (G&SR)",
        source_reference="Chapter IV, Rule 4.08 & Schedule of Dimensions (IRSOD)",
        description="Limits minimum section running time to physical track Maximum Permissible Speed (MPS)."
    ),
    "MAXIMUM_OUTLIER_CEILING": RuleMetadata(
        rule_id="RULE-MOD-CEIL",
        rule_name="MAXIMUM_OUTLIER_CEILING",
        rule_type=RuleType.ENGINEERING_HEURISTIC,
        classification=RuleClassification.MODEL_ASSUMPTION,
        source_document="Empirical Data Quality Specification",
        source_reference="Section Traversal P90 Bounds Analysis",
        description="Bounds section running time to 3x section P90 to eliminate unphysical multi-day anomalies."
    ),
    "RECOVERY_MARGIN_CAP": RuleMetadata(
        rule_id="RULE-ENG-REC15",
        rule_name="RECOVERY_MARGIN_CAP",
        rule_type=RuleType.ENGINEERING_HEURISTIC,
        classification=RuleClassification.ENGINEERING_HEURISTIC,
        source_document="Indian Railways Working Time Table (WTT) Slack Allowance Practice",
        source_reference="Operating Manual Timetable Make-Up Buffer Guidelines",
        description="Caps section recovery to ~15% of scheduled section time (empirical timetable cushion limit, not a statutory G&SR limit)."
    ),
    "TEMPORARY_SPEED_RESTRICTION": RuleMetadata(
        rule_id="RULE-OPS-TSR",
        rule_name="TEMPORARY_SPEED_RESTRICTION",
        rule_type=RuleType.OPERATIONAL_EVENT,
        classification=RuleClassification.OPERATIONAL_SOURCE,
        source_document="IR G&SR Rule 4.09 & Permanent Way Manual TSR Register",
        source_reference="PWM Para 208 / Engineering Restriction Notices",
        description="Computes deceleration, traversal at restricted speed, and acceleration time over affected length."
    ),
    "CAUTION_ORDER": RuleMetadata(
        rule_id="RULE-OPS-T409",
        rule_name="CAUTION_ORDER",
        rule_type=RuleType.OPERATIONAL_EVENT,
        classification=RuleClassification.OPERATIONAL_SOURCE,
        source_document="Indian Railways Form T/409 (Caution Order)",
        source_reference="Operating Department Daily Train Running Circulars",
        description="Enforces speed restriction issued via official Caution Order notice Form T/409."
    ),
    "MAINTENANCE_BLOCK": RuleMetadata(
        rule_id="RULE-OPS-MBLOCK",
        rule_name="MAINTENANCE_BLOCK",
        rule_type=RuleType.OPERATIONAL_EVENT,
        classification=RuleClassification.OPERATIONAL_SOURCE,
        source_document="Traffic and Power Block Registers / COIS Engineering Block Notice",
        source_reference="Engineering Code Chapter XI (Traffic Blocks)",
        description="Adds scheduled track/overhead equipment maintenance hold duration."
    ),
    "UNSCHEDULED_STOP": RuleMetadata(
        rule_id="RULE-OPS-USTOP",
        rule_name="UNSCHEDULED_STOP",
        rule_type=RuleType.OPERATIONAL_EVENT,
        classification=RuleClassification.OPERATIONAL_SOURCE,
        source_document="Section Controller Operational Order / Loop Precedence Hold",
        source_reference="Control Office Log / Precedence Circular",
        description="Adds unscheduled station or block section halt duration (e.g. crossing precedence)."
    ),
    "SIGNAL_HOLD": RuleMetadata(
        rule_id="RULE-OPS-SIGHOLD",
        rule_name="SIGNAL_HOLD",
        rule_type=RuleType.OPERATIONAL_EVENT,
        classification=RuleClassification.OPERATIONAL_SOURCE,
        source_document="Section Controller Automatic Signaling Hold / Home Signal Precedence",
        source_reference="Signal Engineering Manual Part II",
        description="Adds operational wait time at absolute/automatic block signal pending line clearance."
    ),
    "CANCELLATION": RuleMetadata(
        rule_id="RULE-OPS-CANCEL",
        rule_name="CANCELLATION",
        rule_type=RuleType.OPERATIONAL_EVENT,
        classification=RuleClassification.OPERATIONAL_SOURCE,
        source_document="Emergency Operational Bulletin / Commercial Circular",
        source_reference="Railway Board Regulation Order",
        description="Terminates journey progress and marks forward ETAs as non-applicable."
    ),
    "DIVERSION": RuleMetadata(
        rule_id="RULE-OPS-DIVERT",
        rule_name="DIVERSION",
        rule_type=RuleType.OPERATIONAL_EVENT,
        classification=RuleClassification.OPERATIONAL_SOURCE,
        source_document="Traffic Diversion Notice / Route Modification Circular",
        source_reference="Zonal Railway Headquarters Special Circular",
        description="Bypasses blocked corridor sections and recalculates traversal via alternate route."
    ),
    "RESCHEDULE": RuleMetadata(
        rule_id="RULE-OPS-RESCHED",
        rule_name="RESCHEDULE",
        rule_type=RuleType.OPERATIONAL_EVENT,
        classification=RuleClassification.OPERATIONAL_SOURCE,
        source_document="NTES Rescheduling Bulletin / Passenger Notification",
        source_reference="Control Office Rescheduling Order",
        description="Shifts departure baseline by designated rescheduling duration."
    )
}


@dataclass
class SectionInfo:
    from_station: str
    to_station: str
    distance_km: float
    scheduled_section_time: float
    min_historical_time: float
    p90_time: float
    max_permissible_speed_kmh: float = 110.0  # Default Broad Gauge trunk line MPS
    zone: str = "ALL"
    division: str = "ALL"


@dataclass
class OperationalEvent:
    event_type: str  # 'SPEED_RESTRICTION', 'CAUTION_ORDER', 'MAINTENANCE_BLOCK', 'UNSCHEDULED_STOP', 'SIGNAL_HOLD', 'CANCELLATION', 'DIVERSION', 'RESCHEDULE'
    from_station: str
    to_station: str
    affected_km: float = 0.0
    restricted_speed_kmh: float = 30.0
    halt_duration_minutes: float = 0.0
    active: bool = True
    source_type: str = "MANUAL_ENTRY"  # 'LIVE', 'HISTORICAL', 'DERIVED', 'SIMULATED', 'OPERATIONAL'
    event_id: Optional[str] = None
    zone: Optional[str] = None
    division: Optional[str] = None
    effective_from: Optional[str] = None  # YYYY-MM-DD
    effective_to: Optional[str] = None    # YYYY-MM-DD
    created_at: Optional[str] = None

    def affects(self, section: SectionInfo, current_date: Optional[str] = None) -> bool:
        """
        Negative and boundary validation rules:
        - Inactive events are ignored.
        - Wrong zone / division / section events are ignored.
        - Expired or future events are ignored.
        - Invalid speed or negative affected distances are rejected.
        """
        if not self.active:
            return False

        # Route-wide events (Cancellation / Reschedule) affect all sections on train route
        if self.event_type not in ('CANCELLATION', 'RESCHEDULE'):
            if self.from_station != section.from_station or self.to_station != section.to_station:
                return False

        # Zone scoping
        if self.zone and section.zone != "ALL" and self.zone != section.zone:
            return False

        # Division scoping
        if self.division and section.division != "ALL" and self.division != section.division:
            return False

        # Temporal validity scoping
        if current_date:
            if self.effective_from and current_date < self.effective_from:
                return False
            if self.effective_to and current_date > self.effective_to:
                return False

        # Parameter sanity checking
        if self.event_type in ('SPEED_RESTRICTION', 'CAUTION_ORDER'):
            if self.restricted_speed_kmh <= 0 or self.affected_km <= 0:
                return False

        if self.event_type in ('MAINTENANCE_BLOCK', 'UNSCHEDULED_STOP', 'SIGNAL_HOLD', 'RESCHEDULE', 'DIVERSION'):
            if self.halt_duration_minutes < 0:
                return False

        return True


@dataclass
class RuleAdjustmentLog:
    rule_name: str
    original_time: float
    adjusted_time: float
    delta_minutes: float
    reason: str
    rule_type: str = "ENGINEERING_HEURISTIC"
    classification: str = "ENGINEERING_HEURISTIC"
    source_document: str = ""


@dataclass
class RuleEngineResult:
    final_time: float
    original_ml_time: float
    is_clamped: bool
    is_adjusted: bool
    audit_trail: List[RuleAdjustmentLog] = field(default_factory=list)


def apply_railway_rules(
    ml_time: float,
    section: SectionInfo,
    active_events: Optional[List[OperationalEvent]] = None,
    current_dep_delay: float = 0.0
) -> RuleEngineResult:
    """
    Applies the 4-stage rule engine pipeline:
    1. BOUND (Minimum running time floor based on MPS & historical traversal)
    2. ADJUST (Operational events: TSR physics & recovery cap)
    3. VALIDATE (Sanity bounds)
    4. EXPLAIN (Audit trail log)
    """
    if active_events is None:
        active_events = []
        
    t = float(ml_time)
    audit: List[RuleAdjustmentLog] = []
    
    # -------------------------------------------------------------
    # 0. CHECK: Journey / Section Cancellation
    # -------------------------------------------------------------
    cancellation_event = next((e for e in active_events if e.affects(section) and e.event_type == 'CANCELLATION'), None)
    if cancellation_event is not None:
        meta = RULE_METADATA_REGISTRY.get("CANCELLATION")
        audit.append(RuleAdjustmentLog(
            rule_name="CANCELLATION",
            original_time=round(t, 2),
            adjusted_time=0.0,
            delta_minutes=round(-t, 2),
            reason="Train journey / section cancelled by operational regulation order (forward ETA not applicable).",
            rule_type=meta.rule_type.value if meta else "OPERATIONAL_EVENT",
            classification=meta.classification.value if meta else "OPERATIONAL_SOURCE",
            source_document=meta.source_document if meta else ""
        ))
        return RuleEngineResult(
            final_time=0.0,
            original_ml_time=round(ml_time, 2),
            is_clamped=False,
            is_adjusted=True,
            audit_trail=audit
        )

    # -------------------------------------------------------------
    # 1. BOUND: Physical Minimum Running Time & Outlier Ceiling
    # -------------------------------------------------------------
    mps_min_time = (section.distance_km / section.max_permissible_speed_kmh) * 60.0
    min_floor = max(mps_min_time, section.min_historical_time * 0.95, 1.0)
    
    if t < min_floor:
        delta = min_floor - t
        meta = RULE_METADATA_REGISTRY["MINIMUM_PHYSICAL_RUNNING_TIME"]
        audit.append(RuleAdjustmentLog(
            rule_name="MINIMUM_PHYSICAL_RUNNING_TIME",
            original_time=round(t, 2),
            adjusted_time=round(min_floor, 2),
            delta_minutes=round(delta, 2),
            reason=f"Predicted time {t:.1f}m violates track MPS {section.max_permissible_speed_kmh} km/h (min time: {min_floor:.1f}m)",
            rule_type=meta.rule_type.value,
            classification=meta.classification.value,
            source_document=meta.source_document
        ))
        t = min_floor

    # Outlier ceiling: no section should reasonably exceed 3x p90
    max_ceiling = max(section.p90_time * 3.0, section.scheduled_section_time * 3.5, 30.0)
    if t > max_ceiling:
        delta = max_ceiling - t
        meta = RULE_METADATA_REGISTRY["MAXIMUM_OUTLIER_CEILING"]
        audit.append(RuleAdjustmentLog(
            rule_name="MAXIMUM_OUTLIER_CEILING",
            original_time=round(t, 2),
            adjusted_time=round(max_ceiling, 2),
            delta_minutes=round(delta, 2),
            reason=f"Predicted time {t:.1f}m exceeds 3x section p90 ceiling ({max_ceiling:.1f}m)",
            rule_type=meta.rule_type.value,
            classification=meta.classification.value,
            source_document=meta.source_document
        ))
        t = max_ceiling

    # -------------------------------------------------------------
    # 2. ADJUST: Recovery Cap (Applied to base running time)
    # -------------------------------------------------------------
    if current_dep_delay > 0 and t < section.scheduled_section_time:
        max_recovery_minutes = section.scheduled_section_time * 0.15
        capped_min_time = section.scheduled_section_time - max_recovery_minutes
        if t < capped_min_time:
            delta = capped_min_time - t
            meta = RULE_METADATA_REGISTRY["RECOVERY_MARGIN_CAP"]
            audit.append(RuleAdjustmentLog(
                rule_name="RECOVERY_MARGIN_CAP",
                original_time=round(t, 2),
                adjusted_time=round(capped_min_time, 2),
                delta_minutes=round(delta, 2),
                reason=f"Recovery capped to engineering heuristic 15% timetable cushion limit ({max_recovery_minutes:.1f}m max recovery)",
                rule_type=meta.rule_type.value,
                classification=meta.classification.value,
                source_document=meta.source_document
            ))
            t = capped_min_time

    # -------------------------------------------------------------
    # 3. ADJUST: Operational Events (TSR / Caution Orders / Blocks / Diversion / Reschedule)
    # -------------------------------------------------------------
    for event in active_events:
        if event.affects(section):
            if event.event_type in ('SPEED_RESTRICTION', 'CAUTION_ORDER'):
                eff_speed = max(event.restricted_speed_kmh, 10.0)
                eff_dist = min(max(event.affected_km, 0.0), section.distance_km)
                safe_sch_time = max(section.scheduled_section_time, 1.0)
                safe_dist = max(section.distance_km, 0.1)

                if eff_dist > 0.0:
                    normal_speed = safe_dist / (safe_sch_time / 60.0)
                    if eff_speed < normal_speed:
                        restricted_time = (eff_dist / eff_speed) * 60.0
                        normal_time = (eff_dist / normal_speed) * 60.0
                        tsr_delay = max(restricted_time - normal_time, 0.0)
                        new_t = t + tsr_delay
                        rule_name = "CAUTION_ORDER" if event.event_type == 'CAUTION_ORDER' else "TEMPORARY_SPEED_RESTRICTION"
                        meta = RULE_METADATA_REGISTRY.get(rule_name)
                        audit.append(RuleAdjustmentLog(
                            rule_name=rule_name,
                            original_time=round(t, 2),
                            adjusted_time=round(new_t, 2),
                            delta_minutes=round(tsr_delay, 2),
                            reason=f"{rule_name} of {eff_speed:.1f} km/h over {eff_dist:.1f} km added {tsr_delay:.1f}m",
                            rule_type=meta.rule_type.value if meta else "OPERATIONAL_EVENT",
                            classification=meta.classification.value if meta else "OPERATIONAL_SOURCE",
                            source_document=meta.source_document if meta else ""
                        ))
                        t = new_t
            elif event.event_type in ('MAINTENANCE_BLOCK', 'UNSCHEDULED_STOP', 'SIGNAL_HOLD'):
                block_delay = max(event.halt_duration_minutes, 0.0)
                if block_delay > 0.0:
                    new_t = t + block_delay
                    rule_name = event.event_type
                    meta = RULE_METADATA_REGISTRY.get(rule_name)
                    if event.event_type == 'SIGNAL_HOLD':
                        reason_msg = f"Automatic/Manual signal hold waiting on block clearance adds {block_delay:.1f}m delay"
                    elif event.event_type == 'UNSCHEDULED_STOP':
                        reason_msg = f"Unscheduled halt/precedence hold adds {block_delay:.1f}m waiting time"
                    else:
                        reason_msg = f"Active track maintenance block adds {block_delay:.1f}m waiting time"

                    audit.append(RuleAdjustmentLog(
                        rule_name=rule_name,
                        original_time=round(t, 2),
                        adjusted_time=round(new_t, 2),
                        delta_minutes=round(block_delay, 2),
                        reason=reason_msg,
                        rule_type=meta.rule_type.value if meta else "OPERATIONAL_EVENT",
                        classification=meta.classification.value if meta else "OPERATIONAL_SOURCE",
                        source_document=meta.source_document if meta else ""
                    ))
                    t = new_t
            elif event.event_type == 'RESCHEDULE':
                resched_delay = max(event.halt_duration_minutes, 0.0)
                if resched_delay > 0.0:
                    new_t = t + resched_delay
                    meta = RULE_METADATA_REGISTRY.get("RESCHEDULE")
                    audit.append(RuleAdjustmentLog(
                        rule_name="RESCHEDULE",
                        original_time=round(t, 2),
                        adjusted_time=round(new_t, 2),
                        delta_minutes=round(resched_delay, 2),
                        reason=f"Train rescheduled by operational order: +{resched_delay:.1f}m departure shift",
                        rule_type=meta.rule_type.value if meta else "OPERATIONAL_EVENT",
                        classification=meta.classification.value if meta else "OPERATIONAL_SOURCE",
                        source_document=meta.source_document if meta else ""
                    ))
                    t = new_t
            elif event.event_type == 'DIVERSION':
                div_delay = max(event.halt_duration_minutes, 10.0)
                new_t = t + div_delay
                meta = RULE_METADATA_REGISTRY.get("DIVERSION")
                audit.append(RuleAdjustmentLog(
                    rule_name="DIVERSION",
                    original_time=round(t, 2),
                    adjusted_time=round(new_t, 2),
                    delta_minutes=round(div_delay, 2),
                    reason=f"Route diverted via bypass corridor (+{div_delay:.1f}m recalculation allowance)",
                    rule_type=meta.rule_type.value if meta else "OPERATIONAL_EVENT",
                    classification=meta.classification.value if meta else "OPERATIONAL_SOURCE",
                    source_document=meta.source_document if meta else ""
                ))
                t = new_t

    # -------------------------------------------------------------
    # 4. VALIDATE: Physical Absolute Floor
    # -------------------------------------------------------------
    t = max(t, 1.0)
    
    is_clamped = any(log.rule_name in ["MINIMUM_PHYSICAL_RUNNING_TIME", "MAXIMUM_OUTLIER_CEILING", "RECOVERY_MARGIN_CAP"] for log in audit)
    is_adjusted = any(log.rule_name in ["TEMPORARY_SPEED_RESTRICTION", "CAUTION_ORDER", "MAINTENANCE_BLOCK", "UNSCHEDULED_STOP", "SIGNAL_HOLD", "CANCELLATION", "DIVERSION", "RESCHEDULE"] for log in audit)

    return RuleEngineResult(
        final_time=round(t, 2),
        original_ml_time=round(ml_time, 2),
        is_clamped=is_clamped,
        is_adjusted=is_adjusted,
        audit_trail=audit
    )
