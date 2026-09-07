"""
eta_calculator.py - Cumulative ETA Prediction & Accumulator Engine.

Takes live train state, sequence of remaining route sections, and operational events;
generates ML + Rule-constrained cumulative ETAs, confidence scores, and audit trails.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import lightgbm as lgb

from src.engine.rule_engine import (
    SectionInfo,
    OperationalEvent,
    RuleEngineResult,
    apply_railway_rules
)
from src.model.features import get_feature_names, CATEGORICAL_FEATURES
from src.integrations.base import CanonicalTrainState, FreshnessLevel
from src.engine.state_correction import apply_current_state_correction
from src.engine.network_state import DownstreamNetworkStateEngine


@dataclass
class StationETA:
    station_code: str
    station_name: str
    scheduled_arrival: str
    scheduled_departure: str
    predicted_arrival: str
    predicted_departure: str
    predicted_arr_delay_mins: float
    predicted_dep_delay_mins: float
    scheduled_section_time_mins: float
    predicted_section_time_mins: float
    original_ml_time_mins: float
    is_rule_adjusted: bool
    confidence_pct: float
    confidence_level: str
    explanation: str


class ETACalculator:
    def __init__(
        self,
        model_path: str = 'models/lightgbm_eta.txt',
        fallback_speed_kmh: float = 60.0,
        network_grid_path: str = 'data/processed/station_network_grid.npz'
    ):
        self.model_path = model_path
        self.fallback_speed_kmh = fallback_speed_kmh
        self.booster = None
        self.network_engine = DownstreamNetworkStateEngine(grid_path=network_grid_path)
        self.feature_names = get_feature_names(model_tier='M3')
        
        try:
            self.booster = lgb.Booster(model_file=model_path)
        except Exception as e:
            print(f"Warning: Could not load LightGBM model from {model_path}: {e}")

    @staticmethod
    def minutes_to_ampm(total_minutes: float) -> str:
        """Converts minute of day (can be >= 1440 for next day) to 'HH:MM AM/PM'."""
        mins_in_day = int(total_minutes) % 1440
        hours = mins_in_day // 60
        minutes = mins_in_day % 60
        ampm = "AM" if hours < 12 else "PM"
        display_hour = hours % 12
        if display_hour == 0:
            display_hour = 12
        day_offset = int(total_minutes) // 1440
        day_str = f" (+{day_offset}d)" if day_offset > 0 else ""
        return f"{display_hour:02d}:{minutes:02d} {ampm}{day_str}"

    @staticmethod
    def compute_confidence(
        hop_index: int,
        is_foggy: bool = False,
        is_heavy_rain: bool = False,
        edge_ntrains: int = 10,
        net_pressure: float = 0.0,
        canonical_state: Optional[CanonicalTrainState] = None
    ) -> Tuple[float, str]:
        """
        Dynamically computes empirical prediction confidence:
        Base: 95%
        - 1.5% per downstream station hop
        - 5% if adverse weather (fog/rain)
        - 2-4% if downstream route has severe delay pressure
        + density bonus up to 3%
        ± live feed freshness and position reliability adjustment
        """
        base = 95.0
        decay = hop_index * 1.5
        weather_penalty = 5.0 if (is_foggy or is_heavy_rain) else 0.0
        pressure_penalty = 4.0 if net_pressure >= 30.0 else (2.0 if net_pressure >= 15.0 else 0.0)
        density_bonus = min(edge_ntrains * 0.1, 3.0)
        
        telemetry_adj = 0.0
        if canonical_state is not None:
            freshness = canonical_state.freshness_level
            if freshness == FreshnessLevel.FRESH:
                telemetry_adj += 2.0
            elif freshness == FreshnessLevel.AGING:
                telemetry_adj -= 6.0
            elif freshness == FreshnessLevel.STALE:
                telemetry_adj -= 16.0
                
            if not canonical_state.is_actual_position:
                telemetry_adj -= 8.0

        conf = max(min(base - decay - weather_penalty - pressure_penalty + density_bonus + telemetry_adj, 98.0), 25.0)
        conf = round(conf, 1)
        
        if conf >= 80.0:
            level = 'HIGH'
        elif conf >= 60.0:
            level = 'MEDIUM'
        else:
            level = 'LOW'
        return conf, level

    def predict_journey_etas(
        self,
        train_number: int,
        current_station: str,
        current_clock_mins: float,
        current_dep_delay_mins: float,
        remaining_sections: List[Dict[str, Any]],
        active_events: Optional[List[OperationalEvent]] = None,
        canonical_state: Optional[CanonicalTrainState] = None
    ) -> List[StationETA]:
        """
        Computes station-by-station forward ETA trajectory along the train's route,
        integrating live kinematics and downstream network state.
        """
        if active_events is None:
            active_events = []
            
        station_etas: List[StationETA] = []
        running_clock_mins = current_clock_mins
        running_delay = current_dep_delay_mins
        
        ZONE_MAP = {
            'CR': 0, 'ECOR': 1, 'ECR': 2, 'ER': 3, 'KRCL': 4,
            'NCR': 5, 'NE': 6, 'NFR': 7, 'NR': 8, 'NWR': 9,
            'SCR': 10, 'SECR': 11, 'SER': 12, 'SR': 13, 'SWR': 14,
            'WCR': 15, 'WR': 16
        }

        n_sections = len(remaining_sections)
        num_features = self.booster.num_feature() if self.booster is not None else 34
        feat_mat = np.zeros((n_sections, num_features), dtype=np.float64)
        
        for idx, sec in enumerate(remaining_sections):
            sch_t = float(sec.get('scheduled_section_time', 25.0))
            feat_mat[idx, 0] = sch_t
            feat_mat[idx, 1] = float(sec.get('distance_km', 25.0))
            feat_mat[idx, 4] = float(sec.get('scheduled_dwell_from', 2.0))
            feat_mat[idx, 5] = float(sec.get('section_median_time', sch_t))
            feat_mat[idx, 6] = float(sec.get('section_mean_time', sch_t))
            feat_mat[idx, 7] = float(sec.get('section_p90_time', sch_t * 1.3))
            feat_mat[idx, 8] = float(sec.get('section_min_time', sch_t * 0.85))
            feat_mat[idx, 9] = float(sec.get('section_std_time', 5.0))
            feat_mat[idx, 10] = float(sec.get('edge_ntrains', 5))
            feat_mat[idx, 12] = float(sec.get('day_of_week', 2))
            feat_mat[idx, 13] = float(sec.get('is_weekend', 0))
            feat_mat[idx, 14] = float(sec.get('day_of_month', 15))
            feat_mat[idx, 15] = float(sec.get('temperature_2m', 28.0))
            feat_mat[idx, 16] = float(sec.get('precipitation', 0.0))
            feat_mat[idx, 17] = float(sec.get('weather_code', 0))
            feat_mat[idx, 18] = float(sec.get('wind_speed_10m', 10.0))
            feat_mat[idx, 19] = float(sec.get('visibility', 10000.0))
            feat_mat[idx, 20] = float(sec.get('is_foggy', 0))
            feat_mat[idx, 21] = float(sec.get('is_heavy_rain', 0))

        if num_features >= 34:
            # Populate 11 Downstream Network Features (columns 22 to 32)
            cur_hour = int((current_clock_mins // 60) % 24)
            cur_day = int(remaining_sections[0].get('day_of_month', 15)) if remaining_sections else 15
            net_feats = self.network_engine.compute_journey_downstream_features(
                remaining_sections=remaining_sections,
                current_hour_of_day=cur_hour,
                day_of_month=cur_day
            )
            feat_mat[:, 22:33] = net_feats
            for idx, sec in enumerate(remaining_sections):
                feat_mat[idx, 33] = float(ZONE_MAP.get(str(sec.get('zone', 'NR')).upper(), 8))
        else:
            for idx, sec in enumerate(remaining_sections):
                feat_mat[idx, 22] = float(ZONE_MAP.get(str(sec.get('zone', 'NR')).upper(), 8))

        running_pressure = 0.0

        for hop_idx, sec in enumerate(remaining_sections):
            from_stn = sec.get('from_station', '')
            to_stn = sec.get('to_station', '')
            to_name = sec.get('to_station_name', to_stn)
            dist_km = float(sec.get('distance_km', 25.0))
            sch_sec_time = float(sec.get('scheduled_section_time', 25.0))
            sch_dwell = float(sec.get('scheduled_dwell_to', 2.0))
            sch_arr_time_mins = float(sec.get('scheduled_arr_to_mins', running_clock_mins + sch_sec_time))
            sch_dep_time_mins = float(sec.get('scheduled_dep_to_mins', sch_arr_time_mins + sch_dwell))
            
            # Dynamic features for current hop
            feat_mat[hop_idx, 2] = running_delay
            feat_mat[hop_idx, 3] = running_delay
            feat_mat[hop_idx, 11] = float((running_clock_mins // 60) % 24)

            # 1. Fast ML Predict on NumPy slice (0.1ms)
            ml_time = sch_sec_time
            if self.booster is not None:
                try:
                    ml_pred = self.booster.predict(feat_mat[hop_idx:hop_idx+1])
                    ml_time = float(ml_pred[0])
                except Exception:
                    ml_time = sch_sec_time
                    
            # 1.5 Kinematic State Correction (Immediate active hop)
            kinematic_reason = None
            if hop_idx == 0 and canonical_state is not None:
                state_res = apply_current_state_correction(
                    ml_full_section_time=ml_time,
                    distance_km=dist_km,
                    canonical_state=canonical_state,
                    is_immediate_section=True
                )
                if state_res.is_corrected:
                    ml_time = state_res.corrected_time_mins
                    kinematic_reason = state_res.reason

            # 2. Rule Engine
            sec_info = SectionInfo(
                from_station=from_stn,
                to_station=to_stn,
                distance_km=dist_km,
                scheduled_section_time=sch_sec_time,
                min_historical_time=float(sec.get('section_min_time', sch_sec_time * 0.85)),
                p90_time=float(sec.get('section_p90_time', sch_sec_time * 1.3)),
                max_permissible_speed_kmh=float(sec.get('max_permissible_speed_kmh', 110.0))
            )
            
            rule_res: RuleEngineResult = apply_railway_rules(
                ml_time=ml_time,
                section=sec_info,
                active_events=active_events,
                current_dep_delay=running_delay
            )
            
            final_sec_time = rule_res.final_time
            
            # 3. Accumulate clock and delay
            arr_clock_mins = running_clock_mins + final_sec_time
            arr_delay_mins = round(arr_clock_mins - sch_arr_time_mins, 1)
            
            # Dwell at arrival station
            actual_dwell = max(sch_dwell, 1.0)
            dep_clock_mins = arr_clock_mins + actual_dwell
            dep_delay_mins = round(dep_clock_mins - sch_dep_time_mins, 1)
            
            # Confidence (accumulates network delay pressure monotonically)
            net_press = float(feat_mat[hop_idx, 25]) if num_features >= 34 else 0.0
            running_pressure = max(running_pressure, net_press)
            conf_val, conf_lvl = self.compute_confidence(
                hop_index=hop_idx,
                is_foggy=bool(sec.get('is_foggy', 0)),
                is_heavy_rain=bool(sec.get('is_heavy_rain', 0)),
                edge_ntrains=int(sec.get('edge_ntrains', 5)),
                net_pressure=running_pressure,
                canonical_state=canonical_state
            )
            
            # Explanations
            explanations = []
            if kinematic_reason:
                explanations.append(f"[LIVE_KINEMATICS] {kinematic_reason}")
            if net_press >= 15.0:
                explanations.append(f"[NETWORK_CONGESTION] Downstream delay pressure is {net_press:.1f}m")
            if rule_res.audit_trail:
                for log in rule_res.audit_trail:
                    explanations.append(f"[{log.rule_name}] {log.reason}")
            elif abs(final_sec_time - sch_sec_time) > 2.0:
                diff = final_sec_time - sch_sec_time
                dir_str = "added" if diff > 0 else "recovered"
                explanations.append(f"ML forecast: {abs(diff):.1f}m {dir_str} based on section historical density")
            else:
                if not kinematic_reason:
                    explanations.append("Normal section progression on timetable")
                
            eta_record = StationETA(
                station_code=to_stn,
                station_name=to_name,
                scheduled_arrival=self.minutes_to_ampm(sch_arr_time_mins),
                scheduled_departure=self.minutes_to_ampm(sch_dep_time_mins),
                predicted_arrival=self.minutes_to_ampm(arr_clock_mins),
                predicted_departure=self.minutes_to_ampm(dep_clock_mins),
                predicted_arr_delay_mins=arr_delay_mins,
                predicted_dep_delay_mins=dep_delay_mins,
                scheduled_section_time_mins=round(sch_sec_time, 1),
                predicted_section_time_mins=round(final_sec_time, 1),
                original_ml_time_mins=round(ml_time, 1),
                is_rule_adjusted=rule_res.is_adjusted or rule_res.is_clamped,
                confidence_pct=conf_val,
                confidence_level=conf_lvl,
                explanation="; ".join(explanations)
            )
            
            station_etas.append(eta_record)
            
            # Advance train state for next downstream section
            running_clock_mins = dep_clock_mins
            running_delay = dep_delay_mins

        return station_etas
