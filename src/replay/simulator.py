"""
simulator.py - Historical Journey Replay Engine with Multi-Train Support & Event Injection.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

from src.engine.eta_calculator import ETACalculator, StationETA
from src.engine.rule_engine import OperationalEvent
from src.data.loader import load_cleaned_stations
from src.integrations.base import TrainStateProvider, CanonicalTrainState, StationBoardEntry
from src.integrations.replay_provider import ReplayProvider
from src.integrations.railradar import RailRadarProvider


DEMO_TRAINS_CONFIG = [
    {
        'train_number': 12303,
        'train_name': 'Poorva Express',
        'route_desc': 'Howrah Jn (HWH) → New Delhi (NDLS)',
        'category': 'Superfast Trunk Corridor',
        'default_date': '2024-09-28'
    },
    {
        'train_number': 12951,
        'train_name': 'Mumbai Tejas Rajdhani',
        'route_desc': 'Mumbai Central (MMCT) → New Delhi (NDLS)',
        'category': 'Premium High-Speed Corridor',
        'default_date': '2024-09-28'
    },
    {
        'train_number': 12801,
        'train_name': 'Purushottam Express',
        'route_desc': 'Puri (PURI) → New Delhi (NDLS)',
        'category': 'Long-Haul Inter-Zone Express',
        'default_date': '2024-09-28'
    },
    {
        'train_number': 12626,
        'train_name': 'Kerala Express',
        'route_desc': 'New Delhi (NDLS) → Thiruvananthapuram (TVC)',
        'category': 'Pan-India Cross-Country Trunk',
        'default_date': '2024-09-28'
    }
]

STATION_ALIASES: Dict[str, str] = {
    'MMCT': 'BCT',
    'BCT': 'BCT',
    'NDLS': 'NDLS',
    'HWH': 'HWH',
    'PURI': 'PURI',
    'TVC': 'TVC',
    'CSMT': 'CSTM',
    'CSTM': 'CSTM',
    'MAS': 'MAS'
}

KNOWN_STATION_COORDS: Dict[str, Tuple[float, float, str]] = {
    'MMCT': (18.970667, 72.819383, "Mumbai Central"),
    'BCT': (18.970667, 72.819383, "Mumbai Central"),
    'BVI': (19.228739, 72.856412, "Borivali"),
    'ST': (21.206568, 72.840793, "Surat"),
    'BRC': (22.310756, 73.181065, "Vadodara Jn"),
    'RTM': (23.340380, 75.050826, "Ratlam Jn"),
    'NAD': (23.455920, 75.412491, "Nagda Jn"),
    'KOTA': (25.223553, 75.880500, "Kota Jn"),
    'NDLS': (28.642314, 77.220004, "New Delhi"),
    'HWH': (22.589200, 88.343500, "Howrah Jn"),
    'PURI': (19.813500, 85.831200, "Puri"),
    'TVC': (8.487500, 76.952500, "Thiruvananthapuram")
}


class ReplaySimulator:
    def __init__(
        self,
        section_runs_path: str = 'data/processed/section_runs_weather.parquet',
        stations_path: str = 'data/cleaned/stations_cleaned.csv',
        model_path: str = 'models/lightgbm_eta.txt'
    ):
        print("Initializing ReplaySimulator...")
        self.df_all = pd.read_parquet(section_runs_path)
        self.stations = load_cleaned_stations(stations_path)
        self.calculator = ETACalculator(model_path=model_path)
        
        # Dual-Provider Architecture
        self.mode: str = "historical_replay"
        self.replay_provider = ReplayProvider(simulator_ref=self)
        self.railradar_provider = RailRadarProvider()
        self.active_provider: TrainStateProvider = self.replay_provider

        self.active_events: List[OperationalEvent] = []
        self.current_train = 12303
        self.current_date = '2024-09-28'
        self.current_step = 0
        self.journey_sections: List[Dict[str, Any]] = []
        self.stations_route: List[Dict[str, Any]] = []
        
        self.load_journey(self.current_train, self.current_date)

    def _resolve_station_info(self, code: str) -> Tuple[str, float, float]:
        """Resolves station name, latitude, longitude with alias and fallback lookup."""
        clean_code = str(code).upper().strip()
        alias_code = STATION_ALIASES.get(clean_code, clean_code)
        
        # 1. Direct match in stations dataset
        if clean_code in self.stations.index:
            s_row = self.stations.loc[clean_code]
            return s_row.get('station_name', clean_code), float(s_row.get('latitude', 28.6139)), float(s_row.get('longitude', 77.2090))
            
        # 2. Alias match in stations dataset
        if alias_code in self.stations.index:
            s_row = self.stations.loc[alias_code]
            return s_row.get('station_name', clean_code), float(s_row.get('latitude', 28.6139)), float(s_row.get('longitude', 77.2090))
            
        # 3. Known coordinates fallback
        if clean_code in KNOWN_STATION_COORDS:
            lat, lon, name = KNOWN_STATION_COORDS[clean_code]
            return name, lat, lon
        if alias_code in KNOWN_STATION_COORDS:
            lat, lon, name = KNOWN_STATION_COORDS[alias_code]
            return name, lat, lon
            
        return clean_code, 28.6139, 77.2090

    def load_journey(self, train_number: int, date: str):
        """Loads all sections and station stops for a specific train and date."""
        self.current_train = train_number
        self.current_date = date
        self.current_step = 0
        self.active_events = []
        
        # Filter sections for this journey
        sub = self.df_all[(self.df_all['train_number'] == train_number) & (self.df_all['date'] == date)].copy()
        if len(sub) == 0:
            # Fallback to date with sections for this train
            sub_all = self.df_all[self.df_all['train_number'] == train_number]
            if len(sub_all) > 0:
                fallback_date = sub_all['date'].iloc[0]
                sub = self.df_all[(self.df_all['train_number'] == train_number) & (self.df_all['date'] == fallback_date)].copy()
                self.current_date = fallback_date
            else:
                # Fallback to default corridor 12303 if train not in historical dataset
                sub_default = self.df_all[self.df_all['train_number'] == 12303]
                fallback_date = sub_default['date'].iloc[0]
                sub = self.df_all[(self.df_all['train_number'] == 12303) & (self.df_all['date'] == fallback_date)].copy()
                self.current_date = fallback_date
            
        sub = sub.reset_index(drop=True)
        self.journey_sections = sub.to_dict(orient='records')
        
        # Build station list with coordinates
        route_stns = []
        for i, row in enumerate(self.journey_sections):
            from_code = str(row['from_station']).strip().upper()
            if i == 0:
                stn_name, lat, lon = self._resolve_station_info(from_code)
                route_stns.append({
                    'station_code': from_code,
                    'station_name': stn_name,
                    'latitude': lat,
                    'longitude': lon,
                    'is_origin': True,
                    'is_destination': False
                })
                
            to_code = str(row['to_station']).strip().upper()
            stn_name, lat, lon = self._resolve_station_info(to_code)
            route_stns.append({
                'station_code': to_code,
                'station_name': stn_name,
                'latitude': lat,
                'longitude': lon,
                'is_origin': False,
                'is_destination': (i == len(self.journey_sections) - 1)
            })
            
        self.stations_route = route_stns
        print(f"Loaded Train {train_number} on {self.current_date}: {len(self.journey_sections)} sections, {len(self.stations_route)} stations.")

    def set_mode(self, mode: str) -> str:
        """Switch between 'historical_replay' and 'live_external'."""
        if mode not in ("historical_replay", "live_external"):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'historical_replay' or 'live_external'.")
        self.mode = mode
        if mode == "live_external":
            self.active_provider = self.railradar_provider
        else:
            self.active_provider = self.replay_provider
        return self.mode

    def get_provider_health(self) -> Dict[str, Any]:
        """Returns telemetry health from the active provider."""
        return self.active_provider.get_health()

    def get_station_board(self, station_code: str) -> List[Dict[str, Any]]:
        """Returns live arrival/departure board for downstream station traffic."""
        entries = self.active_provider.get_station_board(station_code)
        return [e.to_dict() for e in entries]

    def inject_event(self, event: OperationalEvent):
        """Adds an active operational event."""
        # Replace if same from-to already has an event
        self.active_events = [e for e in self.active_events if not (e.from_station == event.from_station and e.to_station == event.to_station)]
        self.active_events.append(event)

    def clear_events(self):
        """Clears all active operational events."""
        self.active_events = []

    def get_state(self, step: Optional[int] = None) -> Dict[str, Any]:
        """
        Returns full state of journey at requested step (or current_step).
        Calculates real-time comparisons between:
        - Scheduled
        - Naive NTES (Current Delay added to schedule)
        - Historical Median Baseline
        - Our Model + Rule Engine (with Live Kinematics in Live Mode)
        - Actual Ground Truth Recorded
        """
        canonical_state: Optional[CanonicalTrainState] = None

        if self.mode == "live_external":
            canonical_state = self.railradar_provider.get_live_state(
                str(self.current_train),
                self.current_date
            )
            # Match live station position to journey route if step wasn't manually overridden
            if step is None and canonical_state:
                code = canonical_state.current_station_code.strip().upper()
                alias = STATION_ALIASES.get(code, code)
                for i, stn in enumerate(self.stations_route):
                    stn_c = stn['station_code'].strip().upper()
                    if stn_c == code or stn_c == alias:
                        self.current_step = i
                        break
        else:
            canonical_state = self.replay_provider.get_live_state(
                str(self.current_train),
                self.current_date
            )

        if step is not None:
            self.current_step = max(0, min(step, len(self.journey_sections)))
            
        k = self.current_step
        total_sections = len(self.journey_sections)
        
        # Current station position
        if k == 0:
            current_stn_info = self.stations_route[0]
            origin_hour = float(self.journey_sections[0]['hour_of_day']) if self.journey_sections else 8.0
            current_clock_mins = origin_hour * 60.0
            current_delay = 0.0
            if len(self.journey_sections) > 0:
                current_delay = float(self.journey_sections[0]['dep_delay_from'])
        else:
            current_sec = self.journey_sections[k - 1]
            current_stn_info = self.stations_route[min(k, len(self.stations_route) - 1)]
            current_delay = float(current_sec['dep_delay_to'])
            origin_hour = float(self.journey_sections[0]['hour_of_day']) if self.journey_sections else 8.0
            # Cumulative elapsed actual travel and dwell time up to step k
            elapsed_mins = sum(
                float(self.journey_sections[j]['actual_section_time']) +
                float(self.journey_sections[j].get('actual_dwell_from', 2.0))
                for j in range(k)
            )
            current_clock_mins = origin_hour * 60.0 + elapsed_mins

        # If in live mode, override current delay with live observed delay
        if self.mode == "live_external" and canonical_state:
            current_delay = canonical_state.current_delay_min

        # Inject real-time station delay observation into dynamic network state engine
        curr_code = str(current_stn_info.get('station_code', '')).strip().upper()
        if curr_code:
            self.calculator.network_engine.update_live_delay(curr_code, current_delay)
            
        remaining = self.journey_sections[k:]
        
        # Predict remaining ETAs using Our Model + Rules + Live Kinematics
        model_etas = self.calculator.predict_journey_etas(
            train_number=self.current_train,
            current_station=current_stn_info['station_code'],
            current_clock_mins=current_clock_mins,
            current_dep_delay_mins=current_delay,
            remaining_sections=remaining,
            active_events=self.active_events,
            canonical_state=canonical_state
        )
        
        # Build comprehensive station-by-station table
        comparison_table = []
        cumulative_median = current_clock_mins
        cumulative_sch = 0.0
        
        for idx, sec in enumerate(remaining):
            stn_code = sec['to_station']
            stn_name = sec.get('to_station_name', stn_code)
            if stn_code in self.stations.index:
                stn_name = self.stations.loc[stn_code].get('station_name', stn_code)
                
            dist_km = float(sec['distance_km'])
            sch_sec_time = float(sec['scheduled_section_time'])
            act_sec_time = float(sec['actual_section_time'])
            cumulative_sch += sch_sec_time
            
            # Ground truth actual delay at this station
            actual_arr_delay = float(sec['arr_delay_to'])
            
            # 1. Naive NTES: assumes current delay persists forever
            naive_predicted_delay = current_delay
            naive_clock = current_clock_mins + cumulative_sch + current_delay
            
            # 2. Historical median baseline
            hist_median_sec = float(sec['section_median_time'])
            cumulative_median += hist_median_sec
            median_predicted_delay = round(current_delay + (hist_median_sec - sch_sec_time), 1)
            
            # 3. Our Model
            eta_obj = model_etas[idx]
            our_predicted_delay = eta_obj.predicted_arr_delay_mins
            
            # Error deltas relative to actual ground truth
            error_our_model = abs(our_predicted_delay - actual_arr_delay)
            error_naive = abs(naive_predicted_delay - actual_arr_delay)
            
            comparison_table.append({
                'hop': idx + 1,
                'station_code': stn_code,
                'station_name': stn_name,
                'distance_km': round(dist_km, 1),
                'scheduled_arr': eta_obj.scheduled_arrival,
                'naive_ntes_eta': self.calculator.minutes_to_ampm(naive_clock),
                'naive_predicted_delay': round(naive_predicted_delay, 1),
                'median_baseline_delay': round(median_predicted_delay, 1),
                'our_predicted_eta': eta_obj.predicted_arrival,
                'our_predicted_delay': our_predicted_delay,
                'actual_ground_truth_delay': actual_arr_delay,
                'error_our_model_mins': round(error_our_model, 1),
                'error_naive_mins': round(error_naive, 1),
                'is_better_than_naive': (error_our_model <= error_naive),
                'is_rule_adjusted': eta_obj.is_rule_adjusted,
                'confidence_pct': eta_obj.confidence_pct,
                'confidence_level': eta_obj.confidence_level,
                'explanation': eta_obj.explanation
            })
            
        # Summary statistics
        avg_our_err = np.mean([r['error_our_model_mins'] for r in comparison_table]) if comparison_table else 0.0
        avg_naive_err = np.mean([r['error_naive_mins'] for r in comparison_table]) if comparison_table else 0.0
        
        # Real-time closed-loop telemetry trace
        source_label = "RailRadar Live API Stream" if (self.mode == "live_external" and canonical_state and canonical_state.status != "UNAVAILABLE") else "NTES Official Movement Telemetry (Sep 2024)"
        next_hop = comparison_table[0] if comparison_table else None
        
        closed_loop_trace = {
            'latest_observation': {
                'station_code': curr_code,
                'station_name': current_stn_info.get('station_name', curr_code),
                'observed_delay_mins': round(current_delay, 1),
                'observed_clock': self.calculator.minutes_to_ampm(current_clock_mins),
                'source': source_label,
                'is_synthetic': False
            },
            'dynamic_network_update': {
                'station_code': curr_code,
                'injected_delay_mins': round(current_delay, 1),
                'downstream_pressure_mins': round(float(next_hop.get('our_predicted_delay', 0.0) if next_hop else 0.0), 1),
                'active_network_stations': len(self.calculator.network_engine.live_station_delay)
            },
            'recomputed_next_eta': {
                'next_station_code': next_hop['station_code'] if next_hop else 'DEST',
                'next_station_name': next_hop['station_name'] if next_hop else 'Destination',
                'predicted_eta': next_hop['our_predicted_eta'] if next_hop else '--',
                'predicted_delay_mins': next_hop['our_predicted_delay'] if next_hop else 0.0,
                'confidence_pct': next_hop['confidence_pct'] if next_hop else 95.0
            }
        }

        return {
            'mode': self.mode,
            'train_number': self.current_train,
            'date': self.current_date,
            'current_step': k,
            'total_steps': total_sections,
            'is_finished': (k >= total_sections),
            'current_station': current_stn_info,
            'current_delay_mins': round(current_delay, 1),
            'canonical_state': canonical_state.to_dict() if canonical_state else None,
            'provider_health': self.get_provider_health(),
            'stations_route': self.stations_route,
            'active_events': [asdict(e) for e in self.active_events],
            'comparison_table': comparison_table,
            'closed_loop_trace': closed_loop_trace,
            'summary': {
                'avg_error_our_model_mins': round(float(avg_our_err), 2),
                'avg_error_naive_mins': round(float(avg_naive_err), 2),
                'improvement_over_naive_mins': round(float(avg_naive_err - avg_our_err), 2),
                'remaining_stations_count': len(comparison_table)
            }
        }
