"""
network_state.py - Downstream Network State Engine for Spatial-Temporal Delay Propagation.

Inspired by RSTGCN (Residual Spatial-Temporal Graph Convolutional Network) principles:
1. Spatial Downstream Propagation: Evaluates network pressure and congestion ahead of the train
   at 1-hop, 2-hop, and 3-hop downstream stations.
2. Temporal Rolling Dynamics: Computes recent 2-hour, 6-hour, and rate-of-change delay trends
   at downstream stations to capture escalating vs. dissipating congestion.
3. Zero-Leakage: Strictly uses observations strictly prior to current section traversal (H-1).
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


NETWORK_STATE_FEATURE_NAMES = [
    'net_1hop_mean_delay',
    'net_1hop_delayed_count',
    'net_1hop_active_count',
    'net_downstream_weighted_delay',
    'net_2hop_mean_delay',
    'net_2hop_delayed_count',
    'net_3hop_mean_delay',
    'net_downstream_delay_trend',
    'recent_station_mean_delay',
    'rolling_station_mean_delay_6h',
    'station_delay_trend_2h'
]


@dataclass
class DownstreamNetworkState:
    net_1hop_mean_delay: float = 0.0
    net_1hop_delayed_count: float = 0.0
    net_1hop_active_count: float = 0.0
    net_2hop_mean_delay: float = 0.0
    net_2hop_delayed_count: float = 0.0
    net_3hop_mean_delay: float = 0.0
    net_downstream_weighted_delay: float = 0.0
    net_downstream_delay_trend: float = 0.0
    recent_station_mean_delay: float = 0.0
    rolling_station_mean_delay_6h: float = 0.0
    station_delay_trend_2h: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.net_1hop_mean_delay,
            self.net_1hop_delayed_count,
            self.net_1hop_active_count,
            self.net_2hop_mean_delay,
            self.net_2hop_delayed_count,
            self.net_3hop_mean_delay,
            self.net_downstream_weighted_delay,
            self.net_downstream_delay_trend,
            self.recent_station_mean_delay,
            self.rolling_station_mean_delay_6h,
            self.station_delay_trend_2h
        ], dtype=np.float64)


class DownstreamNetworkStateEngine:
    """
    Computes spatial multi-hop and temporal rolling network features ahead of a train.
    Supports both offline grid lookups (for high-speed training/replay) and live stream state.
    """

    def __init__(
        self,
        grid_path: Optional[str] = 'data/processed/station_network_grid.npz',
        global_median_delay: float = 11.0
    ):
        self.grid_path = grid_path
        self.global_median_delay = global_median_delay
        self.stn_to_idx: Dict[str, int] = {}
        self.grid_delay: Optional[np.ndarray] = None
        self.grid_delayed: Optional[np.ndarray] = None
        self.grid_active: Optional[np.ndarray] = None
        self.live_station_delay: Dict[str, float] = {}

        if grid_path and Path(grid_path).exists():
            self.load_grid(grid_path)

    def load_grid(self, grid_path: str):
        """Loads precomputed station-hour network grid from .npz file."""
        data = np.load(grid_path, allow_pickle=True)
        self.grid_delay = data['grid_delay']
        self.grid_delayed = data['grid_delayed']
        self.grid_active = data['grid_active']
        stations = list(data['stations'])
        self.stn_to_idx = {s: i for i, s in enumerate(stations)}

    def save_grid(self, output_path: str, stations: List[str]):
        """Saves current grid to .npz file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            grid_delay=self.grid_delay,
            grid_delayed=self.grid_delayed,
            grid_active=self.grid_active,
            stations=np.array(stations)
        )

    @classmethod
    def build_from_dataframe(
        cls,
        df: pd.DataFrame,
        save_path: Optional[str] = 'data/processed/station_network_grid.npz'
    ) -> 'DownstreamNetworkStateEngine':
        """
        Builds the 2D dense spatial-temporal station grid directly from section_runs dataframe.
        Executes in < 2 seconds for 1.2M rows.
        """
        engine = cls(grid_path=None)
        
        date_dt = pd.to_datetime(df['date'])
        global_hour = ((date_dt.dt.day - 1) * 24 + df['hour_of_day']).to_numpy(dtype=np.int32)
        
        is_delayed = (df['dep_delay_from'] > 5).astype(np.int8)
        
        # Aggregate station-hourly metrics
        temp_df = pd.DataFrame({
            'station': df['from_station'].to_numpy(),
            'global_hour': global_hour,
            'dep_delay': df['dep_delay_from'].to_numpy(dtype=np.float32),
            'is_delayed': is_delayed
        })
        
        stn_hourly = temp_df.groupby(['station', 'global_hour']).agg(
            stn_active=('dep_delay', 'count'),
            stn_mean_delay=('dep_delay', 'mean'),
            stn_delayed=('is_delayed', 'sum')
        ).reset_index()

        stations = sorted(list(set(df['from_station'].unique()) | set(df['to_station'].unique())))
        engine.stn_to_idx = {s: i for i, s in enumerate(stations)}
        n_stns = len(stations)
        n_hours = 720  # 30 days * 24 hours

        engine.grid_delay = np.zeros((n_stns, n_hours), dtype=np.float32)
        engine.grid_delayed = np.zeros((n_stns, n_hours), dtype=np.float32)
        engine.grid_active = np.zeros((n_stns, n_hours), dtype=np.float32)

        stn_indices = stn_hourly['station'].map(engine.stn_to_idx).to_numpy()
        hour_indices = stn_hourly['global_hour'].to_numpy()

        engine.grid_delay[stn_indices, hour_indices] = stn_hourly['stn_mean_delay'].to_numpy()
        engine.grid_delayed[stn_indices, hour_indices] = stn_hourly['stn_delayed'].to_numpy()
        engine.grid_active[stn_indices, hour_indices] = stn_hourly['stn_active'].to_numpy()

        if save_path:
            engine.save_grid(save_path, stations)

        return engine

    def update_live_delay(self, station_code: str, delay_mins: float):
        """Updates real-time in-memory station delay state from live telemetry."""
        self.live_station_delay[station_code.upper()] = max(delay_mins, 0.0)

    def query_station_state(self, station_code: str, global_hour: int) -> Tuple[float, float, float]:
        """
        Returns (mean_delay, delayed_count, active_count) for a station at global_hour H-1.
        Falls back to live in-memory state or defaults if station is unobserved.
        """
        stn_code = str(station_code).upper()
        if stn_code in self.live_station_delay:
            d = self.live_station_delay[stn_code]
            return d, (1.0 if d > 5 else 0.0), 1.0

        if self.grid_delay is not None and stn_code in self.stn_to_idx:
            idx = self.stn_to_idx[stn_code]
            h_lag1 = max(global_hour - 1, 0)
            return (
                float(self.grid_delay[idx, h_lag1]),
                float(self.grid_delayed[idx, h_lag1]),
                float(self.grid_active[idx, h_lag1])
            )

        return 0.0, 0.0, 0.0

    def compute_journey_downstream_features(
        self,
        remaining_sections: List[Dict[str, Any]],
        current_hour_of_day: int,
        day_of_month: int = 15
    ) -> np.ndarray:
        """
        Vectorized computation of all 11 downstream network features for all remaining
        sections along a train journey trajectory.
        Returns array of shape (len(remaining_sections), 11).
        """
        n_sections = len(remaining_sections)
        out = np.zeros((n_sections, 11), dtype=np.float64)
        if n_sections == 0:
            return out

        global_hour = int((day_of_month - 1) * 24 + current_hour_of_day)
        h_lag1 = max(global_hour - 1, 0)
        h_lag2 = max(global_hour - 2, 0)
        lags_6h = [max(global_hour - i, 0) for i in range(1, 7)]

        for i in range(n_sections):
            stn_1hop = str(remaining_sections[i].get('to_station', '')).upper()
            stn_2hop = str(remaining_sections[i+1].get('to_station', stn_1hop)).upper() if i + 1 < n_sections else stn_1hop
            stn_3hop = str(remaining_sections[i+2].get('to_station', stn_2hop)).upper() if i + 2 < n_sections else stn_2hop

            # 1-hop
            d1, c1, a1 = self.query_station_state(stn_1hop, global_hour)
            # 2-hop
            d2, c2, _ = self.query_station_state(stn_2hop, global_hour)
            # 3-hop
            d3, _, _ = self.query_station_state(stn_3hop, global_hour)

            # Rolling temporal patterns at 1-hop station
            if self.grid_delay is not None and stn_1hop in self.stn_to_idx:
                idx1 = self.stn_to_idx[stn_1hop]
                d_h1 = float(self.grid_delay[idx1, h_lag1])
                d_h2 = float(self.grid_delay[idx1, h_lag2])
                if stn_1hop in self.live_station_delay:
                    d_live = self.live_station_delay[stn_1hop]
                    recent_mean = 0.5 * (d_live + d_h1)
                    trend_2h = d_live - d_h1
                else:
                    recent_mean = 0.5 * (d_h1 + d_h2)
                    trend_2h = d_h1 - d_h2
                roll_6h = float(np.mean([self.grid_delay[idx1, lh] for lh in lags_6h]))
            else:
                d_live = self.live_station_delay.get(stn_1hop, d1)
                recent_mean = d_live
                trend_2h = 0.0
                roll_6h = d_live

            weighted_delay = 0.5 * d1 + 0.3 * d2 + 0.2 * d3
            downstream_trend = d1 - d2

            out[i, 0] = d1
            out[i, 1] = c1
            out[i, 2] = a1
            out[i, 3] = weighted_delay
            out[i, 4] = d2
            out[i, 5] = c2
            out[i, 6] = d3
            out[i, 7] = downstream_trend
            out[i, 8] = recent_mean
            out[i, 9] = roll_6h
            out[i, 10] = trend_2h

        return out
