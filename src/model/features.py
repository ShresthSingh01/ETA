"""
features.py - Feature engineering and dataset preparation for baseline and LightGBM models.
"""

from typing import Tuple, List
import pandas as pd
import numpy as np


NUMERIC_FEATURES = [
    'scheduled_section_time',
    'distance_km',
    'dep_delay_from',
    'arr_delay_from',
    'scheduled_dwell_from',
    'section_median_time',
    'section_mean_time',
    'section_p90_time',
    'section_min_time',
    'section_std_time',
    'edge_ntrains',
    'hour_of_day',
    'day_of_week',
    'is_weekend',
    'day_of_month',
    'temperature_2m',
    'precipitation',
    'weather_code',
    'wind_speed_10m',
    'visibility',
    'is_foggy',
    'is_heavy_rain'
]

# M1 features: Basic downstream network state (4 features)
NETWORK_M1_FEATURES = [
    'net_1hop_mean_delay',
    'net_1hop_delayed_count',
    'net_1hop_active_count',
    'net_downstream_weighted_delay'
]

# M2 additional features: Multi-hop spatial granularity & trend (4 features)
NETWORK_M2_FEATURES = [
    'net_2hop_mean_delay',
    'net_2hop_delayed_count',
    'net_3hop_mean_delay',
    'net_downstream_delay_trend'
]

# M3 additional features: Temporal rolling patterns (3 features)
NETWORK_M3_FEATURES = [
    'recent_station_mean_delay',
    'rolling_station_mean_delay_6h',
    'station_delay_trend_2h'
]

NETWORK_STATE_FEATURES = NETWORK_M1_FEATURES + NETWORK_M2_FEATURES + NETWORK_M3_FEATURES

CATEGORICAL_FEATURES = [
    'zone'
]

TARGET_COL = 'actual_section_time'


def load_dataset_splits(
    parquet_path: str = 'data/processed/section_runs_weather.parquet'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads dataset and splits into train (Sep 1-22), val (Sep 23-26), and test (Sep 27-30)."""
    df = pd.read_parquet(parquet_path)
    
    # Ensure categoricals are category dtype
    for cat_col in CATEGORICAL_FEATURES:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype('category')
            
    train_df = df[df['split'] == 'train'].copy().reset_index(drop=True)
    val_df = df[df['split'] == 'val'].copy().reset_index(drop=True)
    test_df = df[df['split'] == 'test'].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df


def get_feature_names(
    include_weather: bool = True,
    include_network: bool = True,
    include_network_state: bool = True,
    model_tier: str = 'M3'
) -> List[str]:
    """
    Return list of active feature names with optional ablation toggles.
    model_tier:
      - 'M0': Baseline 23 features (no downstream network state)
      - 'M1': M0 + Basic downstream state (27 features)
      - 'M2': M1 + Multi-hop spatial granularity & trend (31 features)
      - 'M3': Full RSTGCN-inspired downstream state + rolling temporal patterns (34 features)
    """
    features = list(NUMERIC_FEATURES)
    if not include_weather:
        weather_cols = ['temperature_2m', 'precipitation', 'weather_code', 'wind_speed_10m', 'visibility', 'is_foggy', 'is_heavy_rain']
        features = [f for f in features if f not in weather_cols]
    if not include_network:
        net_cols = ['edge_ntrains']
        features = [f for f in features if f not in net_cols]

    # Add network state features based on tier
    if include_network_state and model_tier.upper() != 'M0':
        tier = model_tier.upper()
        if tier == 'M1':
            features.extend(NETWORK_M1_FEATURES)
        elif tier == 'M2':
            features.extend(NETWORK_M1_FEATURES + NETWORK_M2_FEATURES)
        elif tier == 'M3':
            features.extend(NETWORK_STATE_FEATURES)

    return features + CATEGORICAL_FEATURES
