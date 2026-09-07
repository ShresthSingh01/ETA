"""
loader.py - Data Loading and Time Parsing Utilities for Indian Railways Data.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np


def parse_time_to_minutes(time_series: pd.Series) -> np.ndarray:
    """
    Vectorized parser converting 'HH:MM AM/PM' string series to integer minutes of day [0..1439].
    """
    s = time_series.astype(str)
    hours = s.str.slice(0, 2).astype(np.int32)
    minutes = s.str.slice(3, 5).astype(np.int32)
    ampm = s.str.slice(6, 8)

    hours = np.where((ampm == 'PM') & (hours != 12), hours + 12, hours)
    hours = np.where((ampm == 'AM') & (hours == 12), 0, hours)
    return hours * 60 + minutes


def compute_elapsed_minutes(t_from_mins: np.ndarray, t_to_mins: np.ndarray) -> np.ndarray:
    """
    Computes elapsed minutes between two times of day, handling midnight crossover.
    Assumes consecutive station travel is < 24 hours (standard for railway sections).
    """
    diff = t_to_mins - t_from_mins
    return np.where(diff < 0, diff + 1440, diff)


def load_cleaned_stations(path: str = 'data/cleaned/stations_cleaned.csv') -> pd.DataFrame:
    """Loads cleaned station coordinates dataframe indexed by station_code."""
    return pd.read_csv(path).set_index('station_code')


def load_cleaned_edges(path: str = 'data/cleaned/edges_cleaned.csv') -> pd.DataFrame:
    """Loads cleaned network edges dataframe with composite index (from, to)."""
    edges = pd.read_csv(path)
    edges['key'] = edges['from'] + '_' + edges['to']
    return edges.set_index('key')


def load_zone_mapping(path: str = 'Indian-Railway-Network-and-Delays/stations_zones_mapping.json') -> dict:
    """Loads dictionary mapping station_code -> railway zone string."""
    with open(path, 'r', encoding='utf-8') as f:
        zones = json.load(f)
    # Patch known unmapped stations
    zones.setdefault('TMA', 'ECR')
    return zones


def load_raw_delays(path: str = 'Indian-Railway-Network-and-Delays/train_routes_delays_Sep2024.csv') -> pd.DataFrame:
    """Loads raw delays CSV."""
    return pd.read_csv(path)


def load_raw_routes(path: str = 'Indian-Railway-Network-and-Delays/train_routes_Sep2024.csv') -> pd.DataFrame:
    """Loads raw scheduled train routes timetable."""
    return pd.read_csv(path)
