"""
test_data_quality.py - Quality & Sanity Assertions for Cleaned Network and Stations.
"""

from pathlib import Path
import pandas as pd
import pytest


def test_cleaned_edges_exist():
    assert Path('data/cleaned/edges_cleaned.csv').exists(), "Cleaned edges file must exist"


def test_edges_no_negative_distance():
    edges = pd.read_csv('data/cleaned/edges_cleaned.csv')
    assert (edges['distance'] <= 0).sum() == 0, "Cleaned edges must have strictly positive distances"


def test_edges_no_self_loops():
    edges = pd.read_csv('data/cleaned/edges_cleaned.csv')
    assert (edges['from'] == edges['to']).sum() == 0, "No self loops allowed in railway network graph"


def test_edges_no_duplicates():
    edges = pd.read_csv('data/cleaned/edges_cleaned.csv')
    assert edges.duplicated(subset=['from', 'to']).sum() == 0, "Edge (from, to) pairs must be unique"


def test_cleaned_stations_exist():
    assert Path('data/cleaned/stations_cleaned.csv').exists(), "Cleaned stations file must exist"


def test_stations_codes_unique():
    stations = pd.read_csv('data/cleaned/stations_cleaned.csv')
    assert stations['station_code'].nunique() == len(stations), "Station codes must be unique"


def test_stations_coordinates_bounds():
    stations = pd.read_csv('data/cleaned/stations_cleaned.csv')
    coords = stations.dropna(subset=['latitude', 'longitude'])
    assert len(coords) >= 8800, f"Expected at least 8800 stations with coordinates, got {len(coords)}"
    
    # Subcontinent bounds check
    assert coords['latitude'].between(6.0, 38.0).all(), "Latitudes must fall within India bounds [6, 38]"
    assert coords['longitude'].between(68.0, 98.0).all(), "Longitudes must fall within India bounds [68, 98]"
