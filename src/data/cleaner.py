"""
cleaner.py - Railway Network & Stations Data Cleaning Module.

Cleans IRN edges (removes spurious negative/zero distance edges),
computes geodetic distances for halt sections, and cleans station coordinates.
"""

import math
from pathlib import Path
import pandas as pd
import numpy as np


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate Great Circle distance in kilometers between two lat/lon points."""
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return 0.0
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(R * c, 2)


def clean_edges(raw_edges_path: str, stations_path: str, output_path: str) -> dict:
    """
    Clean IRN network edges:
    1. Removes spurious negative distance edges (e.g. SNNR -> YPR = -624).
    2. Replaces zero distances with physical geodetic distances calculated from station coordinates.
    3. Verifies no self-loops and no duplicate directed edges.
    """
    edges = pd.read_csv(raw_edges_path)
    stations = pd.read_csv(stations_path).set_index('station_code')
    
    initial_count = len(edges)
    
    # 1. Identify and remove negative distance edges
    neg_mask = edges['distance'] < 0
    neg_edges = edges[neg_mask].copy()
    edges = edges[~neg_mask].copy()
    edges['distance'] = edges['distance'].astype(float)
    
    # 2. Fix zero distance edges with geodetic distances
    zero_mask = edges['distance'] == 0
    zero_count = int(zero_mask.sum())
    
    for idx in edges[zero_mask].index:
        from_stn = edges.loc[idx, 'from']
        to_stn = edges.loc[idx, 'to']
        dist = 1.0  # default minimum 1 km
        if from_stn in stations.index and to_stn in stations.index:
            lat1 = stations.loc[from_stn, 'latitude']
            lon1 = stations.loc[from_stn, 'longitude']
            lat2 = stations.loc[to_stn, 'latitude']
            lon2 = stations.loc[to_stn, 'longitude']
            geo_dist = haversine_distance(lat1, lon1, lat2, lon2)
            if geo_dist > 0:
                dist = max(geo_dist, 0.5)
        edges.loc[idx, 'distance'] = dist
    
    # 3. Check for duplicates and self loops
    self_loops = int((edges['from'] == edges['to']).sum())
    edges = edges[edges['from'] != edges['to']]
    
    edges = edges.drop_duplicates(subset=['from', 'to'])
    
    # Ensure distance is positive float
    edges['distance'] = edges['distance'].astype(float)
    edges['ntrains'] = edges['ntrains'].astype(int)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(output_path, index=False)
    
    stats = {
        'initial_edges': initial_count,
        'final_edges': len(edges),
        'negative_edges_removed': int(neg_mask.sum()),
        'zero_edges_repaired': zero_count,
        'self_loops_removed': self_loops,
        'min_distance': float(edges['distance'].min()),
        'max_distance': float(edges['distance'].max()),
        'avg_distance': float(edges['distance'].mean()),
    }
    return stats


def clean_stations(raw_stations_path: str, output_path: str) -> dict:
    """
    Clean station coordinates:
    1. Strips whitespace from station codes and names.
    2. Filters or flags invalid coordinates (lat outside 6-38, lon outside 68-98 for India).
    3. Saves cleaned stations table.
    """
    stations = pd.read_csv(raw_stations_path)
    initial_count = len(stations)
    
    stations['station_code'] = stations['station_code'].astype(str).str.strip().str.upper()
    stations['station_name'] = stations['station_name'].astype(str).str.strip()
    
    # Drop rows without station code
    stations = stations[stations['station_code'].str.len() > 0]
    
    # Valid bounding box for Indian subcontinent: Lat ~6° to 38°N, Lon ~68° to 98°E
    valid_coords = (
        (stations['latitude'] >= 6.0) & (stations['latitude'] <= 38.0) &
        (stations['longitude'] >= 68.0) & (stations['longitude'] <= 98.0)
    )
    
    valid_count = int(valid_coords.sum())
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    stations.to_csv(output_path, index=False)
    
    stats = {
        'initial_stations': initial_count,
        'valid_coordinate_stations': valid_count,
        'missing_or_out_of_bounds': initial_count - valid_count,
        'unique_station_codes': int(stations['station_code'].nunique())
    }
    return stats


def generate_validation_report(edge_stats: dict, station_stats: dict, report_path: str):
    """Write markdown summary report of data cleaning."""
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# IRN Data Validation & Cleaning Report\n\n")
        f.write("Generated automatically during Phase 1: Data Foundation.\n\n")
        f.write("## 1. Network Graph Edges (`IRN_edges.csv`)\n\n")
        f.write(f"- **Initial Edge Count**: {edge_stats['initial_edges']:,}\n")
        f.write(f"- **Final Cleaned Edge Count**: {edge_stats['final_edges']:,}\n")
        f.write(f"- **Negative Distance Edges Removed**: {edge_stats['negative_edges_removed']} (e.g., SNNR->YPR -624km route artifact)\n")
        f.write(f"- **Zero Distance Edges Repaired**: {edge_stats['zero_edges_repaired']} (geodetic distances imputed from station coordinates)\n")
        f.write(f"- **Self Loops Removed**: {edge_stats['self_loops_removed']}\n")
        f.write(f"- **Min Distance**: {edge_stats['min_distance']:.2f} km\n")
        f.write(f"- **Max Distance**: {edge_stats['max_distance']:.2f} km\n")
        f.write(f"- **Avg Distance**: {edge_stats['avg_distance']:.2f} km\n\n")
        f.write("## 2. Station Coordinates (`india_railway_stations.csv`)\n\n")
        f.write(f"- **Total Stations**: {station_stats['initial_stations']:,}\n")
        f.write(f"- **Valid Geospatial Coordinates**: {station_stats['valid_coordinate_stations']:,}\n")
        f.write(f"- **Missing / Out-of-bounds**: {station_stats['missing_or_out_of_bounds']:,}\n")
        f.write(f"- **Unique Station Codes**: {station_stats['unique_station_codes']:,}\n")


if __name__ == '__main__':
    raw_edges = 'Indian-Railway-Network-and-Delays/IRN_edges.csv'
    raw_stations = 'Indian-Railway-Network-and-Delays/india_railway_stations.csv'
    out_edges = 'data/cleaned/edges_cleaned.csv'
    out_stations = 'data/cleaned/stations_cleaned.csv'
    report = 'data/cleaned/validation_report.md'
    
    print("Cleaning edges...")
    e_stats = clean_edges(raw_edges, raw_stations, out_edges)
    print("Edges cleaned:", e_stats)
    
    print("Cleaning stations...")
    s_stats = clean_stations(raw_stations, out_stations)
    print("Stations cleaned:", s_stats)
    
    generate_validation_report(e_stats, s_stats, report)
    print(f"Validation report saved to {report}")
