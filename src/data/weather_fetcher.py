"""
weather_fetcher.py - Real Historical Weather Fetcher & Spatial Mapper using Open-Meteo ERA5 Reanalysis API.
"""

import time
from pathlib import Path
import json
import requests
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

from src.data.loader import load_cleaned_stations


OPEN_METEO_ARCHIVE_URL = 'https://archive-api.open-meteo.com/v1/archive'


def get_target_stations(
    section_runs_path: str = 'data/processed/section_runs.parquet',
    demo_trains: list = [12303, 12951, 12801, 12626],
    top_n_junctions: int = 80
) -> list:
    """Identify key stations covering all demo train routes and major Indian Railway junction hubs."""
    df = pd.read_parquet(section_runs_path, columns=['from_station', 'train_number'])
    demo_stns = set(df[df['train_number'].isin(demo_trains)]['from_station'].unique())
    top_stns = set(df['from_station'].value_counts().head(top_n_junctions).index)
    return sorted(list(demo_stns.union(top_stns)))


def fetch_station_weather(
    station_code: str,
    lat: float,
    lon: float,
    cache_dir: Path,
    start_date: str = '2024-09-01',
    end_date: str = '2024-09-30'
) -> dict:
    """Fetch hourly weather for a single station from Open-Meteo (with local caching)."""
    cache_file = cache_dir / f"{station_code}.json"
    if cache_file.exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    params = {
        'latitude': round(lat, 4),
        'longitude': round(lon, 4),
        'start_date': start_date,
        'end_date': end_date,
        'hourly': 'temperature_2m,precipitation,weather_code,wind_speed_10m,visibility'
    }
    
    try:
        resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=12)
        if resp.status_code == 200:
            data = resp.json()
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            return data
        else:
            print(f"Warning: Open-Meteo returned status {resp.status_code} for {station_code}")
            return {}
    except Exception as e:
        print(f"Error fetching weather for {station_code}: {e}")
        return {}


def build_weather_dataset(
    output_parquet: str = 'data/processed/station_weather.parquet',
    cache_dir_path: str = 'data/raw/weather_cache'
) -> pd.DataFrame:
    """Fetches and compiles hourly weather records for target railway stations."""
    stations = load_cleaned_stations()
    target_stns = get_target_stations()
    cache_dir = Path(cache_dir_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Targeting {len(target_stns)} key railway stations for historical weather ingestion...")
    
    records = []
    fetched_count = 0
    
    for i, stn in enumerate(target_stns):
        if stn not in stations.index:
            continue
        row = stations.loc[stn]
        lat, lon = row['latitude'], row['longitude']
        if pd.isna(lat) or pd.isna(lon):
            continue
            
        data = fetch_station_weather(stn, lat, lon, cache_dir)
        if 'hourly' in data and 'time' in data['hourly']:
            hourly = data['hourly']
            times = hourly['time']
            temps = hourly.get('temperature_2m', [np.nan] * len(times))
            precips = hourly.get('precipitation', [np.nan] * len(times))
            wcodes = hourly.get('weather_code', [0] * len(times))
            winds = hourly.get('wind_speed_10m', [np.nan] * len(times))
            visibs = hourly.get('visibility', [np.nan] * len(times))
            
            for t, temp, precip, wc, wind, vis in zip(times, temps, precips, wcodes, winds, visibs):
                # t format: '2024-09-01T00:00'
                date_str, time_str = t.split('T')
                hour_int = int(time_str.split(':')[0])
                records.append({
                    'weather_station_code': stn,
                    'date': date_str,
                    'hour': hour_int,
                    'temperature_2m': temp,
                    'precipitation': precip,
                    'weather_code': wc,
                    'wind_speed_10m': wind,
                    'visibility': vis
                })
            fetched_count += 1
            if (i + 1) % 25 == 0 or (i + 1) == len(target_stns):
                print(f"Processed {i + 1}/{len(target_stns)} stations ({len(records):,} hourly rows)...")
        time.sleep(0.05)  # respectful gentle throttle
        
    df_weather = pd.DataFrame(records)
    print(f"Successfully compiled {len(df_weather):,} hourly weather observations across {fetched_count} stations.")
    
    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    df_weather.to_parquet(output_parquet, index=False, engine='pyarrow', compression='snappy')
    print(f"Saved weather dataset to {output_parquet}")
    return df_weather


def attach_weather_to_sections(
    sections_parquet: str = 'data/processed/section_runs.parquet',
    weather_parquet: str = 'data/processed/station_weather.parquet',
    stations_path: str = 'data/cleaned/stations_cleaned.csv',
    output_parquet: str = 'data/processed/section_runs_weather.parquet'
) -> pd.DataFrame:
    """
    Attaches hourly weather to section traversals using spatial KDTree to match each
    station to its nearest weather station, then joins on (weather_station, date, hour_of_day).
    """
    print("Loading section runs and weather tables...")
    sections = pd.read_parquet(sections_parquet)
    weather = pd.read_parquet(weather_parquet)
    stations = pd.read_csv(stations_path).set_index('station_code')
    
    # 1. Unique weather stations
    weather_stns = weather['weather_station_code'].unique()
    weather_coords = []
    valid_weather_stns = []
    for ws in weather_stns:
        if ws in stations.index:
            lat = stations.loc[ws, 'latitude']
            lon = stations.loc[ws, 'longitude']
            if not pd.isna(lat) and not pd.isna(lon):
                weather_coords.append([lat, lon])
                valid_weather_stns.append(ws)
                
    weather_coords = np.array(weather_coords)
    kdtree = cKDTree(weather_coords)
    
    # 2. Map every station in section_runs to its nearest weather station
    unique_from_stns = sections['from_station'].unique()
    station_to_weather_stn = {}
    
    for s in unique_from_stns:
        if s in valid_weather_stns:
            station_to_weather_stn[s] = s
        elif s in stations.index and not pd.isna(stations.loc[s, 'latitude']):
            lat = stations.loc[s, 'latitude']
            lon = stations.loc[s, 'longitude']
            _, idx = kdtree.query([lat, lon])
            station_to_weather_stn[s] = valid_weather_stns[idx]
        else:
            station_to_weather_stn[s] = valid_weather_stns[0]  # default fallback
            
    print(f"Mapped {len(unique_from_stns)} route stations to nearest weather hubs.")
    sections['weather_station_code'] = sections['from_station'].map(station_to_weather_stn)
    
    # 3. Join weather on ['weather_station_code', 'date', 'hour_of_day' -> 'hour']
    print("Joining hourly weather features onto section runs...")
    sections = sections.merge(
        weather,
        left_on=['weather_station_code', 'date', 'hour_of_day'],
        right_on=['weather_station_code', 'date', 'hour'],
        how='left'
    )
    
    # Drop redundant column
    if 'hour' in sections.columns:
        sections = sections.drop(columns=['hour'])
        
    # Fill any null weather features with global medians
    sections['temperature_2m'] = sections['temperature_2m'].fillna(28.0)
    sections['precipitation'] = sections['precipitation'].fillna(0.0)
    sections['weather_code'] = sections['weather_code'].fillna(0).astype(np.int32)
    sections['wind_speed_10m'] = sections['wind_speed_10m'].fillna(10.0)
    sections['visibility'] = sections['visibility'].fillna(10000.0)
    
    # Add engineered weather flags:
    # weather_code >= 45: Fog / depositing rime fog (WMO standard)
    sections['is_foggy'] = (sections['weather_code'].isin([45, 48])).astype(np.int32)
    # precipitation > 5.0 mm: Heavy rain
    sections['is_heavy_rain'] = (sections['precipitation'] >= 5.0).astype(np.int32)
    
    print(f"Saving enriched section runs with weather to {output_parquet}...")
    sections.to_parquet(output_parquet, index=False, engine='pyarrow', compression='snappy')
    print(f"Successfully created {output_parquet} with {len(sections):,} records.")
    return sections


if __name__ == '__main__':
    print("Starting Phase 2: Weather Ingestion from Open-Meteo...")
    build_weather_dataset()
    print("Attaching weather features to canonical section runs...")
    attach_weather_to_sections()
    print("Phase 2 Weather Integration complete.")
