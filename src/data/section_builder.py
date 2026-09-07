"""
section_builder.py - Builds the canonical section_runs table from raw datasets.

Extracts consecutive station section traversals for every train-date journey,
joins distance and network topology, applies zero-leakage historical aggregations,
and creates temporal train/val/test splits.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from src.data.loader import (
    parse_time_to_minutes,
    compute_elapsed_minutes,
    load_cleaned_stations,
    load_cleaned_edges,
    load_zone_mapping,
    load_raw_delays,
    load_raw_routes
)


def build_route_distance_lookup(routes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a train-specific section distance and trainName lookup table from scheduled timetable.
    """
    is_next = routes_df['trainNumber'] == routes_df['trainNumber'].shift(-1)
    sub = routes_df[is_next].copy()
    sub['to_station'] = routes_df['station_code'].shift(-1)[is_next]
    sub['dist_km'] = (routes_df['distance'].shift(-1)[is_next] - sub['distance']).astype(float)
    
    # Filter out anomalous negative step (e.g. repeated loop)
    sub = sub[sub['dist_km'] >= 0]
    
    lookup = pd.DataFrame({
        'key': sub['trainNumber'].astype(str) + '_' + sub['station_code'] + '_' + sub['to_station'],
        'train_name': sub['trainName'],
        'route_dist_km': sub['dist_km']
    }).drop_duplicates(subset=['key']).set_index('key')
    return lookup


def build_section_runs(
    delays_path: str = 'Indian-Railway-Network-and-Delays/train_routes_delays_Sep2024.csv',
    routes_path: str = 'Indian-Railway-Network-and-Delays/train_routes_Sep2024.csv',
    edges_path: str = 'data/cleaned/edges_cleaned.csv',
    stations_path: str = 'data/cleaned/stations_cleaned.csv',
    zones_path: str = 'Indian-Railway-Network-and-Delays/stations_zones_mapping.json',
    output_parquet: str = 'data/processed/section_runs.parquet'
) -> pd.DataFrame:
    """
    Constructs the canonical section_runs table.
    """
    print("Loading raw delays data...")
    delays = load_raw_delays(delays_path)
    
    print("Loading auxiliary timetable and network data...")
    routes = load_raw_routes(routes_path)
    edges = load_cleaned_edges(edges_path)
    stations = load_cleaned_stations(stations_path)
    zones = load_zone_mapping(zones_path)
    
    route_dist_lookup = build_route_distance_lookup(routes)
    
    # 1. Vectorized identify consecutive station pairs
    is_next = (delays['train'] == delays['train'].shift(-1)) & (delays['date'] == delays['date'].shift(-1))
    
    df_from = delays[is_next].copy().reset_index(drop=True)
    df_to = delays.shift(-1)[is_next].copy().reset_index(drop=True)
    
    print(f"Constructing {len(df_from):,} section traversals...")
    
    # 2. Parse times to minutes of day
    sch_dep_from_mins = parse_time_to_minutes(df_from['sch_dep'])
    sch_arr_from_mins = parse_time_to_minutes(df_from['sch_arr'])
    act_dep_from_mins = parse_time_to_minutes(df_from['act_dep'])
    act_arr_from_mins = parse_time_to_minutes(df_from['act_arr'])
    
    sch_arr_to_mins = parse_time_to_minutes(df_to['sch_arr'])
    sch_dep_to_mins = parse_time_to_minutes(df_to['sch_dep'])
    act_arr_to_mins = parse_time_to_minutes(df_to['act_arr'])
    act_dep_to_mins = parse_time_to_minutes(df_to['act_dep'])
    
    # 3. Scheduled and actual travel times
    scheduled_section_time = compute_elapsed_minutes(sch_dep_from_mins, sch_arr_to_mins).astype(float)
    
    # Ensure minimum scheduled time is positive
    scheduled_section_time = np.where(scheduled_section_time <= 0, 1.0, scheduled_section_time)
    
    arr_delay_from = df_from['arr_delay'].to_numpy(dtype=float)
    dep_delay_from = df_from['dep_delay'].to_numpy(dtype=float)
    arr_delay_to = df_to['arr_delay'].to_numpy(dtype=float)
    dep_delay_to = df_to['dep_delay'].to_numpy(dtype=float)
    
    delay_change = arr_delay_to - dep_delay_from
    
    # Actual travel time derived from scheduled time and delay change
    actual_section_time = scheduled_section_time + delay_change
    # Physical floor: a train cannot traverse a section in <= 0 minutes
    actual_section_time = np.maximum(actual_section_time, 1.0)
    
    # Dwells
    scheduled_dwell_from = compute_elapsed_minutes(sch_arr_from_mins, sch_dep_from_mins).astype(float)
    actual_dwell_from = np.maximum(scheduled_dwell_from + (dep_delay_from - arr_delay_from), 0.0)
    
    # 4. Assembling base DataFrame
    train_nums = df_from['train'].to_numpy(dtype=np.int32)
    dates = df_from['date'].to_numpy()
    from_stns = df_from['station'].to_numpy()
    to_stns = df_to['station'].to_numpy()
    
    section_keys = [f"{t}_{f}_{s}" for t, f, s in zip(train_nums, from_stns, to_stns)]
    edge_keys = [f"{f}_{s}" for f, s in zip(from_stns, to_stns)]
    
    # Distances
    route_dist_series = pd.Series(section_keys).map(route_dist_lookup['route_dist_km'])
    edge_dist_series = pd.Series(edge_keys).map(edges['distance']) if 'distance' in edges.columns else pd.Series(np.nan, index=range(len(section_keys)))
    edge_ntrains_series = pd.Series(edge_keys).map(edges['ntrains']) if 'ntrains' in edges.columns else pd.Series(1, index=range(len(section_keys)))
    
    # Distance priority: Route timetable > IRN edges > default 5 km minimum
    final_dist = route_dist_series.fillna(edge_dist_series).fillna(5.0).to_numpy(dtype=float)
    final_dist = np.maximum(final_dist, 0.5)
    
    # Train names
    train_names = pd.Series(section_keys).map(route_dist_lookup['train_name']).fillna('Express').to_numpy()
    
    # Zones
    zones_arr = pd.Series(from_stns).map(zones).fillna('NR').to_numpy()
    
    # Dates & temporal
    date_dt = pd.to_datetime(dates)
    hour_of_day = (act_dep_from_mins // 60).astype(np.int32)
    day_of_week = date_dt.dayofweek.to_numpy(dtype=np.int32)
    is_weekend = np.isin(day_of_week, [5, 6]).astype(np.int32)
    day_of_month = date_dt.day.to_numpy(dtype=np.int32)
    
    section_runs = pd.DataFrame({
        'section_run_id': [f"{t}_{d}_{f}_{s}" for t, d, f, s in zip(train_nums, dates, from_stns, to_stns)],
        'train_number': train_nums,
        'train_name': train_names,
        'date': dates,
        'from_station': from_stns,
        'to_station': to_stns,
        'zone': zones_arr,
        'distance_km': final_dist,
        'scheduled_section_time': scheduled_section_time,
        'scheduled_dwell_from': scheduled_dwell_from,
        'actual_section_time': actual_section_time,
        'actual_dwell_from': actual_dwell_from,
        'arr_delay_from': arr_delay_from,
        'dep_delay_from': dep_delay_from,
        'arr_delay_to': arr_delay_to,
        'dep_delay_to': dep_delay_to,
        'delay_change': delay_change,
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'day_of_month': day_of_month,
        'edge_ntrains': edge_ntrains_series.fillna(1).to_numpy(dtype=np.int32)
    })
    
    # 5. Temporal Leakage-Free Historical Aggregates
    print("Computing leakage-free historical section aggregates (Sep 1 - Sep 22 only)...")
    train_mask = section_runs['date'] <= '2024-09-22'
    train_split = section_runs[train_mask]
    
    # Aggregate stats per (from_station, to_station) on train split
    sec_stats = train_split.groupby(['from_station', 'to_station'])['actual_section_time'].agg(
        section_median_time='median',
        section_mean_time='mean',
        section_p90_time=lambda s: np.percentile(s, 90) if len(s) > 0 else np.nan,
        section_min_time='min',
        section_std_time='std'
    ).reset_index()
    
    sec_stats['section_std_time'] = sec_stats['section_std_time'].fillna(0.0)
    
    # Global fallback defaults from train split
    global_median_ratio = float((train_split['actual_section_time'] / train_split['scheduled_section_time']).median())
    
    # Merge onto all rows
    section_runs = section_runs.merge(sec_stats, on=['from_station', 'to_station'], how='left')
    
    # Fill unobserved sections using global median ratio times scheduled section time
    section_runs['section_median_time'] = section_runs['section_median_time'].fillna(section_runs['scheduled_section_time'] * global_median_ratio)
    section_runs['section_mean_time'] = section_runs['section_mean_time'].fillna(section_runs['scheduled_section_time'] * global_median_ratio)
    section_runs['section_p90_time'] = section_runs['section_p90_time'].fillna(section_runs['scheduled_section_time'] * global_median_ratio * 1.3)
    section_runs['section_min_time'] = section_runs['section_min_time'].fillna(section_runs['scheduled_section_time'] * 0.85)
    section_runs['section_std_time'] = section_runs['section_std_time'].fillna(5.0)
    
    # 6. Assign Split Labels
    print("Assigning temporal split labels...")
    conditions = [
        section_runs['date'] <= '2024-09-22',
        (section_runs['date'] >= '2024-09-23') & (section_runs['date'] <= '2024-09-26'),
        section_runs['date'] >= '2024-09-27'
    ]
    choices = ['train', 'val', 'test']
    section_runs['split'] = np.select(conditions, choices, default='train')
    
    print(f"Split distribution:\n{section_runs['split'].value_counts(normalize=True).round(4) * 100}")
    
    # 7. Downstream Network State Features (RSTGCN Inspired, Zero-Leakage)
    print("Computing downstream network state and rolling temporal features...")
    section_runs = attach_network_state_features(section_runs)

    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving canonical table to {output_parquet}...")
    section_runs.to_parquet(output_parquet, index=False, engine='pyarrow', compression='snappy')
    print(f"Successfully built section_runs with {len(section_runs):,} records.")
    return section_runs


def attach_network_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes spatial multi-hop and temporal rolling network features ahead of the train
    with zero leakage (strictly using observations prior to current traversal, H-1).
    """
    date_dt = pd.to_datetime(df['date'])
    global_hour = ((date_dt.dt.day - 1) * 24 + df['hour_of_day']).to_numpy(dtype=np.int32)
    is_delayed = (df['dep_delay_from'] > 5).astype(np.int8)

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
    stn_to_idx = {s: i for i, s in enumerate(stations)}
    n_stns = len(stations)
    n_hours = 720

    grid_delay = np.zeros((n_stns, n_hours), dtype=np.float32)
    grid_delayed = np.zeros((n_stns, n_hours), dtype=np.float32)
    grid_active = np.zeros((n_stns, n_hours), dtype=np.float32)

    stn_indices = stn_hourly['station'].map(stn_to_idx).to_numpy()
    hour_indices = stn_hourly['global_hour'].to_numpy()

    grid_delay[stn_indices, hour_indices] = stn_hourly['stn_mean_delay'].to_numpy()
    grid_delayed[stn_indices, hour_indices] = stn_hourly['stn_delayed'].to_numpy()
    grid_active[stn_indices, hour_indices] = stn_hourly['stn_active'].to_numpy()

    is_next1 = (df['train_number'] == df['train_number'].shift(-1)) & (df['date'] == df['date'].shift(-1))
    is_next2 = is_next1 & (df['train_number'] == df['train_number'].shift(-2)) & (df['date'] == df['date'].shift(-2))

    stn_1hop = df['to_station']
    stn_2hop = np.where(is_next1, df['to_station'].shift(-1), df['to_station'])
    stn_3hop = np.where(is_next2, df['to_station'].shift(-2), stn_2hop)

    idx_1hop = pd.Series(stn_1hop).map(stn_to_idx).fillna(0).astype(int).to_numpy()
    idx_2hop = pd.Series(stn_2hop).map(stn_to_idx).fillna(0).astype(int).to_numpy()
    idx_3hop = pd.Series(stn_3hop).map(stn_to_idx).fillna(0).astype(int).to_numpy()

    h_lag1 = np.maximum(global_hour - 1, 0)
    h_lag2 = np.maximum(global_hour - 2, 0)

    df['net_1hop_mean_delay'] = grid_delay[idx_1hop, h_lag1]
    df['net_1hop_delayed_count'] = grid_delayed[idx_1hop, h_lag1]
    df['net_1hop_active_count'] = grid_active[idx_1hop, h_lag1]

    df['net_2hop_mean_delay'] = grid_delay[idx_2hop, h_lag1]
    df['net_2hop_delayed_count'] = grid_delayed[idx_2hop, h_lag1]

    df['net_3hop_mean_delay'] = grid_delay[idx_3hop, h_lag1]

    df['net_downstream_weighted_delay'] = 0.5 * df['net_1hop_mean_delay'] + 0.3 * df['net_2hop_mean_delay'] + 0.2 * df['net_3hop_mean_delay']
    df['net_downstream_delay_trend'] = df['net_1hop_mean_delay'] - df['net_2hop_mean_delay']

    df['recent_station_mean_delay'] = 0.5 * (grid_delay[idx_1hop, h_lag1] + grid_delay[idx_1hop, h_lag2])
    df['station_delay_trend_2h'] = grid_delay[idx_1hop, h_lag1] - grid_delay[idx_1hop, h_lag2]

    h_lag3 = np.maximum(global_hour - 3, 0)
    h_lag4 = np.maximum(global_hour - 4, 0)
    h_lag5 = np.maximum(global_hour - 5, 0)
    h_lag6 = np.maximum(global_hour - 6, 0)
    df['rolling_station_mean_delay_6h'] = (
        grid_delay[idx_1hop, h_lag1] + grid_delay[idx_1hop, h_lag2] +
        grid_delay[idx_1hop, h_lag3] + grid_delay[idx_1hop, h_lag4] +
        grid_delay[idx_1hop, h_lag5] + grid_delay[idx_1hop, h_lag6]
    ) / 6.0

    return df


if __name__ == '__main__':
    build_section_runs()
