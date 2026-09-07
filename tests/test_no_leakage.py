"""
test_no_leakage.py - Verification that temporal splits and historical aggregates have zero future leakage.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pytest


@pytest.fixture(scope="module")
def df_runs():
    parquet_path = Path('data/processed/section_runs.parquet')
    assert parquet_path.exists(), "Canonical section_runs.parquet must exist"
    return pd.read_parquet(parquet_path)


def test_temporal_split_dates(df_runs):
    train_dates = df_runs[df_runs['split'] == 'train']['date'].unique()
    val_dates = df_runs[df_runs['split'] == 'val']['date'].unique()
    test_dates = df_runs[df_runs['split'] == 'test']['date'].unique()
    
    assert max(train_dates) <= '2024-09-22', f"Train split leaked into future: max is {max(train_dates)}"
    assert min(val_dates) >= '2024-09-23', f"Val split starts too early: {min(val_dates)}"
    assert max(val_dates) <= '2024-09-26', f"Val split extends past Sep 26: {max(val_dates)}"
    assert min(test_dates) >= '2024-09-27', f"Test split starts too early: {min(test_dates)}"
    assert max(test_dates) <= '2024-09-30', f"Test split extends past Sep 30: {max(test_dates)}"
    
    # Mutual exclusivity of date sets
    assert set(train_dates).isdisjoint(set(val_dates)), "Train and Val dates overlap!"
    assert set(train_dates).isdisjoint(set(test_dates)), "Train and Test dates overlap!"
    assert set(val_dates).isdisjoint(set(test_dates)), "Val and Test dates overlap!"


def test_zero_nulls_in_critical_columns(df_runs):
    critical_cols = [
        'section_run_id', 'train_number', 'date', 'from_station', 'to_station',
        'scheduled_section_time', 'actual_section_time', 'arr_delay_from', 'dep_delay_from',
        'distance_km', 'hour_of_day', 'day_of_week', 'section_median_time', 'section_p90_time',
        'split'
    ]
    for col in critical_cols:
        assert col in df_runs.columns, f"Missing column: {col}"
        null_count = df_runs[col].isna().sum()
        assert null_count == 0, f"Column {col} has {null_count} nulls"


def test_physical_sanity(df_runs):
    # Running times must be positive
    assert (df_runs['actual_section_time'] <= 0).sum() == 0, "Actual section time must be strictly positive"
    assert (df_runs['scheduled_section_time'] <= 0).sum() == 0, "Scheduled section time must be strictly positive"
    assert (df_runs['distance_km'] <= 0).sum() == 0, "Distance must be strictly positive"
    
    # Floor sanity: minimum actual section time >= 1 min
    assert df_runs['actual_section_time'].min() >= 1.0, "Actual travel time below 1 min floor"


def test_leakage_free_aggregates(df_runs):
    # Verify that historical section medians match train set medians, not overall medians
    train_df = df_runs[df_runs['split'] == 'train']
    sample_pair = ('HWH', 'BWN')
    
    if len(train_df[(train_df['from_station'] == sample_pair[0]) & (train_df['to_station'] == sample_pair[1])]) > 0:
        expected_median = train_df[(train_df['from_station'] == sample_pair[0]) & (train_df['to_station'] == sample_pair[1])]['actual_section_time'].median()
        row = df_runs[(df_runs['from_station'] == sample_pair[0]) & (df_runs['to_station'] == sample_pair[1])].iloc[0]
        assert np.isclose(row['section_median_time'], expected_median, atol=1e-3), \
            f"Expected train median {expected_median}, got {row['section_median_time']}"
