"""
baselines.py - Implements and evaluates the 3 baseline models against validation and test sets.
"""

from typing import Dict
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.model.features import load_dataset_splits, NUMERIC_FEATURES, TARGET_COL


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics: MAE, RMSE, R2, within 5m %, within 10m %, within 15m %, P90 error."""
    err = np.abs(y_true - y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pct_5 = (err <= 5.0).mean() * 100.0
    pct_10 = (err <= 10.0).mean() * 100.0
    pct_15 = (err <= 15.0).mean() * 100.0
    p90 = float(np.percentile(err, 90))
    
    return {
        'MAE': round(float(mae), 3),
        'RMSE': round(float(rmse), 3),
        'R2': round(float(r2), 4),
        'Pct_within_5m': round(float(pct_5), 2),
        'Pct_within_10m': round(float(pct_10), 2),
        'Pct_within_15m': round(float(pct_15), 2),
        'P90_error': round(float(p90), 2)
    }


def run_baselines() -> Dict[str, Dict[str, float]]:
    """Runs all 3 baselines and returns comparison dictionary."""
    print("Loading splits for baseline benchmarking...")
    train_df, val_df, test_df = load_dataset_splits()
    
    y_val = val_df[TARGET_COL].to_numpy()
    results = {}
    
    # Baseline 1: NTES Delay Propagation / Scheduled Section Time
    # Note: NTES assumes current departure delay propagates unchanged (delta_delay = 0),
    # which implies section traversal time equals scheduled_section_time.
    print("Evaluating Baseline 1: Scheduled Time (NTES constant delay propagation)...")
    b1_pred = val_df['scheduled_section_time'].to_numpy()
    results['Baseline 1 (Schedule Naive)'] = evaluate_predictions(y_val, b1_pred)
    
    # Baseline 2: Historical Section Median
    print("Evaluating Baseline 2: Historical Section Median...")
    b2_pred = val_df['section_median_time'].to_numpy()
    results['Baseline 2 (Historical Median)'] = evaluate_predictions(y_val, b2_pred)
    
    # Baseline 3: Ridge Linear Regression
    print("Training and evaluating Baseline 3: Ridge Linear Regression...")
    linear_features = ['scheduled_section_time', 'distance_km', 'dep_delay_from', 'arr_delay_from', 'hour_of_day', 'day_of_week']
    
    X_train = train_df[linear_features].fillna(0.0).to_numpy()
    y_train = train_df[TARGET_COL].to_numpy()
    X_val = val_df[linear_features].fillna(0.0).to_numpy()
    
    lr = Ridge(alpha=1.0)
    lr.fit(X_train, y_train)
    b3_pred = lr.predict(X_val)
    results['Baseline 3 (Linear Ridge)'] = evaluate_predictions(y_val, b3_pred)
    
    # Print comparison table
    df_results = pd.DataFrame(results).T
    print("\n--- BASELINES BENCHMARK COMPARISON (VALIDATION SET) ---")
    print(df_results.to_string())
    
    return results


if __name__ == '__main__':
    run_baselines()
