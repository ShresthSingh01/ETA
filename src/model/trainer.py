"""
trainer.py - LightGBM Training Pipeline with Early Stopping, Feature Importance, and Ablation Studies.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np
import lightgbm as lgb

from src.model.features import (
    load_dataset_splits,
    get_feature_names,
    CATEGORICAL_FEATURES,
    TARGET_COL
)
from src.model.baselines import evaluate_predictions, run_baselines
from src.engine.rule_engine import SectionInfo, apply_railway_rules


LGBM_PARAMS = {
    'objective': 'regression_l1',  # Directly optimizes Mean Absolute Error
    'metric': ['mae', 'rmse'],
    'learning_rate': 0.05,
    'num_leaves': 127,
    'min_child_samples': 50,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42
}


def train_lgbm_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    params: dict = LGBM_PARAMS,
    max_rounds: int = 500,
    early_stopping_rounds: int = 30
) -> Tuple[lgb.Booster, dict]:
    """Trains LightGBM model with early stopping on validation split."""
    cat_features = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    
    dtrain = lgb.Dataset(
        train_df[feature_cols],
        label=train_df[TARGET_COL],
        categorical_feature=cat_features,
        free_raw_data=False
    )
    dval = lgb.Dataset(
        val_df[feature_cols],
        label=val_df[TARGET_COL],
        categorical_feature=cat_features,
        reference=dtrain,
        free_raw_data=False
    )
    
    evals_result = {}
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.record_evaluation(evals_result)
    ]
    
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=max_rounds,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        callbacks=callbacks
    )
    
    return booster, evals_result


def get_feature_importance(booster: lgb.Booster, feature_names: List[str]) -> pd.DataFrame:
    """Returns ranked feature importance by gain and split."""
    importance_gain = booster.feature_importance(importance_type='gain')
    importance_split = booster.feature_importance(importance_type='split')
    
    df_imp = pd.DataFrame({
        'feature': feature_names,
        'importance_gain': importance_gain,
        'importance_split': importance_split
    }).sort_values(by='importance_gain', ascending=False).reset_index(drop=True)
    
    df_imp['gain_pct'] = (df_imp['importance_gain'] / df_imp['importance_gain'].sum()) * 100.0
    return df_imp


def evaluate_horizon_stratification(test_df: pd.DataFrame, y_pred: np.ndarray) -> Dict[str, dict]:
    """Evaluates prediction accuracy across journey hop horizons."""
    df_eval = test_df[['train_number', 'date', 'scheduled_section_time', 'section_median_time', TARGET_COL]].copy()
    df_eval['y_pred'] = y_pred
    df_eval['hop'] = df_eval.groupby(['train_number', 'date']).cumcount() + 1
    df_eval['hop_bucket'] = pd.cut(
        df_eval['hop'],
        bins=[0, 1, 3, 5, 10, 500],
        labels=['1 hop', '2-3 hops', '4-5 hops', '6-10 hops', '11+ hops']
    )
    
    horizon_results = {}
    for bucket_name, g in df_eval.groupby('hop_bucket', observed=True):
        y_true = g[TARGET_COL].to_numpy()
        pred_m = g['y_pred'].to_numpy()
        pred_b1 = g['scheduled_section_time'].to_numpy()
        pred_b2 = g['section_median_time'].to_numpy()
        
        horizon_results[str(bucket_name)] = {
            'sample_count': int(len(g)),
            'LightGBM': evaluate_predictions(y_true, pred_m),
            'Baseline_Schedule_Naive': evaluate_predictions(y_true, pred_b1),
            'Baseline_Historical_Median': evaluate_predictions(y_true, pred_b2)
        }
    return horizon_results


def evaluate_scenario_buckets(test_df: pd.DataFrame, y_pred: np.ndarray) -> Dict[str, dict]:
    """Evaluates prediction accuracy stratified by initial departure delay severity."""
    df_eval = test_df[['dep_delay_from', 'scheduled_section_time', 'section_median_time', TARGET_COL]].copy()
    df_eval['y_pred'] = y_pred
    df_eval['delay_bucket'] = pd.cut(
        df_eval['dep_delay_from'],
        bins=[-999, 5, 30, 120, 99999],
        labels=['On-time (<5m)', 'Minor delay (5-30m)', 'Severe delay (30-120m)', 'Extreme delay (>120m)']
    )
    
    scenario_results = {}
    for bucket_name, g in df_eval.groupby('delay_bucket', observed=True):
        y_true = g[TARGET_COL].to_numpy()
        pred_m = g['y_pred'].to_numpy()
        pred_b1 = g['scheduled_section_time'].to_numpy()
        pred_b2 = g['section_median_time'].to_numpy()
        
        scenario_results[str(bucket_name)] = {
            'sample_count': int(len(g)),
            'LightGBM': evaluate_predictions(y_true, pred_m),
            'Baseline_Schedule_Naive': evaluate_predictions(y_true, pred_b1),
            'Baseline_Historical_Median': evaluate_predictions(y_true, pred_b2)
        }
    return scenario_results


def evaluate_network_pressure_buckets(
    test_df: pd.DataFrame,
    y_pred_m0: np.ndarray,
    y_pred_m3: np.ndarray
) -> Dict[str, dict]:
    """
    Evaluates prediction accuracy stratified by downstream network congestion pressure.
    Network pressure buckets (based on net_downstream_weighted_delay):
      - Normal: < 5 min
      - Low: 5-15 min
      - Medium: 15-30 min
      - High: >= 30 min
    """
    df_eval = test_df[['scheduled_section_time', 'section_median_time', TARGET_COL]].copy()
    if 'net_downstream_weighted_delay' in test_df.columns:
        df_eval['net_pressure'] = test_df['net_downstream_weighted_delay']
    else:
        df_eval['net_pressure'] = 0.0

    df_eval['y_pred_m0'] = y_pred_m0
    df_eval['y_pred_m3'] = y_pred_m3
    
    df_eval['pressure_bucket'] = pd.cut(
        df_eval['net_pressure'],
        bins=[-999, 5, 15, 30, 99999],
        labels=['Normal (<5m)', 'Low (5-15m)', 'Medium (15-30m)', 'High (>=30m)']
    )
    
    pressure_results = {}
    for bucket_name, g in df_eval.groupby('pressure_bucket', observed=True):
        y_true = g[TARGET_COL].to_numpy()
        pred_m0 = g['y_pred_m0'].to_numpy()
        pred_m3 = g['y_pred_m3'].to_numpy()
        pred_b1 = g['scheduled_section_time'].to_numpy()
        pred_b2 = g['section_median_time'].to_numpy()
        
        m0_metrics = evaluate_predictions(y_true, pred_m0)
        m3_metrics = evaluate_predictions(y_true, pred_m3)
        
        mae_gain = round(((m0_metrics['MAE'] - m3_metrics['MAE']) / m0_metrics['MAE']) * 100.0, 2)
        
        pressure_results[str(bucket_name)] = {
            'sample_count': int(len(g)),
            'Baseline_Schedule_Naive': evaluate_predictions(y_true, pred_b1),
            'Baseline_Historical_Median': evaluate_predictions(y_true, pred_b2),
            'Model_M0_Baseline': m0_metrics,
            'Model_M3_DownstreamNetwork': m3_metrics,
            'mae_improvement_pct': mae_gain
        }
    return pressure_results


def evaluate_rule_engine_impact(test_df: pd.DataFrame, y_pred: np.ndarray) -> Dict[str, Any]:
    """Quantifies deterministic rule engine intervention rate and impact on physical bounds."""
    clamped_preds = []
    audit_stats = {
        'total_samples': len(test_df),
        'clamped_or_adjusted_count': 0,
        'mps_floor_clamps': 0,
        'outlier_ceiling_clamps': 0,
        'recovery_cushion_clamps': 0,
    }
    
    for i, row in enumerate(test_df.itertuples()):
        sec = SectionInfo(
            from_station=row.from_station,
            to_station=row.to_station,
            distance_km=float(row.distance_km),
            scheduled_section_time=float(row.scheduled_section_time),
            min_historical_time=float(row.section_min_time),
            p90_time=float(row.section_p90_time),
            max_permissible_speed_kmh=110.0
        )
        res = apply_railway_rules(float(y_pred[i]), sec, current_dep_delay=float(row.dep_delay_from))
        clamped_preds.append(res.final_time)
        if res.is_clamped or res.is_adjusted:
            audit_stats['clamped_or_adjusted_count'] += 1
            for log in res.audit_trail:
                if 'MINIMUM' in log.rule_name:
                    audit_stats['mps_floor_clamps'] += 1
                elif 'CEILING' in log.rule_name:
                    audit_stats['outlier_ceiling_clamps'] += 1
                elif 'RECOVERY' in log.rule_name:
                    audit_stats['recovery_cushion_clamps'] += 1
                    
    clamped_arr = np.array(clamped_preds)
    y_test = test_df[TARGET_COL].to_numpy()
    
    audit_stats['clamped_pct'] = round(audit_stats['clamped_or_adjusted_count'] / len(test_df) * 100.0, 2)
    audit_stats['raw_ml_metrics'] = evaluate_predictions(y_test, y_pred)
    audit_stats['rule_engine_metrics'] = evaluate_predictions(y_test, clamped_arr)
    
    return audit_stats


def run_full_training(save_dir: str = 'models') -> dict:
    """
    Orchestrates baselines, M0/M1/M2/M3 ablation ladder, network pressure evaluation,
    and final test set evaluation.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading data splits for LightGBM training...")
    train_df, val_df, test_df = load_dataset_splits()
    y_val = val_df[TARGET_COL].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()
    
    # 1. Run Baselines
    print("\n" + "="*50)
    print("STEP 1: EVALUATING BASELINES")
    print("="*50)
    baseline_results = run_baselines()
    
    # 2. M0-M3 Ablation Ladder Training
    print("\n" + "="*50)
    print("STEP 2: TRAINING M0-M3 ABLATION LADDER")
    print("="*50)
    
    tiers = ['M0', 'M1', 'M2', 'M3']
    tier_descriptions = {
        'M0': 'Baseline (23 features, static network only)',
        'M1': 'M0 + Basic Downstream State (27 features)',
        'M2': 'M1 + Multi-hop Spatial & Downstream Trend (31 features)',
        'M3': 'Full RSTGCN-inspired Spatial-Temporal Model (34 features)'
    }
    
    boosters = {}
    val_preds = {}
    test_preds = {}
    val_metrics = {}
    test_metrics = {}
    
    for tier in tiers:
        print(f"\n--- Training Tier {tier}: {tier_descriptions[tier]} ---")
        features = get_feature_names(model_tier=tier)
        print(f"Features ({len(features)}): {features}")
        
        booster, _ = train_lgbm_model(train_df, val_df, features)
        boosters[tier] = booster
        
        v_pred = booster.predict(val_df[features])
        t_pred = booster.predict(test_df[features])
        
        val_preds[tier] = v_pred
        test_preds[tier] = t_pred
        
        val_metrics[tier] = evaluate_predictions(y_val, v_pred)
        test_metrics[tier] = evaluate_predictions(y_test, t_pred)
        
        print(f"Tier {tier} Validation: MAE={val_metrics[tier]['MAE']:.3f} | RMSE={val_metrics[tier]['RMSE']:.3f} | P90={val_metrics[tier]['P90_error']:.3f}")
        print(f"Tier {tier} Test Set:   MAE={test_metrics[tier]['MAE']:.3f} | RMSE={test_metrics[tier]['RMSE']:.3f} | P90={test_metrics[tier]['P90_error']:.3f}")
        
        # Save tier-specific booster
        tier_model_path = save_path / f"lightgbm_eta_{tier.lower()}.txt"
        booster.save_model(str(tier_model_path))
        print(f"Saved {tier} model to {tier_model_path}")
    
    # Save primary production booster (M3) as default
    booster_main = boosters['M3']
    features_main = get_feature_names(model_tier='M3')
    model_file = save_path / 'lightgbm_eta.txt'
    booster_main.save_model(str(model_file))
    print(f"\nPrimary production booster (M3) saved to {model_file}")
    
    # Feature Importance for M3
    df_imp = get_feature_importance(booster_main, features_main)
    print("\nTop 15 Most Important Features by Gain in M3 Model:")
    print(df_imp.head(15)[['feature', 'importance_gain', 'gain_pct']].to_string())
    df_imp.to_csv(save_path / 'feature_importance.csv', index=False)
    
    # Save Ablation Ladder Table
    ablation_ladder = {}
    for tier in tiers:
        ablation_ladder[f'{tier} - {tier_descriptions[tier]}'] = {
            'features_count': len(get_feature_names(model_tier=tier)),
            'val_MAE': val_metrics[tier]['MAE'],
            'val_RMSE': val_metrics[tier]['RMSE'],
            'test_MAE': test_metrics[tier]['MAE'],
            'test_RMSE': test_metrics[tier]['RMSE'],
            'test_Within_5m_pct': test_metrics[tier]['Pct_within_5m'],
            'test_Within_15m_pct': test_metrics[tier]['Pct_within_15m'],
            'improvement_over_M0_pct': round(((test_metrics['M0']['MAE'] - test_metrics[tier]['MAE']) / test_metrics['M0']['MAE']) * 100.0, 2)
        }
    df_ladder = pd.DataFrame(ablation_ladder).T
    print("\n" + "="*70)
    print("M0 - M3 ABLATION LADDER COMPARISON")
    print("="*70)
    print(df_ladder.to_string())
    df_ladder.to_csv(save_path / 'ablation_ladder.csv')
    with open(save_path / 'ablation_ladder.json', 'w') as f:
        json.dump(ablation_ladder, f, indent=2)

    # 3. Network Pressure Stratified Evaluation (Phase 4)
    print("\n" + "="*50)
    print("STEP 3: NETWORK PRESSURE STRATIFIED EVALUATION")
    print("="*50)
    pressure_results = evaluate_network_pressure_buckets(test_df, test_preds['M0'], test_preds['M3'])
    with open(save_path / 'network_pressure_evaluation.json', 'w') as f:
        json.dump(pressure_results, f, indent=2)
    
    pressure_rows = []
    for bucket, bdata in pressure_results.items():
        row = {
            'pressure_bucket': bucket,
            'sample_count': bdata['sample_count'],
            'Baseline_MAE': bdata['Baseline_Schedule_Naive']['MAE'],
            'Median_MAE': bdata['Baseline_Historical_Median']['MAE'],
            'M0_Baseline_MAE': bdata['Model_M0_Baseline']['MAE'],
            'M3_Network_MAE': bdata['Model_M3_DownstreamNetwork']['MAE'],
            'mae_improvement_pct': bdata['mae_improvement_pct']
        }
        pressure_rows.append(row)
    df_pressure = pd.DataFrame(pressure_rows)
    print(df_pressure.to_string(index=False))
    df_pressure.to_csv(save_path / 'network_pressure_evaluation.csv', index=False)
    
    # 4. Final Evaluation on Holdout Test Set
    print("\n" + "="*50)
    print("STEP 4: FINAL TEST SET EVALUATION & RULE ENGINE")
    print("="*50)
    test_pred_b1 = test_df['scheduled_section_time'].to_numpy()
    test_pred_b2 = test_df['section_median_time'].to_numpy()
    test_pred_main = test_preds['M3']
    
    test_metrics_b1 = evaluate_predictions(y_test, test_pred_b1)
    test_metrics_b2 = evaluate_predictions(y_test, test_pred_b2)
    test_metrics_main = test_metrics['M3']
    
    # Rule engine impact
    print("\nEvaluating Rule Engine Impact on Test Set...")
    rule_impact = evaluate_rule_engine_impact(test_df, test_pred_main)
    print(f"Rule intervention rate: {rule_impact['clamped_pct']}% ({rule_impact['clamped_or_adjusted_count']}/{rule_impact['total_samples']})")
    print(f"MPS speed floor violations clamped: {rule_impact['mps_floor_clamps']}")
    print(f"Recovery margin cushions clamped: {rule_impact['recovery_cushion_clamps']}")
    print(f"Raw ML MAE: {rule_impact['raw_ml_metrics']['MAE']}m | Rule Engine Clamped MAE: {rule_impact['rule_engine_metrics']['MAE']}m")
    
    # Horizon stratification
    print("\nEvaluating Horizon Stratification...")
    horizon_results = evaluate_horizon_stratification(test_df, test_pred_main)
    
    # Scenario buckets
    print("Evaluating Delay Severity Scenario Buckets...")
    scenario_results = evaluate_scenario_buckets(test_df, test_pred_main)
    
    # Compile Full Summary Table
    summary_table = {
        'Validation - Baseline 1 (Schedule Naive)': baseline_results['Baseline 1 (Schedule Naive)'],
        'Validation - Baseline 2 (Historical Median)': baseline_results['Baseline 2 (Historical Median)'],
        'Validation - Baseline 3 (Linear Ridge)': baseline_results['Baseline 3 (Linear Ridge)'],
        'Validation - Model M0 (Baseline 23 feats)': val_metrics['M0'],
        'Validation - Model M1 (Basic Downstream)': val_metrics['M1'],
        'Validation - Model M2 (Multi-hop + Trend)': val_metrics['M2'],
        'Validation - Model M3 (Full RSTGCN)': val_metrics['M3'],
        'Validation - Main LightGBM': val_metrics['M3'],
        'Test Set - Baseline 1 (Schedule Naive)': test_metrics_b1,
        'Test Set - Baseline 2 (Historical Median)': test_metrics_b2,
        'Test Set - Model M0 (Baseline 23 feats)': test_metrics['M0'],
        'Test Set - Model M1 (Basic Downstream)': test_metrics['M1'],
        'Test Set - Model M2 (Multi-hop + Trend)': test_metrics['M2'],
        'Test Set - Model M3 (Full RSTGCN)': test_metrics['M3'],
        'Test Set - Main LightGBM': test_metrics['M3'],
        'Test Set - M3 + Rule Engine': rule_impact['rule_engine_metrics'],
        'Test Set - LightGBM + Rule Engine': rule_impact['rule_engine_metrics']
    }
    
    df_summary = pd.DataFrame(summary_table).T
    print("\n" + "="*70)
    print("FINAL BENCHMARK AND EVALUATION REPORT")
    print("="*70)
    print(df_summary.to_string())
    
    # Save results to JSON and CSV
    df_summary.to_csv(save_path / 'evaluation_summary.csv')
    with open(save_path / 'evaluation_summary.json', 'w') as f:
        json.dump(summary_table, f, indent=2)
        
    with open(save_path / 'horizon_evaluation.json', 'w') as f:
        json.dump(horizon_results, f, indent=2)
        
    with open(save_path / 'scenario_evaluation.json', 'w') as f:
        json.dump(scenario_results, f, indent=2)
        
    with open(save_path / 'rule_impact_evaluation.json', 'w') as f:
        json.dump(rule_impact, f, indent=2)
        
    # Also save flattened CSVs
    horizon_rows = []
    for bucket, bdata in horizon_results.items():
        row = {'horizon_bucket': bucket, 'samples': bdata['sample_count']}
        for mkey, mvals in bdata.items():
            if mkey != 'sample_count':
                for k, v in mvals.items():
                    row[f'{mkey}_{k}'] = v
        horizon_rows.append(row)
    pd.DataFrame(horizon_rows).to_csv(save_path / 'horizon_evaluation.csv', index=False)
    
    scenario_rows = []
    for bucket, bdata in scenario_results.items():
        row = {'delay_bucket': bucket, 'samples': bdata['sample_count']}
        for mkey, mvals in bdata.items():
            if mkey != 'sample_count':
                for k, v in mvals.items():
                    row[f'{mkey}_{k}'] = v
        scenario_rows.append(row)
    pd.DataFrame(scenario_rows).to_csv(save_path / 'scenario_evaluation.csv', index=False)
        
    print(f"\nAll artifacts successfully saved to {save_path}/")
    return summary_table


if __name__ == '__main__':
    run_full_training()
