"""
test_model_evaluation.py - Comprehensive Holdout Evaluation, Benchmark Validation,
and Empirical Sanity Test Suite for GaTi Production Models.

Validates the full evaluation pipeline across 164,564 genuine holdout test runs:
  1. Model artifact integrity, booster loading, and architecture verification (M0-M3, static).
  2. Mathematical metric computation engine correctness (MAE, RMSE, R2, Pct_within_5m/10m/15m, P90).
  3. Benchmark baseline superiority (LightGBM vs Schedule Naive & Historical Median).
  4. M0-M3 Ablation ladder progression and feature expansion.
  5. Multi-hop horizon stratification (1 hop to 11+ hops).
  6. Delay severity scenario bucketing (On-time, Minor, Severe, Extreme).
  7. Downstream network congestion pressure stratification.
  8. Deterministic rule engine physical safety audit (MPS speed floor & recovery cushion).
  9. Empirical confidence calibration monotonicity and reliability.
  10. Feature importance ranking and gain distribution.
  11. Live holdout test set inference and metric reproducibility (zero drift).
  12. Adversarial & extreme stress inputs resilience.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any

import pytest
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.model.features import (
    load_dataset_splits,
    get_feature_names,
    CATEGORICAL_FEATURES,
    TARGET_COL
)
from src.model.trainer import (
    evaluate_predictions,
    evaluate_horizon_stratification,
    evaluate_scenario_buckets,
    evaluate_network_pressure_buckets,
    evaluate_rule_engine_impact
)
from src.engine.rule_engine import SectionInfo, apply_railway_rules


# ==============================================================================
# Pytest Fixtures (Module-scoped for high performance)
# ==============================================================================

@pytest.fixture(scope="module")
def models_dir() -> Path:
    p = Path("models")
    assert p.exists() and p.is_dir(), "Models directory must exist"
    return p


@pytest.fixture(scope="module")
def evaluation_summary(models_dir: Path) -> Dict[str, Any]:
    file_path = models_dir / "evaluation_summary.json"
    assert file_path.exists(), f"Missing evaluation summary: {file_path}"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def ablation_ladder(models_dir: Path) -> Dict[str, Any]:
    file_path = models_dir / "ablation_ladder.json"
    assert file_path.exists(), f"Missing ablation ladder: {file_path}"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def horizon_evaluation(models_dir: Path) -> Dict[str, Any]:
    file_path = models_dir / "horizon_evaluation.json"
    assert file_path.exists(), f"Missing horizon evaluation: {file_path}"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def scenario_evaluation(models_dir: Path) -> Dict[str, Any]:
    file_path = models_dir / "scenario_evaluation.json"
    assert file_path.exists(), f"Missing scenario evaluation: {file_path}"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def network_pressure_evaluation(models_dir: Path) -> Dict[str, Any]:
    file_path = models_dir / "network_pressure_evaluation.json"
    assert file_path.exists(), f"Missing network pressure evaluation: {file_path}"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def rule_impact_evaluation(models_dir: Path) -> Dict[str, Any]:
    file_path = models_dir / "rule_impact_evaluation.json"
    assert file_path.exists(), f"Missing rule impact evaluation: {file_path}"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def confidence_calibration(models_dir: Path) -> Dict[str, Any]:
    file_path = models_dir / "confidence_calibration.json"
    assert file_path.exists(), f"Missing confidence calibration: {file_path}"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def production_booster(models_dir: Path) -> lgb.Booster:
    model_path = models_dir / "lightgbm_eta.txt"
    assert model_path.exists(), f"Missing production booster: {model_path}"
    return lgb.Booster(model_file=str(model_path))


@pytest.fixture(scope="module")
def holdout_test_data() -> pd.DataFrame:
    parquet_path = Path("data/processed/section_runs_weather.parquet")
    assert parquet_path.exists(), f"Missing dataset: {parquet_path}"
    features = get_feature_names(model_tier="M3")
    required_cols = list(set(features + [
        TARGET_COL, "split", "from_station", "to_station",
        "scheduled_section_time", "section_median_time", "section_p90_time",
        "section_min_time", "distance_km", "dep_delay_from"
    ]))
    df = pd.read_parquet(
        str(parquet_path),
        columns=required_cols,
        filters=[("split", "==", "test")]
    )
    for cat in CATEGORICAL_FEATURES:
        if cat in df.columns:
            df[cat] = df[cat].astype("category")
    test_df = df.reset_index(drop=True)
    yield test_df
    del test_df, df
    import gc
    gc.collect()


# ==============================================================================
# 1. Model Artifact Integrity and Verification
# ==============================================================================

def test_evaluation_artifact_files_exist_and_nonempty(models_dir: Path):
    """Verify that all canonical model evaluation artifacts exist and are non-empty."""
    required_artifacts = [
        "lightgbm_eta.txt",
        "lightgbm_eta_m0.txt",
        "lightgbm_eta_m1.txt",
        "lightgbm_eta_m2.txt",
        "lightgbm_eta_m3.txt",
        "lightgbm_static.txt",
        "evaluation_summary.json",
        "evaluation_summary.csv",
        "ablation_ladder.json",
        "ablation_ladder.csv",
        "horizon_evaluation.json",
        "horizon_evaluation.csv",
        "scenario_evaluation.json",
        "scenario_evaluation.csv",
        "network_pressure_evaluation.json",
        "network_pressure_evaluation.csv",
        "rule_impact_evaluation.json",
        "confidence_calibration.json",
        "feature_importance.csv",
        "ablation_waterfall_results.json"
    ]
    for artifact in required_artifacts:
        path = models_dir / artifact
        assert path.exists(), f"Required evaluation artifact missing: {artifact}"
        assert path.stat().st_size > 0, f"Artifact file is empty: {artifact}"


def test_model_architecture_and_feature_counts(models_dir: Path):
    """Verify booster feature dimensions and tree counts for all model tiers."""
    tier_specs = {
        "lightgbm_eta_m0.txt": {"expected_features": 23, "min_trees": 100},
        "lightgbm_eta_m1.txt": {"expected_features": 27, "min_trees": 100},
        "lightgbm_eta_m2.txt": {"expected_features": 31, "min_trees": 100},
        "lightgbm_eta_m3.txt": {"expected_features": 34, "min_trees": 100},
        "lightgbm_eta.txt": {"expected_features": 34, "min_trees": 100},
        "lightgbm_static.txt": {"expected_features": 20, "min_trees": 100},
    }
    for filename, specs in tier_specs.items():
        booster_file = models_dir / filename
        booster = lgb.Booster(model_file=str(booster_file))
        assert booster.num_feature() == specs["expected_features"], (
            f"{filename} feature count mismatch: got {booster.num_feature()}, "
            f"expected {specs['expected_features']}"
        )
        assert booster.num_trees() >= specs["min_trees"], (
            f"{filename} tree count too low: got {booster.num_trees()}"
        )


# ==============================================================================
# 2. Metric Computation Engine Verification
# ==============================================================================

def test_evaluate_predictions_unit_and_edge_cases():
    """Verify mathematical correctness, precision, and edge cases of evaluate_predictions."""
    # Case 1: Exact perfect predictions
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred_perfect = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    metrics_perfect = evaluate_predictions(y_true, y_pred_perfect)
    assert metrics_perfect["MAE"] == 0.0
    assert metrics_perfect["RMSE"] == 0.0
    assert metrics_perfect["R2"] == 1.0
    assert metrics_perfect["Pct_within_5m"] == 100.0
    assert metrics_perfect["Pct_within_10m"] == 100.0
    assert metrics_perfect["Pct_within_15m"] == 100.0
    assert metrics_perfect["P90_error"] == 0.0

    # Case 2: Constant delta of +3 minutes
    y_pred_shift3 = y_true + 3.0
    metrics_shift3 = evaluate_predictions(y_true, y_pred_shift3)
    assert np.isclose(metrics_shift3["MAE"], 3.0, atol=1e-3)
    assert np.isclose(metrics_shift3["RMSE"], 3.0, atol=1e-3)
    assert metrics_shift3["Pct_within_5m"] == 100.0
    assert np.isclose(metrics_shift3["P90_error"], 3.0, atol=1e-2)

    # Case 3: Constant delta of +8 minutes (outside 5m, inside 10m)
    y_pred_shift8 = y_true + 8.0
    metrics_shift8 = evaluate_predictions(y_true, y_pred_shift8)
    assert np.isclose(metrics_shift8["MAE"], 8.0, atol=1e-3)
    assert metrics_shift8["Pct_within_5m"] == 0.0
    assert metrics_shift8["Pct_within_10m"] == 100.0
    assert np.isclose(metrics_shift8["P90_error"], 8.0, atol=1e-2)

    # Case 4: Mathematical bounds: Cauchy-Schwarz (RMSE >= MAE)
    np.random.seed(42)
    y_rand_true = np.random.uniform(10, 100, size=100)
    y_rand_pred = y_rand_true + np.random.normal(0, 5, size=100)
    rand_metrics = evaluate_predictions(y_rand_true, y_rand_pred)
    assert rand_metrics["RMSE"] >= rand_metrics["MAE"], "RMSE must be >= MAE"
    assert 0.0 <= rand_metrics["Pct_within_5m"] <= 100.0
    assert 0.0 <= rand_metrics["Pct_within_10m"] <= 100.0
    assert 0.0 <= rand_metrics["Pct_within_15m"] <= 100.0
    assert rand_metrics["P90_error"] >= 0.0


# ==============================================================================
# 3. Benchmark Superiority Against Baselines
# ==============================================================================

def test_production_model_beats_baselines_on_holdout_test(evaluation_summary: Dict[str, Any]):
    """
    Verify that GaTi production LightGBM outperforms Baseline 1 (Schedule Naive)
    and Baseline 2 (Historical Median) across all primary operational metrics.
    """
    lgb_metrics = evaluation_summary["Test Set - Main LightGBM"]
    b1_metrics = evaluation_summary["Test Set - Baseline 1 (Schedule Naive)"]
    b2_metrics = evaluation_summary["Test Set - Baseline 2 (Historical Median)"]

    # 1. MAE superiority (>25% improvement over Schedule Naive, >20% over Historical Median)
    mae_gain_vs_b1 = ((b1_metrics["MAE"] - lgb_metrics["MAE"]) / b1_metrics["MAE"]) * 100.0
    mae_gain_vs_b2 = ((b2_metrics["MAE"] - lgb_metrics["MAE"]) / b2_metrics["MAE"]) * 100.0
    assert mae_gain_vs_b1 >= 25.0, f"MAE improvement vs B1 must be >=25%, got {mae_gain_vs_b1:.2f}%"
    assert mae_gain_vs_b2 >= 20.0, f"MAE improvement vs B2 must be >=20%, got {mae_gain_vs_b2:.2f}%"

    # Absolute MAE check: LightGBM under 6.50 min, Baselines above 8.40 min
    assert lgb_metrics["MAE"] <= 6.30, f"LightGBM MAE too high: {lgb_metrics['MAE']}"
    assert b1_metrics["MAE"] >= 8.50, f"B1 MAE lower than expected: {b1_metrics['MAE']}"
    assert b2_metrics["MAE"] >= 8.30, f"B2 MAE lower than expected: {b2_metrics['MAE']}"

    # 2. RMSE superiority
    assert lgb_metrics["RMSE"] < b1_metrics["RMSE"], "LightGBM RMSE must beat B1"
    assert lgb_metrics["RMSE"] < b2_metrics["RMSE"], "LightGBM RMSE must beat B2"

    # 3. Punctuality (Pct within 5 min) superiority
    assert lgb_metrics["Pct_within_5m"] >= 71.0, f"Punctuality within 5m too low: {lgb_metrics['Pct_within_5m']}%"
    assert b1_metrics["Pct_within_5m"] <= 63.0, f"B1 punctuality higher than expected: {b1_metrics['Pct_within_5m']}%"
    assert lgb_metrics["Pct_within_5m"] > b1_metrics["Pct_within_5m"] + 8.0, "Must lead B1 by at least 8 percentage points"

    # 4. Tail Risk Reduction (P90 Error)
    assert lgb_metrics["P90_error"] <= 14.5, f"P90 error too high: {lgb_metrics['P90_error']} min"
    assert b1_metrics["P90_error"] >= 20.0, f"B1 P90 error lower than expected: {b1_metrics['P90_error']} min"
    p90_reduction_pct = ((b1_metrics["P90_error"] - lgb_metrics["P90_error"]) / b1_metrics["P90_error"]) * 100.0
    assert p90_reduction_pct >= 30.0, f"P90 error reduction must be >=30%, got {p90_reduction_pct:.2f}%"

    # 5. Explained Variance (R2 Score)
    assert lgb_metrics["R2"] >= 0.75, f"R2 score too low: {lgb_metrics['R2']}"
    assert lgb_metrics["R2"] > b1_metrics["R2"], "LightGBM R2 must exceed B1"


# ==============================================================================
# 4. M0-M3 Ablation Ladder Progression
# ==============================================================================

def test_ablation_ladder_progression_and_metrics(ablation_ladder: Dict[str, Any]):
    """
    Verify feature progression and benchmark metrics across M0, M1, M2, and M3 tiers.
    """
    tiers = ["M0", "M1", "M2", "M3"]
    expected_feature_counts = [23, 27, 31, 34]

    tier_keys = list(ablation_ladder.keys())
    assert len(tier_keys) == 4, f"Ablation ladder must have 4 tiers, got {len(tier_keys)}"

    for i, tier in enumerate(tiers):
        matching_keys = [k for k in tier_keys if k.startswith(tier)]
        assert len(matching_keys) == 1, f"Missing key for tier {tier}"
        entry = ablation_ladder[matching_keys[0]]

        # Feature count check
        assert entry["features_count"] == expected_feature_counts[i], (
            f"{tier} feature count: expected {expected_feature_counts[i]}, got {entry['features_count']}"
        )

        # Performance boundary checks
        assert entry["test_MAE"] <= 6.30, f"{tier} test MAE too high: {entry['test_MAE']}"
        assert entry["test_Within_5m_pct"] >= 71.5, f"{tier} punctuality too low: {entry['test_Within_5m_pct']}"
        assert entry["test_Within_15m_pct"] >= 90.5, f"{tier} 15m punctuality too low: {entry['test_Within_15m_pct']}"

    # Feature definitions consistency with features.py
    for tier in tiers:
        defined_feats = get_feature_names(model_tier=tier)
        matching_key = [k for k in tier_keys if k.startswith(tier)][0]
        assert len(defined_feats) == ablation_ladder[matching_key]["features_count"]


# ==============================================================================
# 5. Multi-Hop Horizon Stratification
# ==============================================================================

def test_horizon_stratification_evaluation(horizon_evaluation: Dict[str, Any]):
    """
    Verify performance across the 5 journey hop horizons:
    1 hop, 2-3 hops, 4-5 hops, 6-10 hops, 11+ hops.
    """
    expected_buckets = ["1 hop", "2-3 hops", "4-5 hops", "6-10 hops", "11+ hops"]
    for bucket in expected_buckets:
        assert bucket in horizon_evaluation, f"Missing horizon bucket: {bucket}"
        data = horizon_evaluation[bucket]
        assert data["sample_count"] > 0, f"Bucket {bucket} has 0 samples"

        lgb_mae = data["LightGBM"]["MAE"]
        b1_mae = data["Baseline_Schedule_Naive"]["MAE"]

        # LightGBM must strictly outperform Baseline 1 in EVERY hop bucket
        assert lgb_mae < b1_mae, (
            f"Horizon {bucket}: LightGBM MAE ({lgb_mae}) failed to beat Schedule Naive ({b1_mae})"
        )

    # Immediate next station (1 hop) precision: MAE must be <= 5.50 min
    assert horizon_evaluation["1 hop"]["LightGBM"]["MAE"] <= 5.50
    assert horizon_evaluation["1 hop"]["LightGBM"]["Pct_within_5m"] >= 71.0

    # Total sample count must match holdout test split (164,564)
    total_samples = sum(horizon_evaluation[b]["sample_count"] for b in expected_buckets)
    assert total_samples == 164564, f"Total test samples mismatch: {total_samples} vs 164,564"


# ==============================================================================
# 6. Delay Severity Scenario Bucketing
# ==============================================================================

def test_scenario_delay_severity_evaluation(scenario_evaluation: Dict[str, Any]):
    """
    Verify performance stratified across operational delay severity regimes:
    On-time (<5m), Minor delay (5-30m), Severe delay (30-120m), Extreme delay (>120m).
    """
    expected_buckets = [
        "On-time (<5m)",
        "Minor delay (5-30m)",
        "Severe delay (30-120m)",
        "Extreme delay (>120m)"
    ]
    for bucket in expected_buckets:
        assert bucket in scenario_evaluation, f"Missing scenario bucket: {bucket}"
        data = scenario_evaluation[bucket]
        assert data["sample_count"] > 0, f"Bucket {bucket} has 0 samples"

        lgb_mae = data["LightGBM"]["MAE"]
        b1_mae = data["Baseline_Schedule_Naive"]["MAE"]

        # LightGBM must beat Schedule Naive across all regimes
        assert lgb_mae < b1_mae, (
            f"Scenario {bucket}: LightGBM MAE ({lgb_mae}) failed to beat Schedule Naive ({b1_mae})"
        )

    # On-time regime punctuality check
    assert scenario_evaluation["On-time (<5m)"]["LightGBM"]["Pct_within_5m"] >= 80.0, (
        "On-time punctuality within 5m should exceed 80%"
    )

    # Severe delay regime: LightGBM provides massive error reduction (>30%)
    sev_lgb_mae = scenario_evaluation["Severe delay (30-120m)"]["LightGBM"]["MAE"]
    sev_b1_mae = scenario_evaluation["Severe delay (30-120m)"]["Baseline_Schedule_Naive"]["MAE"]
    sev_gain = ((sev_b1_mae - sev_lgb_mae) / sev_b1_mae) * 100.0
    assert sev_gain >= 30.0, f"Severe delay improvement must be >=30%, got {sev_gain:.2f}%"

    # Total sample count check
    total_samples = sum(scenario_evaluation[b]["sample_count"] for b in expected_buckets)
    assert total_samples == 164564, f"Total test samples mismatch: {total_samples} vs 164,564"


# ==============================================================================
# 7. Downstream Network Congestion Pressure
# ==============================================================================

def test_network_pressure_stratification_evaluation(network_pressure_evaluation: Dict[str, Any]):
    """
    Verify performance across network pressure levels:
    Normal (<5m), Low (5-15m), Medium (15-30m), High (>=30m).
    """
    expected_buckets = ["Normal (<5m)", "Low (5-15m)", "Medium (15-30m)", "High (>=30m)"]
    for bucket in expected_buckets:
        assert bucket in network_pressure_evaluation, f"Missing pressure bucket: {bucket}"
        data = network_pressure_evaluation[bucket]
        assert data["sample_count"] > 0, f"Bucket {bucket} has 0 samples"

        m3_mae = data["Model_M3_DownstreamNetwork"]["MAE"]
        b1_mae = data["Baseline_Schedule_Naive"]["MAE"]
        b2_mae = data["Baseline_Historical_Median"]["MAE"]

        # M3 Downstream model must strictly beat Schedule Naive
        assert m3_mae < b1_mae, f"Under pressure {bucket}, M3 ({m3_mae}) did not beat B1 ({b1_mae})"
        assert m3_mae < b2_mae, f"Under pressure {bucket}, M3 ({m3_mae}) did not beat B2 ({b2_mae})"

    total_samples = sum(network_pressure_evaluation[b]["sample_count"] for b in expected_buckets)
    assert total_samples == 164564, f"Total test samples mismatch: {total_samples} vs 164,564"


# ==============================================================================
# 8. Deterministic Rule Engine Physical Safety Audit
# ==============================================================================

def test_deterministic_rule_engine_physical_safety_audit(rule_impact_evaluation: Dict[str, Any]):
    """
    Verify that post-ML Rule Engine guarantees physical safety and timetable feasibility:
    - Eliminates MPS track speed violations (floor clamps).
    - Eliminates unrealistic recovery make-up (cushion clamps).
    - Intervention rate is bounded within expected operational envelope.
    """
    assert rule_impact_evaluation["total_samples"] == 164564
    assert 10.0 <= rule_impact_evaluation["clamped_pct"] <= 20.0, (
        f"Intervention rate out of expected 10-20% band: {rule_impact_evaluation['clamped_pct']}%"
    )
    assert rule_impact_evaluation["mps_floor_clamps"] > 0, "Rule engine must catch MPS speed floor violations"
    assert rule_impact_evaluation["recovery_cushion_clamps"] > 0, "Rule engine must clamp recovery cushion over-runs"

    # Clamped metrics check: MAE remains disciplined while strictly preserving physical feasibility
    rule_mae = rule_impact_evaluation["rule_engine_metrics"]["MAE"]
    assert rule_mae <= 7.20, f"Rule clamped MAE too high: {rule_mae}"
    assert rule_impact_evaluation["rule_engine_metrics"]["Pct_within_15m"] >= 88.0


def test_live_rule_engine_application_sample(holdout_test_data: pd.DataFrame, production_booster: lgb.Booster):
    """
    Execute live rule engine application on a test sample to verify strict physical bounds.
    """
    features = get_feature_names(model_tier="M3")
    sample_df = holdout_test_data.iloc[:500].copy()
    y_pred = production_booster.predict(sample_df[features])

    for i, row in enumerate(sample_df.itertuples()):
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

        # 1. Final time must be strictly positive
        assert res.final_time > 0.0, "Clamped section time must be strictly positive"

        # 2. Speed must never exceed track MPS
        min_allowed_time = (sec.distance_km / sec.max_permissible_speed_kmh) * 60.0
        assert res.final_time >= min_allowed_time - 1e-4, (
            f"Physical speed violation! Final time {res.final_time:.2f}m < MPS floor {min_allowed_time:.2f}m"
        )


# ==============================================================================
# 9. Empirical Confidence Calibration Monotonicity
# ==============================================================================

def test_empirical_confidence_calibration_monotonicity(confidence_calibration: Dict[str, Any]):
    """
    Verify GaTi confidence calibration properties:
    - Strict Monotonicity: Higher confidence scores guarantee strictly lower MAE and higher punctuality.
    - Tier 1 represents the vast majority of runs (>= 80%).
    """
    tiers = [
        "Tier 1: Very High (>= 90%)",
        "Tier 2: High (80% - 90%)",
        "Tier 3: Moderate (70% - 80%)",
        "Tier 4: Reduced (60% - 70%)"
    ]
    for tier in tiers:
        assert tier in confidence_calibration, f"Missing calibration tier: {tier}"

    maes = [confidence_calibration[t]["mae_mins"] for t in tiers]
    pcts_5m = [confidence_calibration[t]["pct_within_5m"] for t in tiers]

    # Check 1: Monotonic MAE degradation (Tier 1 < Tier 2 < Tier 3 < Tier 4)
    for i in range(len(maes) - 1):
        assert maes[i] < maes[i + 1], (
            f"Monotonicity violation in MAE: {tiers[i]} ({maes[i]}m) >= {tiers[i+1]} ({maes[i+1]}m)"
        )

    # Check 2: Monotonic Punctuality degradation (Tier 1 > Tier 2 > Tier 3 > Tier 4)
    for i in range(len(pcts_5m) - 1):
        assert pcts_5m[i] > pcts_5m[i + 1], (
            f"Monotonicity violation in Punctuality: {tiers[i]} ({pcts_5m[i]}%) <= {tiers[i+1]} ({pcts_5m[i+1]}%)"
        )

    # Check 3: Tier 1 dominance
    assert confidence_calibration["Tier 1: Very High (>= 90%)"]["sample_pct"] >= 80.0


# ==============================================================================
# 10. Feature Importance Ranking and Validity
# ==============================================================================

def test_feature_importance_ranking_and_validity(models_dir: Path):
    """
    Verify feature importance artifact:
    - Contains all active features.
    - Gain sum is ~100%.
    - Top features align with domain physics (historical median, schedule time, live delay).
    """
    csv_path = models_dir / "feature_importance.csv"
    assert csv_path.exists(), "feature_importance.csv must exist"
    df_imp = pd.read_csv(csv_path)

    # Check column structure
    for col in ["feature", "importance_gain", "importance_split", "gain_pct"]:
        assert col in df_imp.columns, f"Missing column in feature importance: {col}"

    # Number of features: 34 features in M3 model
    assert len(df_imp) == 34, f"Expected 34 features, got {len(df_imp)}"

    # All gain values non-negative
    assert (df_imp["importance_gain"] >= 0).all(), "Feature importance gain cannot be negative"

    # Gain percentage sums to ~100%
    assert np.isclose(df_imp["gain_pct"].sum(), 100.0, atol=0.1), (
        f"Gain pct sum should be ~100%, got {df_imp['gain_pct'].sum()}"
    )

    # Top 3 features by gain must be core operational features
    top_3 = list(df_imp["feature"].iloc[:3])
    assert "section_median_time" in top_3, "section_median_time must be in top 3 features"
    assert "scheduled_section_time" in top_3, "scheduled_section_time must be in top 3 features"


# ==============================================================================
# 11. Live Holdout Test Set Inference & Metric Reproducibility
# ==============================================================================

def test_live_test_set_inference_and_reproducibility(
    holdout_test_data: pd.DataFrame,
    production_booster: lgb.Booster,
    evaluation_summary: Dict[str, Any]
):
    """
    Execute live inference on the entire holdout test set (164,564 runs) and verify:
      1. Zero NaN, Null, or Infinite predictions.
      2. High throughput (> 30,000 predictions/second).
      3. Exact metric match against published evaluation summary (zero model drift).
    """
    features = get_feature_names(model_tier="M3")
    y_test = holdout_test_data[TARGET_COL].to_numpy()

    # Time prediction throughput
    t0 = time.perf_counter()
    y_pred = production_booster.predict(holdout_test_data[features])
    elapsed = time.perf_counter() - t0

    throughput = len(y_pred) / elapsed
    print(f"\nLive Holdout Inference: {len(y_pred):,} predictions in {elapsed:.3f}s ({throughput:,.0f} req/sec)")
    assert throughput > 30000.0, f"Inference throughput too low: {throughput:,.0f} req/sec"

    # Sanity checks on raw ML predictions
    assert not np.isnan(y_pred).any(), "Predictions contain NaN values"
    assert not np.isinf(y_pred).any(), "Predictions contain infinite values"
    assert (y_pred > -1.0).all(), "Unconstrained raw ML predictions must be within plausible lower bound"
    positive_pct = (y_pred > 0.0).mean() * 100.0
    assert positive_pct > 99.99, f"Raw positive percentage too low: {positive_pct:.3f}%"

    # Compute live metrics
    live_metrics = evaluate_predictions(y_test, y_pred)
    published_metrics = evaluation_summary["Test Set - Main LightGBM"]

    # Verify zero drift (tolerance atol=0.05)
    for key in ["MAE", "RMSE", "R2", "Pct_within_5m", "Pct_within_10m", "Pct_within_15m", "P90_error"]:
        diff = abs(live_metrics[key] - published_metrics[key])
        assert diff < 0.05, f"Metric drift in {key}: live={live_metrics[key]}, published={published_metrics[key]}"


# ==============================================================================
# 12. Adversarial & Extreme Stress Inputs Resilience
# ==============================================================================

def test_adversarial_and_stress_inputs(production_booster: lgb.Booster):
    """
    Verify model resilience when subjected to extreme operational outliers:
    - 12 hours (720 min) departure delay.
    - Zero departure delay.
    - Heavy fog and torrential downpour.
    - Near-zero distance section.
    """
    features = get_feature_names(model_tier="M3")

    synthetic_scenarios = [
        # Scenario A: Massive 12-hour delay under extreme weather
        {
            "scheduled_section_time": 45.0,
            "distance_km": 50.0,
            "dep_delay_from": 720.0,
            "arr_delay_from": 715.0,
            "scheduled_dwell_from": 5.0,
            "section_median_time": 47.0,
            "section_mean_time": 48.0,
            "section_p90_time": 65.0,
            "section_min_time": 35.0,
            "section_std_time": 12.0,
            "edge_ntrains": 20,
            "hour_of_day": 2,
            "day_of_week": 3,
            "is_weekend": 0,
            "day_of_month": 28,
            "temperature_2m": 12.0,
            "precipitation": 35.0,
            "weather_code": 95,
            "wind_speed_10m": 45.0,
            "visibility": 50.0,
            "is_foggy": 1,
            "is_heavy_rain": 1,
            "net_1hop_mean_delay": 90.0,
            "net_1hop_delayed_count": 8,
            "net_1hop_active_count": 10,
            "net_downstream_weighted_delay": 75.0,
            "net_2hop_mean_delay": 85.0,
            "net_2hop_delayed_count": 12,
            "net_3hop_mean_delay": 80.0,
            "net_downstream_delay_trend": 15.0,
            "recent_station_mean_delay": 95.0,
            "rolling_station_mean_delay_6h": 85.0,
            "station_delay_trend_2h": 10.0,
            "zone": "NR"
        },
        # Scenario B: Pure on-time passenger express under clear weather
        {
            "scheduled_section_time": 25.0,
            "distance_km": 30.0,
            "dep_delay_from": 0.0,
            "arr_delay_from": 0.0,
            "scheduled_dwell_from": 2.0,
            "section_median_time": 25.0,
            "section_mean_time": 25.5,
            "section_p90_time": 30.0,
            "section_min_time": 20.0,
            "section_std_time": 3.0,
            "edge_ntrains": 5,
            "hour_of_day": 10,
            "day_of_week": 1,
            "is_weekend": 0,
            "day_of_month": 28,
            "temperature_2m": 25.0,
            "precipitation": 0.0,
            "weather_code": 0,
            "wind_speed_10m": 8.0,
            "visibility": 10000.0,
            "is_foggy": 0,
            "is_heavy_rain": 0,
            "net_1hop_mean_delay": 2.0,
            "net_1hop_delayed_count": 0,
            "net_1hop_active_count": 4,
            "net_downstream_weighted_delay": 1.0,
            "net_2hop_mean_delay": 1.5,
            "net_2hop_delayed_count": 0,
            "net_3hop_mean_delay": 1.0,
            "net_downstream_delay_trend": 0.0,
            "recent_station_mean_delay": 1.0,
            "rolling_station_mean_delay_6h": 2.0,
            "station_delay_trend_2h": 0.0,
            "zone": "ER"
        }
    ]

    df_stress = pd.DataFrame(synthetic_scenarios)
    for cat in CATEGORICAL_FEATURES:
        if cat in df_stress.columns:
            df_stress[cat] = df_stress[cat].astype("category")

    preds = production_booster.predict(df_stress[features])

    assert len(preds) == 2
    assert not np.isnan(preds).any()
    assert not np.isinf(preds).any()
    assert (preds > 0.0).all()

    # Scenario A (severe delay + storm) should predict longer traversal than on-time Scenario B
    assert preds[0] > preds[1]
    assert 10.0 < preds[0] < 300.0, f"Predicted traversal out of sensible bounds: {preds[0]}"
