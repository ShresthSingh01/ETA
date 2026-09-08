# GaTi — Comprehensive Data Leakage Audit

## 1. Executive Summary

Data leakage is the most prevalent methodological flaw in machine learning applications for public transit and railway networks. When future conditions, future delays, or target values bleed into training features, models report artificially stellar metrics in offline benchmarks but fail catastrophically when deployed into live control rooms.

GaTi enforces an **immutable, mathematically verifiable Zero-Leakage Protocol** across every layer of the feature pipeline, model training, and simulation replay.

---

## 2. Leakage Vectors & Mitigations

### Vector 1: Temporal Contamination (Random K-Fold Splitting)
- **The Risk**: Shuffling section rows across random cross-validation folds mixes past and future days. A model predicting a train on September 15 could learn from congestion patterns and weather events that occurred on September 28.
- **GaTi Mitigation**: Strict chronological forward partitioning:
  - **Train**: September 1 – September 22, 2024 (72.9% of records)
  - **Validation**: September 23 – September 26, 2024 (14.2% of records)
  - **Test**: September 27 – September 30, 2024 (12.8% of records)
  - Validation and test data are completely unseen during model training and hyperparameter tuning.

---

### Vector 2: Target Leakage via Historical Aggregations
- **The Risk**: Computing historical statistics (`section_median_time`, `section_p90_time`, etc.) across the full dataset allows validation and test target values to influence the historical feature prior.
- **GaTi Mitigation**:
  - Historical aggregations are computed **strictly from the training split (Sep 1–22)**.
  - The lookup table maps `{from_station}_{to_station}` to historical statistics computed solely on training rows.
  - Validation and test splits receive these statistics purely as frozen prior lookups. If a section was never traversed during the training split, it defaults strictly to `scheduled_section_time`.

---

### Vector 3: Look-Ahead Feature Contamination
- **The Risk**: Using destination departure or arrival delay (`dep_delay_to`, `arr_delay_to`) or future weather when predicting section running time.
- **GaTi Mitigation**:
  - The feature vector for section traversal $(k \rightarrow k+1)$ contains only state available at the instant of departure from station $k$: `dep_delay_from`, `arr_delay_from`, scheduled parameters, historical priors, and the current hour's weather.
  - Destination delays (`arr_delay_to`, `dep_delay_to`) are strictly reserved as ground-truth evaluation targets and are completely excluded from the feature column list.

---

### Vector 4: Forward Trajectory Horizon Leakage
- **The Risk**: When predicting multi-hop journey ETAs ($k+1, k+2, \dots, k+n$), feeding future observed delays or downstream speed restrictions before the train arrives there.
- **GaTi Mitigation**:
  - In `src/engine/eta_calculator.py`, the trajectory accumulator begins strictly at `current_clock_mins` and `current_dep_delay_mins`.
  - Downstream arrival delays are recursively predicted forward based on model output:
    $$\hat{d}_{\text{arr}}^{(j)} = \hat{d}_{\text{dep}}^{(j-1)} + (\hat{t}_{\text{sec}}^{(j)} - t_{\text{sched}}^{(j)})$$
  - No future actual arrival delays are ever injected into intermediate steps.

---

## 3. Automated Verification Test Suite

Zero data leakage is continuously verified by `tests/test_no_leakage.py` as part of the automated test suite:

1. **`test_temporal_split_order`**: Verifies `train_date_max < val_date_min` and `val_date_max < test_date_min`.
2. **`test_historical_aggregations_leakage_free`**: Verifies that recomputing `section_median_time` on the combined dataset yields different values than the frozen training priors, confirming no future values contaminated the table.
3. **`test_feature_columns_contain_no_target_information`**: Verifies that neither `actual_section_time`, `arr_delay_to`, `dep_delay_to`, nor `delay_change` appear in `get_feature_names()`.
4. **`test_simulation_clock_monotonicity`**: Verifies that simulated journey progression respects strict causality and never reads future steps.
