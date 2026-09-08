# GaTi — Empirical Evaluation Methodology & Defense

## 1. Evaluation Philosophy & Zero-Leakage Protocol

GaTi models section running times ($t_{\text{actual}}$) and station arrival delays ($d_{\text{arrival}}$) across the Indian Railways Broad Gauge network. To provide defensible, judge-proof performance numbers, the evaluation protocol adheres strictly to real-world operational constraints:

1. **Strict Temporal Forward Split**:
   - **Training Set**: September 1 – September 22, 2024 (~935,000 section traversals)
   - **Validation Set**: September 23 – September 26, 2024 (~182,000 section traversals)
   - **Test Set**: September 27 – September 30, 2024 (~165,000 section traversals)
   - *No random K-fold shuffling*: Shuffling railway journey data leaks future congestion and weather states into historical predictions. Temporal splitting ensures the model only predicts the future from the past.

2. **Zero-Leakage Target Encoding**:
   - Historical aggregations (`section_median_time`, `section_mean_time`, `section_p90_time`, `section_min_time`, `section_std_time`) are computed **strictly from the training split**.
   - Validation and test splits receive historical statistics as lookup features. If a section is unseen in training, it falls back gracefully to `scheduled_section_time`.
   - Verified automatically by `tests/test_no_leakage.py`.

3. **Causal Horizon Replay**:
   - In simulation and inference, at station $k$, only features available prior to or at departure from station $k$ are utilized. Actual downstream events and ground truth delays are strictly withheld until post-hoc validation.

---

## 2. The Baseline Ladder

To establish fair and rigorous benchmarks, GaTi evaluates against a multi-tier baseline ladder reflecting both operational practice and statistical baselines:

| Baseline ID | Name | Operational Interpretation | Mathematical Formulation |
|:---|:---|:---|:---|
| **B0** | **Schedule Naive (Timetable)** | Nominal timetable expectation; assumes trains arrive on published schedule. | $\hat{t}_{\text{sec}} = t_{\text{sched}}$<br>$\hat{d}_{\text{arr}} = 0$ |
| **B1** | **Current Delay Propagation (NTES)** | National Train Enquiry System default; assumes current departure delay persists indefinitely ($\Delta \text{delay} = 0$). | $\hat{t}_{\text{sec}} = t_{\text{sched}}$<br>$\hat{d}_{\text{arr}} = d_{\text{dep}}$ |
| **B2** | **Historical Section Median** | Typical operational running time observed historically for that specific track section. | $\hat{t}_{\text{sec}} = \text{median}(t_{\text{train\_split}})$<br>$\hat{d}_{\text{arr}} = d_{\text{dep}} + (\hat{t} - t_{\text{sched}})$ |
| **B3** | **Ridge Linear Regression** | Linear multivariate baseline utilizing timetable, distance, live delay, and temporal features. | $\hat{t}_{\text{sec}} = \mathbf{w}^T \mathbf{x} + b$ (L2 regularized) |
| **B4** | **LightGBM Ablations** | Feature-isolated gradient boosting models (excluding weather or excluding network topology). | Evaluates marginal value of external data sources. |
| **B5** | **GaTi Main LightGBM** | 23-feature gradient boosted decision tree optimizing L1 loss (MAE). | Non-linear feature interactions across temporal, network, live delay, and ERA5 weather. |
| **B6** | **GaTi + Deterministic Rule Engine** | Physics-bounded post-ML constraint layer enforcing track MPS, TSR physics, and timetable cushion limits. | $t_{\text{final}} = \text{Rules}(\hat{t}_{\text{ML}}, \text{track MPS}, \text{TSR}, \text{cushion})$ |

---

## 3. Evaluation Metrics

Because squared error metrics (MSE/RMSE) overly penalize non-recurring operational anomalies (e.g. major signal failure delays), Indian Railways operations prioritize median absolute deviation and punctuality windows. We report:

1. **Mean Absolute Error (MAE)**: Primary optimization loss (`regression_l1`). Direct measure of average error in minutes.
2. **Root Mean Squared Error (RMSE)**: Monitors variance and large outlier penalties.
3. **Coefficient of Determination ($R^2$)**: Quantifies percentage of section variance explained.
4. **Punctuality Windows**:
   - **$\% \le \pm 5 \text{ min}$**: Indian Railways operational punctuality benchmark.
   - **$\% \le \pm 10 \text{ min}$**: Intermediate passenger arrival confidence window.
   - **$\% \le \pm 15 \text{ min}$**: Standard passenger utility expectation window.
5. **P90 Absolute Error**: 90th percentile error; quantifies the performance boundary for the worst 10% of operational edge cases.

---

## 4. Horizon Stratification (Hop Buckets)

A single overall MAE obscures error growth over multi-station journeys. GaTi partitions the held-out test set into 5 forecast horizons:

1. **1 Hop (Immediate Next Station)**: Critical for loop line precedence and junction approach dispatching.
2. **2–3 Hops (Short Range, ~50–100 km)**: Block section conflict resolution.
3. **4–5 Hops (Medium Range, ~150–250 km)**: Crew changing and locomotive servicing planning.
4. **6–10 Hops (Intermediate Range, ~300–600 km)**: Divisional handover punctuality.
5. **11+ Hops (Long-Haul Destination)**: Passenger arrival ETA and long-distance rake turnaround.

Results demonstrate that while naive schedule accumulation degrades rapidly over long horizons, GaTi maintains a disciplined error profile due to empirical median anchoring and live delay regression.

---

## 5. Scenario Bucketing (Delay Severity)

A common weakness of pure machine learning models is overfitting to the ~50% of trains running near on-time, resulting in catastrophic failure during heavy disruption. GaTi evaluates 4 operational delay regimes:

- **On-time / Nominal ($d_{\text{dep}} < 5 \text{ min}$)**: Normal traffic flow.
- **Minor Delay ($5 \le d_{\text{dep}} < 30 \text{ min}$)**: Absorbed by section margins or moderate priority conflict.
- **Severe Delay ($30 \le d_{\text{dep}} < 120 \text{ min}$)**: Major line congestion, overtaking by higher-priority trains.
- **Extreme Delay ($d_{\text{dep}} \ge 120 \text{ min}$)**: Non-scheduled platforming, out-of-slot dispatching, knock-on congestion.

In severe delay scenarios ($30\text{--}120\text{ min}$), Schedule Naive error escalates to over 12.9 minutes MAE, whereas GaTi restricts error to 7.6 minutes MAE (a 41% reduction in error).

---

## 6. The ML vs. Physical Feasibility Trade-off

A critical contribution of GaTi is demonstrating why **pure unconstrained ML is insufficient for railway deployment**:

- Unconstrained LightGBM optimizes purely for mathematical point loss (MAE) over the training distribution. In doing so, it occasionally predicts speeds exceeding track Maximum Permissible Speed (MPS) or unrealistically high recovery rates.
- The **Deterministic Rule Engine** acts as an immutable safety and physical guardrail:
  1. **MPS Floor Enforcement**: Prevents any prediction that would require speeds $> 110\text{ km/h}$ (or section track limit). Clamped over 3,000 physical violations on the test set.
  2. **15% Recovery Cushion Limit**: Prevents unrealistic timetable make-up that human section controllers cannot deliver in practice.
  3. **Temporary Speed Restriction (TSR) Integration**: Deterministically adds deceleration, restricted traversal, and acceleration time when caution orders are active.
  4. **Full Explainable Audit Trail**: Every modification outputs a numerical delta and human-readable operational reason.

---

## 7. Artifact Traceability

All metrics, ablation figures, and stratified evaluations are automatically generated and preserved in reproducible machine-readable files:
- `models/evaluation_summary.json` & `models/evaluation_summary.csv`
- `models/horizon_evaluation.json` & `models/horizon_evaluation.csv`
- `models/scenario_evaluation.json` & `models/scenario_evaluation.csv`
- `models/rule_impact_evaluation.json`
- `models/feature_importance.csv`
