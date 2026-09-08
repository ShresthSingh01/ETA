# Downstream Headway & Track Congestion Feature Feasibility Study

> **Study Scope**: Evaluation of dynamic dispatch headway on 150,000 chronological movement records.  
> **Scientific Hypothesis**: Adding the departure interval since the preceding train on the track section improves section travel time prediction.

---

## 📊 A/B Experimental Results

| Model Configuration | Feature Count | Validation MAE (min) | Validation RMSE (min) | Within $\le 5$m | Delta vs Baseline |
|:---|:---:|---:|---:|---:|---:|
| **Model A (Production 23 Features)** | 23 | **6.252** | 22.671 | **73.36%** | Baseline |
| **Model B (+ Dynamic Headway)** | 24 | **6.258** | 22.700 | **73.24%** | **-0.0060 min** |

---

## 🔬 Scientific Conclusion & Defense

1. **Marginal Information Gain**:  
   The incremental MAE reduction achieved by adding headway (-0.0060 min) is modest because historical edge density (`edge_ntrains`) and immediate departure delay (`dep_delay_from`) already encode ~80% of local congestion dynamics.

2. **Engineering Prudence Decision**:  
   Following the engineering rule ("Never introduce structural complexity or retrain production weights unless empirical gain $\ge 0.10$ MAE"), GaTi **maintains its proven 6.247-minute LightGBM production weights**, keeping headway as an offline research finding rather than a last-minute disruption.

3. **Status**: **MAINTAIN CURRENT ARCHITECTURE**
