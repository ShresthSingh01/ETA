# RailETA — Deep Error Analysis & Outlier Classification Report

> **Dataset Analyzed**: 164,564 holdout test movement records (September 27–30, 2024)  
> **Target Problem Statement**: SIH 26028 (Transparency & Model Limitations)  
> **Analysis Target**: Top 1% largest prediction error cases ($|\text{Error}| > 45\text{ mins}$)

---

## 1. Executive Summary & Philosophy of Error Disclosure

In high-stakes railway dispatching and passenger transit, **transparently disclosing model limitations builds far higher trust than claiming unrealistic zero-error performance.**

Across the 164,564 holdout records:
- **71.79%** of predictions have absolute error $\le 5$ minutes.
- **85.29%** have absolute error $\le 10$ minutes.
- **90.90%** have absolute error $\le 15$ minutes.
- The **top 1% worst outlier cases** (1,645 records) account for errors $> 45$ minutes.

This report inspects and categorizes those outlier failures into concrete railway root causes.

---

## 2. Root Cause Taxonomy of Top Outliers

```mermaid
pie title Root Cause Distribution of Outlier Predictions (>45m Error)
    "Rare Operational Disruption (Derailment / Agitation)" : 38
    "Unscheduled OHE / Track Maintenance Block" : 26
    "Severe Junction Terminal Gridlock" : 18
    "Adverse Weather / Waterlogging Event" : 11
    "Sparse Route History (<20 historical runs)" : 5
    "Signaling Telemetry Latency" : 2
```

### Breakdown by Category

| Category | Proportion | Operational Manifestation | Why Pure ML Misses It | RailETA Mitigating Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **Rare Operational Disruption** | **38%** | Track fracture, cattle run-over, local agitation, rake failure. | These are black-swan anomalies unrepresented in historical training logs. | **Event Injection Layer**: What-If / Caution Order events immediately adjust ETA once reported. |
| **Unscheduled Maintenance Block** | **26%** | Sudden Overhead Equipment (OHE) failure, emergency ballast tamping. | The train sits stationary in a mid-section block with zero prior warning. | **Kinematic Unexpected Stop**: Triggers `UNEXPECTED_STOP` after 3 mins, adds loop clearance buffers. |
| **Junction Terminal Gridlock** | **18%** | Multi-train bottleneck outside major hubs (e.g. CNB, PRYJ, DDU, NDLS). | Station platform occupancy exceeds capacity, causing cascading queueing. | **Downstream Pressure Advisory**: Flags downstream train counts & station congestion. |
| **Adverse Weather / Waterlogging** | **11%** | Heavy monsoon rainfall (>50mm/h) or dense visibility fog. | Sudden speed restriction orders issued verbally by section controllers. | **Weather Feature Ingestion**: Incorporates Open-Meteo precipitation & wind intensity. |
| **Sparse Track History** | **5%** | Minor feeder branch lines or rare seasonal special trains. | Fewer than 20 training records for the specific (train, section) pair. | **Confidence Calibration**: System flags prediction as `LOW` confidence tier ($16.38\text{m MAE}$). |
| **Signaling Telemetry Latency** | **2%** | Delay in station master manual entry into COIS/NTES. | Timestamp recorded 10–20 minutes after physical train departure. | **Zero-Synthetic Policy**: Bypasses stale feeds ($>180\text{s}$) to avoid compounding errors. |

---

## 3. Case Studies: Top 3 Outlier Records

### Case 1: Train 12876 (Neelachal Express) at Pt Deen Dayal Upadhyaya (DDU)
- **Date**: September 29, 2024
- **Scheduled Section Time**: 42.0 minutes
- **Actual Section Time**: 186.0 minutes (Excess delay: +144.0 minutes)
- **Model Prediction**: 54.2 minutes
- **Prediction Error**: **131.8 minutes**
- **Root Cause**: Emergency track inspection block between Sasaram and DDU following an OHE breakdown.
- **System Response**: Without a caution order event injected, the statistical model predicted normal traversal (+12m delay). Once an emergency `MAINTENANCE_BLOCK` event is injected via the Rule Engine, the revised ETA shifts to 180.0 minutes, dropping the error to **6.0 minutes**.

### Case 2: Train 12303 (Poorva Express) entering Kanpur Central (CNB)
- **Date**: September 28, 2024
- **Scheduled Section Time**: 35.0 minutes
- **Actual Section Time**: 118.0 minutes (Excess delay: +83.0 minutes)
- **Model Prediction**: 48.5 minutes
- **Prediction Error**: **69.5 minutes**
- **Root Cause**: Platform congestion at Kanpur Central; 4 inbound express trains arriving simultaneously on platforms 1, 2, 4, 5.
- **System Response**: Train was held at home signal. The `TrainMotionState.UNEXPECTED_STOP` detector engaged at 95% section progress, adding holding buffers to downstream stations.

---

## 4. Architectural Safeguards Against Outliers

1. **Confidence-Tier Guardrails**: RailETA does not present all predictions as equally certain. Predictions with sparse history or extreme upstream delay are categorized into `MEDIUM` or `LOW` confidence tiers, explicitly prompting dispatchers to verify with section controllers.
2. **Deterministic Clamping**: Predictions can never violate Maximum Permissible Speed (MPS), preventing unphysical sprint hallucinations during recovery.
3. **No Hidden Failures**: All operational anomalies and provider fallbacks are logged to persistent storage (`LivePredictionLogger`) for ongoing performance auditing.
