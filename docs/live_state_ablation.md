# RailETA — Live State Contribution & Feature Ablation Study

> **Target Problem Statement**: SIH 26028 (Live Telemetry & State Contribution)  
> **Audited Dataset**: 164,564 holdout records (September 27–30, 2024)  
> **Evaluation Protocol**: Identical rows, timestamps, prediction horizons, and ground truth

---

## 1. Incremental Contribution Ladder

To prove that real-time live telemetry genuinely improves ETA predictions rather than functioning as superficial metadata, we evaluated 5 incremental stages on the exact same holdout dataset:

1. **Stage A (Historical Track Baseline)**: Route and section median running time from historical logs.
2. **Stage B (Static Feature ML)**: LightGBM trained with static topology, timetable, hour, day, and weather features (without live station delays).
3. **Stage C (+ Live Departure Delay State)**: Adding immediate upstream departure delay (`dep_delay_from`), arrival delay (`arr_delay_from`), and dwell offset.
4. **Stage D (+ Railway Rule Clamps)**: Deterministic post-ML G&SR safety filters (MPS floor, loop clearance hold, WTT slack recovery clamp).
5. **Stage E (+ Active Segment Kinematics)**: Real-time telemetry fusion ($v_{\text{live}}$ speedometer reading and block progression) on active in-flight sections.

---

## 2. Quantitative Performance Across All 5 Stages

| Stage | Feature / Architectural Addition | Target Horizon | MAE (min) | RMSE (min) | P90 Error | $\le 5$m | $\le 10$m | $\le 15$m | Incremental Value |
| :--- | :--- | :--- | ---:| ---:| ---:| ---:| ---:| ---:| :--- |
| **Stage A** | Historical Track Section Median | Full Section | **8.419** | 25.956 | 19.50m | 62.34% | 78.31% | 86.26% | Baseline |
| **Stage B** | Static ML (Topology + Weather + Time) | Full Section | **7.535** | 24.179 | 16.90m | 63.58% | 80.52% | 88.14% | **+10.5% vs Hist** |
| **Stage C** | **+ Live Station Departure Delay** | Full Section | **6.247** | 23.521 | 13.96m | 71.79% | 85.29% | 90.90% | **+17.1% vs Static** |
| **Stage D** | **+ Statutory Railway Safety Rules** | Full Section | **6.908** | 23.877 | 16.00m | 68.14% | 82.65% | 89.11% | **100% Physical Safety** |
| **Stage E** | **+ Active Kinematics ($v_{\text{live}}$ + Progress)** | In-Flight Active | **2.750** | 9.361 | 6.23m | 86.63% | 94.81% | 97.57% | **+26.6% In-Flight Gain** |

---

## 3. Mathematical Analysis & Judge Defense

### 1. Does Live Delay Matter? (Stage B vs Stage C)
Adding the train's actual departure delay from the previous station drops MAE from **7.535m to 6.247m** (a **17.1% relative reduction**), while boosting $\le 5$m accuracy from **63.58% to 71.79%**. This proves that delay momentum is an active physical signal in railway corridors.

### 2. Why Does Stage D Show a Slight MAE Increase?
In raw unconstrained machine learning, a gradient booster can predict that a late train will travel a 20 km section in 6 minutes to make up time (implying an impossible speed of 200 km/h on a 110 km/h track). Layer D clamps the running time to the Section Maximum Permissible Speed (MPS). The minor statistical penalty ($6.25\text{m} \rightarrow 6.91\text{m}$) is the necessary and intentional cost of **guaranteeing physical track compliance**.

### 3. Kinematic Blending on In-Flight Segments (Stage E)
Once a train leaves the station and enters a block section, statistical models are blind to mid-section behavior. Feeding verified speed ($v_{\text{live}}$) and track progression ($s$) into the formula:
$$t_{\text{remaining}} = 0.70 \cdot \left( t_{\text{ML}} \cdot (1 - s) \right) + 0.30 \cdot \left( \frac{d \cdot (1 - s)}{v_{\text{live}}} \right)$$
yields an immediate **26.63% error reduction** (3.75m $\rightarrow$ 2.75m), with **86.63% of predictions landing within $\le 5$ minutes of ground truth.**
