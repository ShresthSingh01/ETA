# GaTi — Final Research Paper Integration Notes
## RSTGCN: Railway-centric Spatio-Temporal Graph Convolutional Network for Train Delay Prediction

---

# 1. Purpose

This document captures the important ideas, evidence, limitations, and practical integration strategy from the research paper:

> “RSTGCN: Railway-centric Spatio-Temporal Graph Convolutional Network for Train Delay Prediction”

The goal is **not** to copy the paper blindly or replace GaTi with RSTGCN.

The goal is:

> **Use the paper's network-delay insight to strengthen GaTi's existing network representation without unnecessary complexity or unsupported claims.**

Final recommendation:

> **Upgrade GaTi from basic/static network features toward a dynamic, train-relative downstream network state. Keep the current individual-train section-time LightGBM as the core model. Only introduce a full graph model if controlled experiments prove that it is necessary.**

---

# 2. The paper in one sentence

The paper argues and demonstrates that:

> **Railway delays are not isolated events; they exhibit temporal patterns and spatial relationships across connected stations, and those relationships can be useful for forecasting future railway delay.**

That is the main insight we should borrow.

---

# 3. What problem does the paper actually solve?

This distinction is critical.

The paper does **not** solve individual-train ETA.

Its target is:

> **Average arrival delay at each station over a future time horizon.**

The paper defines a railway network as a graph containing stations, direct track connections, connectivity, distance and train-frequency information.

Its prediction target is future **station-level average arrival delay**.

Conceptually:

```text
PAPER

Railway Network
      ↓
Station-level historical state
      ↓
RSTGCN
      ↓
Future station-average delay
```

GaTi instead solves:

```text
GATI

Individual train
      ↓
Current train state
      ↓
Remaining route
      ↓
Future sectional travel times
      ↓
Station-wise ETA
```

These problems are related, but they have different targets.

---

# 4. Why the paper is relevant to GaTi

The paper provides evidence for an important idea:

> **The future state of a train can be influenced by what is happening elsewhere on the railway network.**

A train can currently be on time while the railway ahead is experiencing increasing delay pressure.

Example:

```text
                 OUR TRAIN
                     ↓
A ───── B ───── C ───── D ───── E

                    C:
              many delayed trains
```

If only the train's own current state is considered, information at C/D can be missed.

The paper motivates the idea that spatial and temporal railway conditions can provide predictive information.

---

# 5. What the paper adds beyond a simple train count

The paper does not rely only on:

```text
number of trains
```

It introduces and uses railway-specific features such as:

- hourly average arrival delay
- hourly average departure delay
- total hourly arrival delay
- total hourly departure delay
- hourly headway
- train frequency

It also considers temporal history at different scales.

Therefore:

```text
ntrains = 8
```

is much weaker than:

```text
8 active trains
6 delayed
mean arrival delay = +17 min
departure delay = +13 min
recent trend = increasing
headway = compressed
```

Those represent different operational situations.

---

# 6. The paper's temporal insight

The paper uses three temporal views:

```text
RECENT
DAILY
WEEKLY
```

### Recent

What has been happening during the immediately preceding time period.

### Daily

What tends to happen around the same time on another day.

### Weekly

What tends to happen around the corresponding time in the previous week.

The underlying idea is:

> Railway traffic and delays contain recurring temporal patterns.

This can be adapted to GaTi without adopting the full RSTGCN architecture.

---

# 7. The paper's spatial insight

The paper treats stations as nodes connected by railway edges.

Conceptually:

```text
A ─── B ─── C ─── D
      │
      E
```

It argues that delays at connected/nearby stations can be correlated.

Therefore:

```text
Delay at B
    ↓
future conditions at C/D may be affected
```

The exact propagation is not guaranteed, but the network contains predictive information.

This is the strongest reason to strengthen GaTi's network layer.

---

# 8. What RSTGCN does

RSTGCN means:

> **Railway-centric Spatio-Temporal Graph Convolutional Network**

It combines:

```text
spatial railway structure
+
temporal delay history
+
railway-specific features
```

and forecasts future station-level average delays.

It is substantially more complex than the current GaTi LightGBM approach.

---

# 9. What the paper demonstrates experimentally

The paper reports that RSTGCN outperforms its evaluated comparison models, with improvements of up to:

```text
18% MAE
14% MAPE
1–8% RMSE
```

It also evaluates multiple forecasting horizons and delay thresholds.

This supports:

\[
oxed{
	ext{Network-aware spatio-temporal information can be useful for railway delay forecasting.}
}
\]

But it does **not** establish that the same percentage improvement will occur in GaTi.

---

# 10. What the paper does NOT prove

Do not claim:

> “The paper proves GaTi's ETA accuracy.”

It does not.

Do not claim:

> “RSTGCN directly predicts individual-train ETA.”

It does not.

Do not claim:

> “GaTi must use an RSTGCN.”

The paper does not establish that.

Do not claim:

> “The paper's 18% improvement will transfer to our model.”

Not justified.

The defensible statement is:

> **The paper provides evidence that railway delay contains spatial and temporal dependencies. We adapt those ideas to our different target: individual-train sectional ETA forecasting.**

---

# 11. GaTi already has a network layer

This is important.

GaTi already includes:

```text
railway topology
+
network edges
+
train activity
+
network-related ML features
```

and an `edge_ntrains`-type feature.

Therefore:

> **The network is not missing. The current network representation is simply not strong enough.**

The problem is not:

```text
NO NETWORK
```

The problem is:

```text
LIMITED NETWORK STATE
```

---

# 12. The current weakness

Consider:

```text
edge_ntrains = 8
```

This does not tell the model whether those 8 trains are:

```text
mostly on time
```

or:

```text
mostly delayed
```

or:

```text
delays are increasing
```

or:

```text
station traffic is becoming compressed
```

Therefore the upgrade should be:

> **Represent what the network is currently doing, not just how much network activity exists.**

---

# 13. The exact upgrade: Downstream Network State Engine

Create a lightweight component:

> **Downstream Network State Engine**

Its job:

> Given the current train and its remaining route, summarize the current state of the railway ahead.

Example:

```text
OUR TRAIN
    ↓
A → B → C → D → E

Network state ahead:

B:
active trains = 4
delayed = 1
mean delay = +5

C:
active trains = 7
delayed = 5
mean delay = +17

D:
active trains = 6
delayed = 4
mean delay = +15
```

This creates a much richer description of the future environment.

---

# 14. Recommended network features

Start small.

### Downstream station features

```text
downstream_active_train_count
downstream_delayed_train_count
downstream_mean_arrival_delay
downstream_mean_departure_delay
downstream_total_arrival_delay
downstream_total_departure_delay
downstream_p90_delay
```

### Temporal features

```text
recent_downstream_delay
same_hour_previous_day_delay
same_hour_previous_week_delay
downstream_delay_trend
```

### Traffic features

```text
train_frequency
headway
```

### Train-relative route features

```text
stations_ahead
sections_ahead
distance_ahead
estimated_minutes_ahead
```

Do not add dozens of features before measuring whether the small set helps.

---

# 15. Why train-relative network context is better for GaTi

The paper asks:

> What will be the average delay at station C?

GaTi asks:

> What will Train 12919 experience when it reaches C?

Therefore:

```text
station C mean delay = +15
```

is not enough.

We want:

```text
C is 3 sections ahead
C has high delay pressure
C has 5 delayed trains
delay trend is increasing
```

This context is tied to the specific train's future route.

---

# 16. Multi-hop context

For:

```text
A → B → C → D → E
```

and the train currently at A, represent:

```text
1-hop network state
2-hop network state
3-hop network state
4-hop network state
```

For example:

```text
network_1hop_delay
network_2hop_delay
network_3hop_delay
```

This lets the model distinguish:

```text
problem immediately ahead
```

from:

```text
problem several sections ahead
```

which matters for ETA.

---

# 17. Use live data to make the network state dynamic

The strongest version uses real observations:

```text
RailRadar
   ↓
Current states of trains/stations
   ↓
Map onto railway graph
   ↓
Calculate downstream network state
```

The result becomes:

> **Current network state around and ahead of this particular train.**

That is much closer to dynamic network intelligence than a static `ntrains` count.

---

# 18. Do not use the target train's future information

If prediction is being made at time `t`, allowed data is:

```text
your train observations ≤ t
other-train observations ≤ t
historical information ≤ t
weather information available by t
known operational events ≤ t
```

Do not use:

```text
your train's future arrival
future delay
future station conditions caused by the future journey
```

That would create leakage.

The paper's use of historical windows before the prediction time reinforces this principle.

---

# 19. Use the paper's temporal idea without adopting RSTGCN

A lightweight implementation can create:

```text
recent_network_delay
same_hour_previous_day_network_delay
same_hour_previous_week_network_delay
```

provided all values are computed without future leakage.

Then the network representation becomes:

```text
network now
+
network recently
+
recurring network pattern
```

This may be more useful than simply adding a headway column.

---

# 20. Your existing headway experiment is important

GaTi already performed a controlled headway test:

```text
Production model:
6.252 min MAE

+ Dynamic headway:
6.258 min MAE
```

The difference was negligible.

Therefore:

> **Do not simply add headway again and claim the network problem is solved.**

Instead, test headway together with richer delay state and train activity.

The previous experiment is useful evidence that a single traffic feature does not necessarily provide enough incremental information.

---

# 21. The correct ML architecture

Current:

\[
T_i=f(X_i)
\]

where `X_i` contains existing train/section/time/weather/network information.

Upgraded:

\[
T_i=f(X_i,N_i)
\]

where:

- \(X_i\) = existing GaTi features
- \(N_i\) = dynamic downstream network state

Conceptually:

```text
Current train
      ↓
Remaining route
      ↓
Historical section behaviour ─┐
Weather ──────────────────────┤
Train context ────────────────┤
Current state ────────────────┤
Downstream network state ─────┤
                               ↓
                           LightGBM
                               ↓
                     Section travel time
```

This keeps the existing model architecture intact.

---

# 22. Why we should NOT replace LightGBM immediately

GaTi's target is:

```text
individual-train sectional travel time
```

The paper's target is:

```text
station-level average arrival delay
```

The paper itself formally defines the latter.

Therefore RSTGCN is not a drop-in replacement.

Replacing the model would add complexity without proving that the new target is served better.

---

# 23. The best experiment

Run four controlled versions.

### M0 — Current production model

```text
existing 23 features
```

### M1 — Basic downstream state

```text
M0
+
downstream delay
+
active train count
+
delayed train count
```

### M2 — Multi-hop network context

```text
M1
+
1-hop/2-hop/3-hop state
+
delay trend
```

### M3 — Multi-scale temporal network state

```text
M2
+
recent
+
daily
+
weekly network history
```

Evaluate all on:

```text
same records
same timestamps
same target
same horizon
same ground truth
```

---

# 24. Metrics

Use:

```text
MAE
RMSE
P90 absolute error
within ±5 min
within ±10 min
within ±15 min
```

Also report:

```text
immediate next station
mid-route stations
destination
```

This ensures the network feature is improving the actual ETA task.

---

# 25. The most interesting evaluation: difficult network conditions

Do not only look at overall MAE.

Create buckets:

```text
NORMAL
LOW NETWORK PRESSURE
MEDIUM NETWORK PRESSURE
HIGH NETWORK PRESSURE
```

Then compare:

```text
Current model
vs
Network-enhanced model
```

Network features may be most useful when the railway is under stress.

Example:

```text
Normal:
6.0 → 5.9

High pressure:
12.0 → 8.8
```

Such an improvement can be more operationally important than a tiny overall MAE change.

---

# 26. Keep the operational event layer separate

Network prediction should not replace your rule/event system.

Maintain:

```text
Network model
"What is likely to happen?"
```

and:

```text
Event engine
"What disruption is currently known?"
```

and:

```text
Rule engine
"What constraints must be respected?"
```

Then:

```text
network prediction
+
live train state
+
operational event
+
deterministic constraints
→
ETA
```

This remains the strongest overall architecture.

---

# 27. Why the paper's limitation matters

The paper itself reports that its one-month dataset contains limited highly irregular cases and that performance degrades in sudden/unforeseen disruptions such as system failures.

This supports our architectural separation:

```text
ML / network model
        +
explicit operational events
        +
deterministic constraints
```

A graph model should not be expected to magically know about every rare operational disruption.

---

# 28. Final architecture after the upgrade

```text
                        RAILRADAR
                            │
                            ▼
                   Current Train State
                            │
                            ▼
                     Remaining Route
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
    Historical Section              Downstream Network
       Behaviour                         State
             │                             │
             │                    ┌────────┼────────┐
             │                    ▼        ▼        ▼
             │                  1-hop    2-hop    3-hop
             │                  state    state    state
             │
             └──────────────┬──────────────────────┘
                            ▼
                         LightGBM
                            │
                            ▼
                 Section Travel-Time Forecast
                            │
                            ▼
                    Kinematic Correction
                            │
                            ▼
                    Operational Events
                            │
                            ▼
                     Railway Rule Engine
                            │
                            ▼
                        ETA Engine
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
          Station-wise ETA       Destination ETA
                 │
                 ▼
             Confidence
                 │
                 ▼
          Actual Arrival
                 │
                 ▼
          Self Evaluation
```

---

# 29. What the paper strengthens specifically

Before:

```text
Network
   ↓
basic network features
```

After applying the paper's insight:

```text
Network
   ↓
spatial relationships
+
temporal evolution
+
downstream delay state
+
train activity
+
route-relative context
   ↓
richer ETA features
```

The improvement is therefore primarily in **representation**, not in adding a fashionable model.

---

# 30. What to say in the SIH presentation

A concise defensible statement:

> **“Recent Indian Railway research shows that railway delays exhibit spatial and temporal dependencies. Inspired by that finding, GaTi represents the evolving state of the network ahead of a train using downstream train activity, delays and temporal trends, while keeping our individual-train sectional ETA model. We validate these additions experimentally rather than assuming that a graph model will automatically improve ETA.”**

---

# 31. What to say if a judge asks why you didn't use RSTGCN

> **“RSTGCN addresses a different target—station-level average delay forecasting. Our primary target is individual-train sectional travel time. We therefore adapt the useful network signals first. If a full graph model demonstrates a measurable improvement on our ETA target, we can introduce it later. We avoid complexity unless the data justifies it.”**

---

# 32. Final decision rule

Use this:

```text
Paper insight
     ↓
Network-state features
     ↓
Controlled ablation
     ↓
Does ETA improve?
```

### YES

```text
Keep network enhancement
```

### NO

```text
Do not force it
```

### Significant and consistent YES

```text
Consider a graph model as a research extension
```

---

# 33. Final takeaway

The RSTGCN paper is valuable because it gives a strong, domain-specific basis for one statement:

\[
oxed{
	ext{A railway's current spatial and temporal state contains information about future delay.}
}
\]

GaTi already has a network layer, so we do **not** need to build another network system.

We should strengthen what already exists:

\[
oxed{
	ext{Basic Network Features}

ightarrow
	ext{Dynamic Downstream Network State}
}
\]

using:

```text
downstream delays
active/delayed trains
arrival/departure delay
delay trends
multi-hop context
train frequency
headway
recent/daily/weekly network history
```

Then:

\[
oxed{
	ext{Train State}
+
	ext{Section History}
+
	ext{Downstream Network State}
+
	ext{Operational Constraints}

ightarrow
	ext{Dynamic Individual-Train ETA}
}
\]

This is the strongest way to use the paper in GaTi because it is:

- scientifically motivated,
- aligned with the actual PS,
- simpler than a full RSTGCN implementation,
- compatible with your current system,
- testable,
- and honest about what the paper does and does not prove.
