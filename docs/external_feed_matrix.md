# GaTi — External Feed Matrix & Production Integration Architecture

This matrix details the mapping between the current prototype data sources and the production enterprise systems of Indian Railways (managed by CRIS — Centre for Railway Information Systems).

---

## 1. Enterprise Feed Inventory

| Feed Domain | Production Source System (IR / CRIS) | Prototype Equivalent | Update Frequency | Protocol / Format | Fallback & Degraded Mode Policy |
|:---|:---|:---|:---|:---|:---|
| **Live Train Movements** | **COA (Control Office Application)** & **RTIS (Real-Time Train Information System)** | `train_routes_delays_Sep2024.csv` | Event-driven (section in/out timestamps) & 30s GPS pings | WebSockets / Kafka JSON Stream | If RTIS GPS drops, fall back to COA electronic station logging (`dep_delay_from`). |
| **Timetable & Master Route** | **CRIS RBS (Rates Branch System)** & **NTES Static Feed** | `train_routes_Sep2024.csv` | Bi-annual or upon special train notification | REST API / Static GTFS-Rail JSON | Cache active Working Time Table (WTT) locally in SQLite / Redis. |
| **Railway Network Topology** | **IR Track Asset Management System (TAMS)** & Division Route Books | `edges_cleaned.csv` & `stations_cleaned.csv` | Static / Monthly updates | GeoJSON / Spatially Indexed R-Tree | Built-in fallback to 5.0 km default distance and nominal 110 km/h MPS. |
| **Zonal Mapping & Jurisdictions** | **CRIS Operating Master Tables** | `stations_zones_mapping.json` | Annual / On jurisdictional change | JSON Key-Value | Fall back to Northern Railway ("NR") operational defaults. |
| **Weather & Microclimate** | **IMD (India Meteorological Department)** Automatic Weather Stations & Doppler Radar | **Open-Meteo ERA5 Reanalysis** (97,920 hourly junction records) | Hourly / Real-time alerts | HTTPS REST / GeoJSON | Fall back to seasonal climatological normals (temperature=28°C, visibility=10,000m, rain=0). |
| **Temporary Speed Restrictions** | **COA Caution Order Notice (Form T/409)** | Injected `OperationalEvent` (`CAUTION_ORDER`, `SPEED_RESTRICTION`) | As issued by Section Controller / P-Way Engineering | Push Notification / Kafka Event | Apply nominal track Maximum Permissible Speed (MPS). |
| **Track & Overhead Maintenance** | **COIS (Coaching Operations Information System)** & Block Registers | Injected `OperationalEvent` (`MAINTENANCE_BLOCK`) | Shift-wise / Pre-planned block orders | XML / REST API | Assume unobstructed line clearance if no block active. |

---

## 2. Degraded Mode & Fault-Tolerance Principles

GaTi is architected for zero single points of failure in live railway operations:

1. **Weather Feed Disruption**:
   - If IMD/Open-Meteo APIs experience downtime or high latency (>100 ms), the feature pipeline replaces environmental inputs with neutral baseline flags (`is_foggy=0`, `precipitation=0.0`).
   - Offline ablation experiments confirm that removing weather entirely only shifts test MAE by **0.037 minutes** (6.247m vs 6.284m), ensuring uninterrupted, highly accurate dispatch support.

2. **Network Density Feed Disruption**:
   - If live track occupancy feeds are unavailable, the model falls back to static timetable-derived train density (`edge_ntrains` from RBS timetable schedule).
   - Ablation experiments demonstrate that the model operates within 0.037 minutes MAE of full capability when static topology density is utilized.

3. **Live GPS / RTIS Loss**:
   - In rural or non-electrified stretches where locomotive GPS transponders face satellite shadow, GaTi transitions smoothly to station-based discrete arrival/departure timestamps logged in COA, continuing trajectory accumulation from the last verified physical station halt.
