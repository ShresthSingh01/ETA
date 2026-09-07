# RailETA — Data Dictionary & Schema Specification

This document provides the definitive data dictionary for the canonical training and inference dataset `section_runs_weather.parquet` (1,282,325 records) and auxiliary network tables.

---

## 1. Primary Dataset: `section_runs_weather.parquet`

The primary unit of analysis is a **Section Traversal**: a single train traveling between two consecutive timetable stations on a specific date.

### Identification & Metadata
| Column | Type | Description | Source File / Derivation | Null Policy |
|:---|:---|:---|:---|:---|
| `section_run_id` | `string` | Unique identifier formatted as `{train}_{date}_{from}_{to}` | Concatenation | Mandatory |
| `train_number` | `int32` | 5-digit Indian Railways train number | `train_routes_delays_Sep2024.csv` | Mandatory |
| `train_name` | `string` | Train service name (e.g. "Poorva Express") | `train_routes_Sep2024.csv` | Defaults to "Express" |
| `date` | `string` | Date of origin departure (YYYY-MM-DD) | `train_routes_delays_Sep2024.csv` | Mandatory |
| `from_station` | `string` | Station code of departure station (e.g. "HWH") | `train_routes_delays_Sep2024.csv` | Mandatory |
| `to_station` | `string` | Station code of arrival station (e.g. "BWN") | `train_routes_delays_Sep2024.csv` | Mandatory |
| `zone` | `category` | Zonal railway code (e.g. "ER", "NR", "NCR") | `stations_zones_mapping.json` | Defaults to "NR" |
| `split` | `string` | Dataset partition: `'train'`, `'val'`, or `'test'` | Temporal split rule | Mandatory |

---

### Timetable & Physical Topology
| Column | Type | Units | Description | Source File / Derivation |
|:---|:---|:---|:---|:---|
| `distance_km` | `float64` | Kilometers | Physical rail distance between stations | Timetable cumulative distance delta; fallback to IRN cleaned edges; fallback 5 km |
| `scheduled_section_time` | `float64` | Minutes | Timetable allotted running time | `(sch_arr_to - sch_dep_from) mod 1440` (minimum 1.0 min) |
| `scheduled_dwell_from` | `float64` | Minutes | Timetable halt duration at origin station | `(sch_dep_from - sch_arr_from) mod 1440` |

---

### Target & Ground Truth Traversal
| Column | Type | Units | Description | Source File / Derivation |
|:---|:---|:---|:---|:---|
| `actual_section_time` | `float64` | Minutes | **PRIMARY ML TARGET**: Elapsed travel time on section | `max(scheduled_section_time + delay_change, 1.0)` |
| `actual_dwell_from` | `float64` | Minutes | Actual recorded halt duration at origin station | `max(scheduled_dwell_from + (dep_delay_from - arr_delay_from), 0.0)` |
| `delay_change` | `float64` | Minutes | Delay accumulated or recovered on section | `arr_delay_to - dep_delay_from` |

---

### Real-Time Live State
| Column | Type | Units | Description | Source File / Derivation |
|:---|:---|:---|:---|:---|
| `arr_delay_from` | `float64` | Minutes | Arrival delay at departure station | `train_routes_delays_Sep2024.csv` |
| `dep_delay_from` | `float64` | Minutes | Departure delay at departure station | `train_routes_delays_Sep2024.csv` |
| `arr_delay_to` | `float64` | Minutes | Actual recorded arrival delay at destination | `train_routes_delays_Sep2024.csv` (used only for validation) |
| `dep_delay_to` | `float64` | Minutes | Actual recorded departure delay at destination | `train_routes_delays_Sep2024.csv` (used only for validation) |

---

### Temporal Attributes
| Column | Type | Range | Description | Derivation |
|:---|:---|:---|:---|:---|
| `hour_of_day` | `int32` | 0 – 23 | Hour of departure from station | `(act_dep_from_mins // 60) mod 24` |
| `day_of_week` | `int32` | 0 – 6 | Day of week (0 = Monday, 6 = Sunday) | Derived from `date` |
| `is_weekend` | `int32` | 0 or 1 | 1 if Saturday or Sunday, else 0 | `day_of_week in [5, 6]` |
| `day_of_month` | `int32` | 1 – 30 | Day of calendar month | Derived from `date` |

---

### Historical Section Aggregations (Zero-Leakage)
*All historical metrics are computed strictly on the training partition (Sep 1–22) and joined as historical priors.*
| Column | Type | Units | Description |
|:---|:---|:---|:---|
| `section_median_time` | `float64` | Minutes | Median actual travel time for this section in training split |
| `section_mean_time` | `float64` | Minutes | Mean actual travel time for this section in training split |
| `section_p90_time` | `float64` | Minutes | 90th percentile travel time (operational congestion tail) |
| `section_min_time` | `float64` | Minutes | Minimum recorded traversal time across training split |
| `section_std_time` | `float64` | Minutes | Standard deviation of travel times (volatility metric) |

---

### Network Topology & Congestion
| Column | Type | Description | Source File / Derivation |
|:---|:---|:---|:---|
| `edge_ntrains` | `int32` | Number of scheduled daily trains sharing this track segment | `data/cleaned/edges_cleaned.csv` (fallback: 1) |

---

### Weather & Environmental Features (ERA5 Reanalysis)
*Sourced from Open-Meteo hourly ERA5 reanalysis mapped to the closest junction station coordinates.*
| Column | Type | Units | Description |
|:---|:---|:---|:---|
| `weather_station_code` | `string` | Station code of the nearest major meteorological hub |
| `temperature_2m` | `float64` | °C | 2-meter air temperature |
| `precipitation` | `float64` | mm/hr | Hourly precipitation rate |
| `weather_code` | `int32` | WMO code | World Meteorological Organization weather code (0=clear, 45=fog, 65=heavy rain) |
| `wind_speed_10m` | `float64` | km/h | 10-meter wind speed |
| `visibility` | `float64` | Meters | Horizontal surface visibility |
| `is_foggy` | `int32` | 0 or 1 | 1 if `weather_code in [45, 48]` or `visibility < 1000m` |
| `is_heavy_rain` | `int32` | 0 or 1 | 1 if `precipitation >= 5.0 mm/hr` or `weather_code in [65, 82]` |
