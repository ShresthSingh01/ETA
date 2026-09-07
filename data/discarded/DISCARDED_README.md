# Discarded Datasets Log & Provenance Audit

This document records the datasets inspected during project initialization and discarded from the active modeling pipeline, along with the precise methodological and empirical justification.

## 1. `part-00000-9194fa62-016b-4ae5-8f5e-92b5d2075f1a-c000.csv` (583.8 MB)

- **Origin**: Found in initial root workspace.
- **Header Analysis**:
  `YEAR,MONTH,DAY,DAY_OF_WEEK,TRAIN_OPERATOR,TRAIN_NUMBER,COACH_ID,SOURCE_STATION,DESTINATION_STATION,...WEATHER_DELAY`
- **Sample Inspection**:
  Row 1: `2015,1,1,4,AS,98,N407AS,ANC,SEA,...`
  - `AS` = Alaska Airlines
  - `ANC` = Ted Stevens Anchorage International Airport (Alaska, USA)
  - `SEA` = Seattle-Tacoma International Airport (Washington, USA)
  - `N407AS` = FAA Tail Number registration for Boeing 737 aircraft
- **Conclusion**: This is the 2015 US Bureau of Transportation Statistics (BTS) Airline On-Time Performance dataset, which had columns renamed (`AIRLINE` -> `TRAIN_OPERATOR`, `FLIGHT_NUMBER` -> `TRAIN_NUMBER`, `TAIL_NUMBER` -> `COACH_ID`).
- **Action**: **DROPPED**. Irrelevant foreign aviation data. Excluded to ensure 100% data integrity and credibility.

---

## 2. Anurag Raturi Kaggle Indian Railways Delay Dataset

- **Origin**: Kaggle synthetic delay dataset.
- **Audit Findings**:
  - Author documentation explicitly confirms records were "manually created".
  - Incidents (`SIGNAL_FAILURE`, `CATTLE_ON_TRACK`) were synthetically simulated and injected.
  - Environmental factors (`Weather_Conditions: Clear/Rainy/Foggy`, `Route_Congestion: Low/Medium/High`) are coarse categorical placeholders without physical station/time grounding.
- **Conclusion**: Unacceptable for production-grade, empirically validated ETA prediction.
- **Action**: **DROPPED**. The project relies exclusively on 1.28M real September 2024 operational records, genuine IRN network graphs, genuine station coordinates, and verified ERA5 reanalysis weather via Open-Meteo.
