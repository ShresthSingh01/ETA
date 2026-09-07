# IRN Data Validation & Cleaning Report

Generated automatically during Phase 1: Data Foundation.

## 1. Network Graph Edges (`IRN_edges.csv`)

- **Initial Edge Count**: 9,336
- **Final Cleaned Edge Count**: 9,335
- **Negative Distance Edges Removed**: 1 (e.g., SNNR->YPR -624km route artifact)
- **Zero Distance Edges Repaired**: 4 (geodetic distances imputed from station coordinates)
- **Self Loops Removed**: 0
- **Min Distance**: 0.63 km
- **Max Distance**: 3433.00 km
- **Avg Distance**: 38.12 km

## 2. Station Coordinates (`india_railway_stations.csv`)

- **Total Stations**: 8,990
- **Valid Geospatial Coordinates**: 8,801
- **Missing / Out-of-bounds**: 189
- **Unique Station Codes**: 8,990
