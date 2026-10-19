**Question.** After the 2024 drought, which forest areas in the Sette Fratelli massif, bbox [9.33, 39.19, 9.52, 39.33], should the forest service inspect first? Use summer 2025 as the recovery year.

*Model:* `hf.co/eve-esa/EVE-Instruct-GGUF-Q4_K_M` (digest `n/a`), 2026-10-19T20:16:27+00:00, 24.0 s

→ **calls `dieback_alerts`** with
```json
{
 "bbox": [
  9.33,
  39.19,
  9.52,
  39.33
 ],
 "event_year": 2024,
 "recovery_year": 2025
}
```
← `dieback_alerts` returned
```json
{
 "area_km2": 254.7,
 "event_year": 2024,
 "recovery_year": 2025,
 "baseline_years": [
  2019,
  2020,
  2021,
  2022,
  2023
 ],
 "forest_ha": 17067,
 "excluded_fire_or_logging_ha": 402,
 "flagged_ha": 2211,
 "flagged_pct_of_forest": 13.3,
 "priority_cells_25ha": 40,
 "top_cells": [
  {
   "rank": 1,
   "lat": 39.23812,
   "lon": 9.43093,
   "flagged_ha": 19.1,
   "forest_ha": 22.0
  },
  {
   "rank": 2,
   "lat": 39.23366,
   "lon": 9.41932,
   "flagged_ha": 19.0,
   "forest_ha": 24.9
  },
  {
   "rank": 3,
   "lat": 39.31029,
   "lon": 9.40818,
   "flagged_ha": 16.6,
   "forest_ha": 25.0
  },
  {
   "rank": 4,
   "lat": 39.24263,
   "lon": 9.43096,
   "flagged_ha": 16.0,
   "forest_ha": 18.6
  },
  {
   "rank": 5,
   "lat": 39.26945,
   "lon": 9.48329,
   "flagged_ha": 15.8,
   "forest_ha": 24.8
  }
 ],
 "cells_csv": "outputs/mcp/dieback_cells_8a0b9f.csv",
 "cells_geojson": "outputs/mcp/dieback_cells_8a0b9f.geojson",
 "runtime_s": 653.5,
 "cached": true
}
```

**Answer.** The forest service should prioritize inspecting 2,211 hectares of forest in the Sette Fratelli massif that did not recover from the 2024 drought, representing 13.3% of the total forest area (17,067 ha). The top five priority cells are located at coordinates such as (39.23812° N, 9.43093° E) with 19.1 ha flagged and (39.23366° N, 9.41932° E) with 19.0 ha flagged. The full list of 40 priority cells (each ~25 ha) is available in the CSV file at `outputs/mcp/dieback_cells_8a0b9f.csv` and as a GeoJSON file at `outputs/mcp/dieback_cells_8a0b9f.geojson`.

*Grounding check:* 12 numbers in the answer, all found in tool results.