# Post-drought forest dieback in Sardinia: where should foresters inspect first?

Earth Observation AI Engineer: home assignment. Everything is in [`notebook.ipynb`](notebook.ipynb), executed top to bottom.

![Method](figures/method_flow.jpg)

## Choices

| | |
|---|---|
| Use case | Forest dieback after the 2024 drought in Sardinia (Sette Fratelli massif, Sarrabus) |
| Decision question | Which forest areas should the regional forest service inspect first: canopy that dried out in summer 2024 **and did not recover** in 2025? |
| AOI and time range | `(9.33, 39.19, 9.52, 39.33)`, July–September 2019–2025 |
| Data | Sentinel-2 L2A (98 acquisitions), ESA WorldCover 2021 |
| Backend | Microsoft Planetary Computer, STAC API (no account needed) |
| Indices | NDMI (main), NDVI, CRswir, NBR (fire / logging screen) |
| Foundation models (bonus B) | TerraMind 1.0 base, run locally (generated land cover, embeddings); TESSERA v1 pre-computed embeddings |

## Outputs (`outputs/`)

| File | What |
|---|---|
| `composites_true_false_colour_2023_2025.png` | True- and false-colour summer composites |
| `timeseries_ndmi_ndvi.png` | NDMI and NDVI per acquisition, 2019–2025 |
| `ndmi_z2024.tif`, `ndmi_recovery2025.tif`, `dieback_alert.tif` | Cloud-Optimised GeoTIFFs, EPSG:32632, 20 m |
| `alert_cells.geojson`, `alert_patches.geojson` | Priority 25 ha cells and flagged patches (EPSG:4326) |
| `alert_cells.csv` | Ranked inspection list |
| `anomaly_recovery_flag.png` | Anomaly, recovery and alert maps |
| `priority_cells_true_colour.png` | Top-3 priority cells in true colour, 2023–2025 |
| `terramind_lulc_2023_2025.png`, `terramind_embedding_change.png` | Bonus B: TerraMind generated land cover and embedding change |
| `tessera_clusters_alert_rate.png` | Bonus B: TESSERA forest types and alert rate |

## Repository layout

- `notebook.ipynb`: narrated analysis, all cells executed.
- `src/pipeline.py`: all processing functions (search, masking, indices, composites, anomaly / recovery, aggregation, export). The notebook calls them, and the same functions are meant to be exposed as MCP tools.
- `figures/`: method diagrams and illustrations used in the notebook.
- `outputs/`: generated files.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebook.ipynb   # or: jupyter nbconvert --to notebook --execute notebook.ipynb
```

A full run takes about 20 minutes on a laptop CPU: Sentinel-2 windows for seven summers, 12-band 10 m composites for TerraMind, and about 1.7 GB of TESSERA embeddings on first run (cached in `data/`).
