# Clay Vegetation-Stress Project — Milestones

> **Resume rule.** Any future Claude session reading this file should also
> open `PROGRESS.md` (same directory) to see which milestone is in flight
> and what was completed last. Memory key `clay_veg_stress_project` carries
> the cluster paths, env, and SSH access info.

## Project goal

Detect **vegetation stress in US agricultural fields** from Sentinel-2
optical + Sentinel-1 SAR imagery, using the **Clay v1.5 foundation model**
(frozen, ~632 M-param ViT) to generate per-tile embeddings, then a small
temporal head (or an embedding-distance anomaly detector) to flag stress.

## Scope (deliberately tight)

- **AOI:** Story County, Iowa (FIPS 19169) — corn / soybean belt
- **Crops:** corn (CDL=1) + soybean (CDL=5) only
- **Time range:** 2017-01 → 2024-12 (8 years × 12 months = 96 monthly composites)
- **Source:** Earth Engine for Sentinel imagery + USDA CDL direct downloads
- **Storage:** a fast scratch volume for chips and intermediate artefacts,
  a persistent volume for code + the small final tables

## Compute environment

- A SLURM-managed HPC cluster (or any environment where you can submit
  CPU and GPU batch jobs).
- Two volumes on the cluster — one fast scratch, one persistent — for
  intermediate data and for code respectively.
- SSH ControlMaster on the development machine to minimise re-login
  overhead.
- A virtual environment with rasterio, geopandas, pystac-client,
  planetary-computer, odc-stac, xarray, rioxarray, torch, scikit-learn,
  pandas, pyarrow, matplotlib for the data and analysis steps; a
  separate Python ≥ 3.11 venv for `claymodel`-driven inference and the
  trained GRU.
- SLURM partition / account flags vary by site — adapt the placeholder
  `.slurm` scripts in `src/` to your scheduler.

## Milestone list

### M1. Build the AOI: fields & boundaries
- Pull TIGER 2024 counties, extract Story County polygon
- Pull USDA CDL 2017–2024 clipped to the county via CropScape API
- Define field-equivalent units (decision: **640 m grid tiles** — see PROGRESS.md)
- Save tile/field table as GeoPackage on persistent storage
- **Status:** in progress (Option B pivot — grid tiles)

### M2. Pull Sentinel-2 optical + Sentinel-1 SAR from GEE → cluster
- Sentinel-2 SR L2A, cloud <20 %, monthly median composites, 10 bands
- Sentinel-1 GRD, VV + VH, monthly mean composites
- Export per-tile 64 × 64 chips as multi-band GeoTIFFs to Drive
- `rclone` sync to `<SCRATCH_ROOT>/m02_satellite_chips/`
- Expected: 96 months × ~2000 tiles × 2 sensors = ~380 k chips, ~50–150 GB

### M3. Generate Clay v1.5 embeddings on a GPU node
- Install Clay v1.5 + weights (`made-with-clay/Clay` on HF) into clay_env
- SLURM job on `gpushort` (A100-40 GB, sbg5/sbg19): load each chip, feed
  Clay frozen with metadata (centroid lon/lat, date, sensor, GSD)
- Save 768-dim embeddings as Zarr `[n_tiles × n_months × 768]` (per sensor)

### M4. Compute proxy labels (NDVI / EVI z-scores)
- NDVI and EVI per tile per month from Sentinel-2 chips
- Per-tile, per-calendar-month historical mean + std from 2017–2023
- z-score for each 2024 month; flag `z < -1.5` as stressed
- Save tidy Parquet: `[tile_id, year_month, ndvi, ndvi_z, label]`

### M5. Validate the proxy
- US Drought Monitor weekly county shapefiles for 2024
- USDA NASS Crop Progress & Condition state-level via Quick Stats API
- Spatial overlay + agreement percentage
- This is itself a deliverable result

### M6. Simple baseline (the one to beat)
- NDVI-anomaly threshold alone, no Clay, no learned model
- Compute F1 / Precision / Recall / AUC over Story County 2024
- Save reference metrics that every later method must beat

### M7. Path A — supervised temporal head over Clay embeddings
- Input: `[T × 768]` Clay sequence per tile
- Model: 1-layer GRU (hidden=128) → linear → sigmoid per timestep
- Loss: BCEWithLogits on proxy labels, masked for missing months
- Train on 2017–2023, validate on 2024
- SLURM job on `gpushort`

### M8. Path B — unsupervised embedding-distance anomaly
- Historical mean Clay embedding from 2017–2023 per (tile, calendar-month)
- 2024 embedding → cosine distance to historical mean
- Threshold distance to flag stress

### M9. Three-way comparison
- Baseline vs Path A vs Path B
- F1 + lead time + confusion matrices + ROC curves + per-month breakdown
- Save tables + plots

### M10. Outputs and visualisations
- Stress map of Story County per 2024 month (12 PNGs)
- 5 example tile time-series plots
- Drought Monitor overlay image
- Confusion matrices

### M11. Package into a clean repo and push
- Layout mirroring `seasonal-precip-swin` (config + lazy imports + CLI + smoke tests)
- README: motivation → data → Clay role → two paths → results → honest-labels section
- Push to `zahirnik/vegetation-stress-clay`, add to Remote Sensing list
- Code in `<PERSIST_ROOT>/` (persistent), artefacts in `<SCRATCH_ROOT>/` (scratch)

### M12. Short write-up / deck (6–8 slides)
- Problem framing (vegetation stress from multi-sensor satellite data)
- Clay role + data flow
- Proxy-labels honesty section
- Results (baseline vs Path A vs Path B)
- Stress map + Drought Monitor overlay
- Limitations + next steps

## Recommended execution order

1, 2, 4 can run in parallel; **3 is the heaviest GPU step**, do once data
is ready; then 5; then 6, 7, 8 in parallel; then 9 → 10 → 11 → 12.
