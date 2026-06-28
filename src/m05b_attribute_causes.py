"""
Milestone 5b — attribute the cause of each stress flag.

For every (tile, window) with label_z15 == 1, attach four binary
attribution flags + a single best-guess category:

    is_drought   USDM drought class >= 2 for this (tile, window)
    is_flooded   tile-median S1 VV dB <= -20 (water-like backscatter)
    is_sudden    NDVI delta over the prior ~4 weeks > 0.30
    is_isolated  fewer than 2 of the 8 neighbouring tiles were also flagged

then category by priority:
    flood      > drought (slow) > climatic_event (sudden, clustered) >
    local_field (sudden, isolated) > drought (catch-all) > unknown

Outputs:
    m05b_attribution/
        stress_attribution.parquet      one row per stressed (tile, window)
        category_summary.json           counts by category and by window
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds as window_from_bounds


PERSIST_ROOT = Path(
    "<PERSIST_ROOT>"
)
SCRATCH_ROOT = Path("<SCRATCH_ROOT>")

AOI_TILES_GPKG_PATH = PERSIST_ROOT / "m01_aoi_fields" / "central_iowa_tiles.gpkg"
LABELS_PATH         = SCRATCH_ROOT / "m04_proxy_labels" / "labels.parquet"
USDM_PATH           = SCRATCH_ROOT / "m05_validation_refs" / "usdm_per_tile_window.parquet"
SENTINEL1_DIR       = SCRATCH_ROOT / "m02_satellite_chips" / "sentinel1"

OUTPUT_DIR    = SCRATCH_ROOT / "m05b_attribution"
ATTRIBUTION_PATH = OUTPUT_DIR / "stress_attribution.parquet"
SUMMARY_PATH     = OUTPUT_DIR / "category_summary.json"

COMPOSITE_CRS_EPSG = "EPSG:32615"
TILE_PIXELS = 512
SCORE_YEAR = 2024

DROUGHT_USDM_CUTOFF = 2          # D2+ counts as drought-confirmed
FLOOD_VV_DB_CUTOFF = -20.0       # water-like backscatter
SUDDEN_NDVI_DROP_CUTOFF = 0.30   # NDVI drop over the prior ~28 days
SUDDEN_LOOKBACK_WINDOWS = 2      # compare to (window - 2)
ISOLATED_NEIGHBOUR_CUTOFF = 2    # < 2 stressed neighbours -> isolated
NEIGHBOUR_DISTANCE_FACTOR = 1.6  # ~ sqrt(2) * 1.13 for 8-connectivity slack

S1_CHIP_FILENAME_RE = re.compile(r"^s1_(\d{4})_w(\d{2})_([a-z]{3})(\d{2})\.tif$")
MONTH_NAMES = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]


def load_tile_table_in_utm() -> gpd.GeoDataFrame:
    tiles_in_wgs84 = gpd.read_file(AOI_TILES_GPKG_PATH).sort_values("tile_id").reset_index(drop=True)
    return tiles_in_wgs84.to_crs(COMPOSITE_CRS_EPSG)


def list_s1_chip_for_year_window(year: int, window_index: int) -> Optional[Path]:
    for path in SENTINEL1_DIR.glob(f"s1_{year}_w{window_index:02d}_*.tif"):
        if path.stat().st_size > 1024:
            return path
    return None


def extract_per_tile_vv_dB_median(
    s1_chip_path: Path,
    tiles_in_utm: gpd.GeoDataFrame,
) -> np.ndarray:
    """Return per-tile median VV (dB) — uses band 0 of the S1 composite."""
    n_tiles = len(tiles_in_utm)
    medians = np.full(n_tiles, np.nan, dtype=np.float32)
    bounds_xy = tiles_in_utm.geometry.bounds.to_numpy()
    with rasterio.open(s1_chip_path) as src:
        for tile_index in range(n_tiles):
            xmin, ymin, xmax, ymax = bounds_xy[tile_index]
            window = window_from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)
            chip = src.read(
                indexes=1,                  # VV is the first band
                window=window,
                out_shape=(TILE_PIXELS, TILE_PIXELS),
                boundless=True,
                fill_value=np.nan,
            ).astype(np.float32)
            chip = np.where(chip < -32000.0, np.nan, chip)
            if np.isfinite(chip).any():
                medians[tile_index] = float(np.nanmedian(chip))
    return medians


def build_neighbour_lookup(tiles_in_utm: gpd.GeoDataFrame, tile_size_metres: float = 5120.0) -> Dict[int, List[int]]:
    """For each tile_id, list of tile_ids within 8-connectivity range."""
    centroids = np.array([(g.x, g.y) for g in tiles_in_utm.geometry.centroid])
    tile_ids = tiles_in_utm["tile_id"].to_numpy()
    distance_threshold = tile_size_metres * NEIGHBOUR_DISTANCE_FACTOR
    neighbour_map: Dict[int, List[int]] = {}
    for i, center_i in enumerate(centroids):
        deltas = centroids - center_i
        distances = np.sqrt((deltas ** 2).sum(axis=1))
        neighbour_indices = np.where((distances > 1.0) & (distances <= distance_threshold))[0]
        neighbour_map[int(tile_ids[i])] = tile_ids[neighbour_indices].tolist()
    return neighbour_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=SCORE_YEAR)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tiles_in_utm = load_tile_table_in_utm()
    print(f"[m05b] {len(tiles_in_utm)} tiles loaded")

    labels_df = pd.read_parquet(LABELS_PATH)
    usdm_df = pd.read_parquet(USDM_PATH)

    merged = labels_df.merge(usdm_df, on=["tile_id", "year", "window_index"], how="left")
    score_year_mask = merged["year"] == args.year
    score_rows = merged.loc[score_year_mask].copy()

    stressed_mask = (score_rows["label_z15"] == 1)
    stressed_rows = score_rows.loc[stressed_mask].copy()
    print(f"[m05b] {len(stressed_rows)} stressed (tile, window) rows in {args.year}")

    if stressed_rows.empty:
        raise SystemExit("[m05b] no stressed rows — nothing to attribute")

    stressed_rows["is_drought"] = (
        stressed_rows["usdm_class"].fillna(-1).astype(float) >= DROUGHT_USDM_CUTOFF
    )

    affected_windows = sorted(stressed_rows["window_index"].unique().tolist())
    print(f"[m05b] flood check: reading S1 chips for windows {affected_windows}")
    vv_db_lookup: Dict[Tuple[int, int], np.ndarray] = {}
    for window_index in affected_windows:
        s1_chip_path = list_s1_chip_for_year_window(args.year, window_index)
        if s1_chip_path is None:
            print(f"[m05b]   no usable S1 chip for window {window_index}")
            continue
        per_tile_vv = extract_per_tile_vv_dB_median(s1_chip_path, tiles_in_utm)
        vv_db_lookup[(args.year, window_index)] = per_tile_vv
        print(f"[m05b]   window {window_index:2d}: VV median  min={np.nanmin(per_tile_vv):.2f} "
              f" max={np.nanmax(per_tile_vv):.2f}", flush=True)

    tile_id_to_index = {int(tid): i for i, tid in enumerate(tiles_in_utm["tile_id"].to_numpy())}

    def vv_for_row(year: int, window_index: int, tile_id: int) -> float:
        per_tile_vv = vv_db_lookup.get((year, window_index))
        if per_tile_vv is None:
            return float("nan")
        return float(per_tile_vv[tile_id_to_index[int(tile_id)]])

    stressed_rows["vv_db"] = [
        vv_for_row(year=row.year, window_index=row.window_index, tile_id=row.tile_id)
        for row in stressed_rows.itertuples(index=False)
    ]
    stressed_rows["is_flooded"] = stressed_rows["vv_db"] <= FLOOD_VV_DB_CUTOFF

    score_rows_pivot = score_rows.pivot_table(
        index="tile_id", columns="window_index", values="ndvi", aggfunc="first",
    )

    def ndvi_drop_over_lookback(tile_id: int, window_index: int) -> float:
        lookback_window = window_index - SUDDEN_LOOKBACK_WINDOWS
        if lookback_window < 1 or lookback_window not in score_rows_pivot.columns:
            return float("nan")
        if window_index not in score_rows_pivot.columns:
            return float("nan")
        return float(score_rows_pivot.at[tile_id, lookback_window] - score_rows_pivot.at[tile_id, window_index])

    stressed_rows["ndvi_drop_recent"] = [
        ndvi_drop_over_lookback(row.tile_id, row.window_index)
        for row in stressed_rows.itertuples(index=False)
    ]
    stressed_rows["is_sudden"] = stressed_rows["ndvi_drop_recent"] > SUDDEN_NDVI_DROP_CUTOFF

    stressed_per_window: Dict[int, set] = (
        stressed_rows.groupby("window_index")["tile_id"]
        .apply(lambda series: set(int(x) for x in series))
        .to_dict()
    )
    neighbour_map = build_neighbour_lookup(tiles_in_utm)

    def count_stressed_neighbours(tile_id: int, window_index: int) -> int:
        peers = stressed_per_window.get(window_index, set())
        neighbour_tile_ids = neighbour_map.get(int(tile_id), [])
        return int(sum(1 for n in neighbour_tile_ids if n in peers))

    stressed_rows["n_stressed_neighbours"] = [
        count_stressed_neighbours(row.tile_id, row.window_index)
        for row in stressed_rows.itertuples(index=False)
    ]
    stressed_rows["is_isolated"] = stressed_rows["n_stressed_neighbours"] < ISOLATED_NEIGHBOUR_CUTOFF

    def categorize(row) -> str:
        if row.is_flooded:
            return "flood"
        if row.is_sudden and not row.is_isolated:
            return "climatic_event_or_frost"
        if row.is_sudden and row.is_isolated:
            return "local_field"
        if row.is_drought:
            return "drought"
        return "other_unknown"

    stressed_rows["category"] = [categorize(row) for row in stressed_rows.itertuples(index=False)]

    stressed_rows.to_parquet(ATTRIBUTION_PATH, index=False)
    print(f"[m05b] wrote {ATTRIBUTION_PATH}  ({len(stressed_rows)} rows)")

    overall_counts = stressed_rows["category"].value_counts().to_dict()
    by_window = (
        stressed_rows.groupby("window_index")["category"]
        .value_counts()
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )
    summary = {
        "year":               args.year,
        "n_stressed_rows":    int(len(stressed_rows)),
        "overall_counts":     {k: int(v) for k, v in overall_counts.items()},
        "by_window":          {int(k): {kk: int(vv) for kk, vv in v.items()} for k, v in by_window.items()},
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"[m05b] wrote {SUMMARY_PATH}")
    print("[m05b] overall category counts:")
    for k, v in overall_counts.items():
        print(f"        {k}: {v}")


if __name__ == "__main__":
    main()
