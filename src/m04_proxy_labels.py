"""
Milestone 4 — NDVI / EVI proxy labels.

For each (tile, window) S2 composite produced in M2, compute per-tile mean
NDVI and EVI. Then for each calendar-window slot (e.g. "week-of-year #14"),
build a historical mean+std across 2017-2023 *per tile*. Finally, score
2024 windows as z-scores against that history and flag those with
ndvi_z < -1.5 as "stressed" (the proxy label).

Output: parquet at
    <SCRATCH_ROOT>/m04_proxy_labels/labels.parquet
with columns:
    tile_id, year, window_index, date_start,
    ndvi, evi, valid_fraction,
    ndvi_z, evi_z,
    label_z15  (1 if ndvi_z < -1.5 else 0)

We mean-pool over the tile rather than running per-pixel because Clay sees
the tile as one chip — same granularity makes label/embedding aligned.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path
from typing import List, Optional, Tuple

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
SENTINEL2_DIR = SCRATCH_ROOT / "m02_satellite_chips" / "sentinel2"
OUTPUT_DIR = SCRATCH_ROOT / "m04_proxy_labels"
OUTPUT_PARQUET_PATH = OUTPUT_DIR / "labels.parquet"

COMPOSITE_CRS_EPSG = "EPSG:32615"
TILE_PIXELS = 512

S2_BAND_ORDER = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
B02_INDEX = S2_BAND_ORDER.index("B02")
B04_INDEX = S2_BAND_ORDER.index("B04")
B08_INDEX = S2_BAND_ORDER.index("B08")

DN_SCALE = 10000.0
S2_NODATA_DN = 0

HISTORY_YEARS = list(range(2017, 2024))
SCORE_YEAR = 2024
LABEL_Z_THRESHOLD = -1.5
EPSILON = 1e-6

CHIP_FILENAME_RE = re.compile(r"^s2_(\d{4})_w(\d{2})_([a-z]{3})(\d{2})\.tif$")
MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def parse_window_date_from_filename(path: Path) -> Tuple[int, int, dt.date]:
    m = CHIP_FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Unrecognised chip filename: {path.name}")
    year = int(m.group(1))
    window_index = int(m.group(2))
    month = MONTH_NAMES.index(m.group(3)) + 1
    day = int(m.group(4))
    return year, window_index, dt.date(year, month, day)


def load_tile_table_in_utm() -> gpd.GeoDataFrame:
    tiles_in_wgs84 = gpd.read_file(AOI_TILES_GPKG_PATH).sort_values("tile_id").reset_index(drop=True)
    return tiles_in_wgs84.to_crs(COMPOSITE_CRS_EPSG)


def extract_per_tile_subchips(composite_path: Path, tiles_in_utm: gpd.GeoDataFrame) -> np.ndarray:
    """[n_tiles, 10 bands, TILE_PIXELS, TILE_PIXELS] int16 DN."""
    n_tiles = len(tiles_in_utm)
    bounds_xy = tiles_in_utm.geometry.bounds.to_numpy()
    out: Optional[np.ndarray] = None
    with rasterio.open(composite_path) as src:
        for tile_index in range(n_tiles):
            xmin, ymin, xmax, ymax = bounds_xy[tile_index]
            window = window_from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)
            chip = src.read(
                window=window,
                out_shape=(src.count, TILE_PIXELS, TILE_PIXELS),
                boundless=True,
                fill_value=S2_NODATA_DN,
            )
            if out is None:
                out = np.empty((n_tiles, src.count, TILE_PIXELS, TILE_PIXELS), dtype=chip.dtype)
            out[tile_index] = chip
    return out


def compute_indices_per_tile(chips_int16: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    chips_int16: [n_tiles, 10, H, W] reflectance DN (multiplied by 10000).
    Returns (ndvi_per_tile, evi_per_tile, valid_fraction_per_tile) — each
    one float32 of shape [n_tiles]. The tile-level value is the *median*
    over valid pixels (robust to cloud/saturation outliers that would
    swing the mean — see EVI denominator blow-up for high-reflectance
    cloud pixels). A pixel is valid only when all three of B02/B04/B08
    are positive AND reflectance is in [0, 1] (DN <= DN_SCALE).
    """
    blue = chips_int16[:, B02_INDEX].astype(np.float32) / DN_SCALE
    red  = chips_int16[:, B04_INDEX].astype(np.float32) / DN_SCALE
    nir  = chips_int16[:, B08_INDEX].astype(np.float32) / DN_SCALE

    in_range = (
        (chips_int16[:, B02_INDEX] > 0) & (chips_int16[:, B02_INDEX] <= DN_SCALE)
        & (chips_int16[:, B04_INDEX] > 0) & (chips_int16[:, B04_INDEX] <= DN_SCALE)
        & (chips_int16[:, B08_INDEX] > 0) & (chips_int16[:, B08_INDEX] <= DN_SCALE)
    )

    ndvi_per_pixel = (nir - red) / np.maximum(nir + red, EPSILON)
    evi_per_pixel  = 2.5 * (nir - red) / np.maximum(nir + 6.0 * red - 7.5 * blue + 1.0, EPSILON)
    # EVI is physically bounded to [-1, 1]; clip to drop residual outliers
    # from edge-case denominators.
    evi_per_pixel = np.clip(evi_per_pixel, -1.0, 1.0)

    ndvi_for_median = np.where(in_range, ndvi_per_pixel, np.nan)
    evi_for_median  = np.where(in_range, evi_per_pixel,  np.nan)

    n_tiles, pixels_per_tile = chips_int16.shape[0], TILE_PIXELS * TILE_PIXELS
    ndvi_per_tile = np.nanmedian(ndvi_for_median.reshape(n_tiles, -1), axis=1).astype(np.float32)
    evi_per_tile  = np.nanmedian(evi_for_median.reshape(n_tiles, -1),  axis=1).astype(np.float32)
    valid_fraction = (in_range.reshape(n_tiles, -1).sum(axis=1) / pixels_per_tile).astype(np.float32)
    return ndvi_per_tile, evi_per_tile, valid_fraction


def list_s2_composites() -> List[Path]:
    return sorted(SENTINEL2_DIR.glob("s2_*.tif"))


def collect_per_window_indices(tiles_in_utm: gpd.GeoDataFrame) -> pd.DataFrame:
    """Returns long-format DataFrame: tile_id × (year, window_index, date_start, ndvi, evi, valid_fraction)."""
    composites = list_s2_composites()
    tile_id_array = tiles_in_utm["tile_id"].to_numpy()
    n_tiles = len(tile_id_array)
    print(f"[m04] {len(composites)} S2 composites, {n_tiles} tiles per composite")

    long_rows = []
    for composite_path in composites:
        if composite_path.stat().st_size < 1024:
            print(f"[m04] SKIP  {composite_path.name}  (size={composite_path.stat().st_size} B — corrupt)", flush=True)
            continue
        try:
            chips = extract_per_tile_subchips(composite_path, tiles_in_utm)
        except Exception as exc:
            print(f"[m04] SKIP  {composite_path.name}  ({type(exc).__name__}: {exc})", flush=True)
            continue
        ndvi_per_tile, evi_per_tile, valid_fraction = compute_indices_per_tile(chips)
        year, window_index, date_start = parse_window_date_from_filename(composite_path)
        long_rows.append(
            pd.DataFrame({
                "tile_id":      tile_id_array,
                "year":         np.full(n_tiles, year, dtype=np.int16),
                "window_index": np.full(n_tiles, window_index, dtype=np.int16),
                "date_start":   np.full(n_tiles, np.datetime64(date_start)),
                "ndvi":         ndvi_per_tile,
                "evi":          evi_per_tile,
                "valid_fraction": valid_fraction,
            })
        )
        print(f"[m04] {composite_path.name}  ndvi mean={np.nanmean(ndvi_per_tile):.3f}  evi mean={np.nanmean(evi_per_tile):.3f}", flush=True)
    return pd.concat(long_rows, ignore_index=True)


def attach_zscores_and_labels(long_indices_df: pd.DataFrame) -> pd.DataFrame:
    """For each (tile_id, window_index), compute mean+std over HISTORY_YEARS, then z-score every row."""
    historical_mask = long_indices_df["year"].isin(HISTORY_YEARS)
    historical_subset = long_indices_df[historical_mask]

    grouped = historical_subset.groupby(["tile_id", "window_index"])
    history_stats = grouped.agg(
        ndvi_history_mean=("ndvi", "mean"),
        ndvi_history_std =("ndvi", "std"),
        evi_history_mean =("evi", "mean"),
        evi_history_std  =("evi", "std"),
        history_count    =("ndvi", "count"),
    ).reset_index()

    merged = long_indices_df.merge(history_stats, on=["tile_id", "window_index"], how="left")

    merged["ndvi_z"] = (merged["ndvi"] - merged["ndvi_history_mean"]) / merged["ndvi_history_std"].replace(0, np.nan)
    merged["evi_z"]  = (merged["evi"]  - merged["evi_history_mean"])  / merged["evi_history_std"].replace(0, np.nan)

    merged["label_z15"] = (merged["ndvi_z"] < LABEL_Z_THRESHOLD).astype(np.int8)
    merged.loc[merged["ndvi"].isna() | merged["ndvi_z"].isna(), "label_z15"] = -1   # missing
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="(debug) only process the first N composites")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[m04] loading tile table: {AOI_TILES_GPKG_PATH}")
    tiles_in_utm = load_tile_table_in_utm()

    long_indices_df = collect_per_window_indices(tiles_in_utm)
    if args.limit is not None:
        long_indices_df = long_indices_df.head(args.limit * len(tiles_in_utm))

    print(f"[m04] collected {len(long_indices_df)} (tile, window) rows")

    labelled = attach_zscores_and_labels(long_indices_df)
    print(f"[m04] z-scores attached; "
          f"{int((labelled['label_z15'] == 1).sum())} rows flagged stressed, "
          f"{int((labelled['label_z15'] == -1).sum())} missing")

    OUTPUT_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    labelled.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    print(f"[m04] wrote {OUTPUT_PARQUET_PATH}  ({len(labelled)} rows)")

    score_year_mask = labelled["year"] == SCORE_YEAR
    summary = labelled.loc[score_year_mask].groupby("window_index").agg(
        n=("tile_id", "count"),
        stressed=("label_z15", lambda s: int((s == 1).sum())),
        ndvi_mean=("ndvi", "mean"),
        ndvi_z_mean=("ndvi_z", "mean"),
    )
    print(f"[m04] {SCORE_YEAR} summary by window:")
    print(summary)


if __name__ == "__main__":
    main()
