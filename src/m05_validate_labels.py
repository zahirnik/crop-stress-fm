"""
Milestone 5 — validate the M4 proxy labels against the US Drought Monitor.

For each of our 2024 windows we pick the closest Tuesday (USDM's weekly
release day), download the USDM shapefile, clip to Iowa, and assign a
drought class (D0/D1/D2/D3/D4 or none) to each tile centroid via spatial
join. Then we compare the USDM-derived "is droughted" flag against our
NDVI z-score label.

Outputs:
    <SCRATCH_ROOT>/m05_validation_refs/
        usdm_per_tile_window.parquet   per (tile, window) USDM class
        validation_metrics.json        agreement metrics
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError
from urllib.request import urlopen

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PERSIST_ROOT = Path(
    "<PERSIST_ROOT>"
)
SCRATCH_ROOT = Path("<SCRATCH_ROOT>")

AOI_TILES_GPKG_PATH = PERSIST_ROOT / "m01_aoi_fields" / "central_iowa_tiles.gpkg"
LABELS_PATH = SCRATCH_ROOT / "m04_proxy_labels" / "labels.parquet"
OUTPUT_DIR  = SCRATCH_ROOT / "m05_validation_refs"
USDM_CACHE_DIR = OUTPUT_DIR / "usdm_cache"
USDM_PER_TILE_WINDOW_PATH = OUTPUT_DIR / "usdm_per_tile_window.parquet"
METRICS_PATH = OUTPUT_DIR / "validation_metrics.json"

USDM_URL_FORMAT = "https://droughtmonitor.unl.edu/data/shapefiles_m/USDM_{date}_M.zip"
SCORE_YEAR = 2024

DROUGHT_BINARY_CUTOFF_CLASS = 2   # D2 and above (severe+) -> "drought"
IOWA_STATE_FIPS = "19"


def closest_tuesday(target_date: dt.date) -> dt.date:
    """USDM is released every Thursday for the Tuesday snapshot before it."""
    offset_to_tuesday = (target_date.weekday() - 1) % 7
    return target_date - dt.timedelta(days=offset_to_tuesday)


def download_usdm_shapefile_for_tuesday(snapshot_date: dt.date) -> Optional[gpd.GeoDataFrame]:
    """Download + extract the USDM shapefile for the given Tuesday."""
    cache_path = USDM_CACHE_DIR / f"USDM_{snapshot_date.strftime('%Y%m%d')}_M.zip"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        url = USDM_URL_FORMAT.format(date=snapshot_date.strftime("%Y%m%d"))
        try:
            with urlopen(url, timeout=60) as response:
                cache_path.write_bytes(response.read())
        except HTTPError as exc:
            print(f"[m05]   download failed for {snapshot_date}: HTTP {exc.code}", flush=True)
            return None

    with zipfile.ZipFile(cache_path) as zip_file:
        extract_dir = USDM_CACHE_DIR / snapshot_date.strftime("%Y%m%d")
        extract_dir.mkdir(parents=True, exist_ok=True)
        zip_file.extractall(extract_dir)

    shapefile_paths = list(extract_dir.glob("*.shp"))
    if not shapefile_paths:
        print(f"[m05]   no .shp inside zip for {snapshot_date}", flush=True)
        return None
    return gpd.read_file(shapefile_paths[0])


def load_window_table() -> pd.DataFrame:
    """Distinct (year, window_index, date_start) rows from the labels parquet."""
    labels_df = pd.read_parquet(LABELS_PATH, columns=["year", "window_index", "date_start"])
    return (
        labels_df.drop_duplicates(["year", "window_index"])
        .sort_values(["year", "window_index"])
        .reset_index(drop=True)
    )


def load_tile_centroids_in_wgs84() -> gpd.GeoDataFrame:
    tiles_in_wgs84 = gpd.read_file(AOI_TILES_GPKG_PATH).sort_values("tile_id").reset_index(drop=True)
    centroids = tiles_in_wgs84.geometry.centroid
    return gpd.GeoDataFrame(
        {"tile_id": tiles_in_wgs84["tile_id"].values},
        geometry=centroids,
        crs=tiles_in_wgs84.crs,
    )


def assign_drought_class_for_window(
    snapshot_date: dt.date,
    tile_centroids: gpd.GeoDataFrame,
) -> Optional[pd.Series]:
    """For one Tuesday, return Series[tile_id -> drought class int 0..4, NaN if no drought]."""
    usdm_gdf = download_usdm_shapefile_for_tuesday(snapshot_date)
    if usdm_gdf is None:
        return None
    if usdm_gdf.crs is None:
        usdm_gdf = usdm_gdf.set_crs(4326)
    usdm_gdf = usdm_gdf.to_crs(tile_centroids.crs)

    drought_class_column = next(
        (c for c in ("DM", "dm", "Class", "CLASS") if c in usdm_gdf.columns),
        None,
    )
    if drought_class_column is None:
        print(f"[m05]   no drought-class column in USDM for {snapshot_date}", flush=True)
        return None

    joined = gpd.sjoin(tile_centroids, usdm_gdf[[drought_class_column, "geometry"]],
                       how="left", predicate="intersects")
    joined = joined.drop_duplicates("tile_id", keep="first")
    return pd.Series(
        data=joined[drought_class_column].astype("Float64").to_numpy(),
        index=joined["tile_id"].to_numpy(),
        name="usdm_class",
    )


def build_usdm_per_tile_window_table(
    window_table: pd.DataFrame,
    tile_centroids: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """For each row of `window_table` (one per year+window_index), assign the USDM class per tile."""
    rows = []
    score_year_mask = window_table["year"] == SCORE_YEAR
    score_windows_df = window_table.loc[score_year_mask].reset_index(drop=True)
    print(f"[m05] fetching USDM for {len(score_windows_df)} 2024 windows")

    for _, row in score_windows_df.iterrows():
        date_start = pd.Timestamp(row["date_start"]).date()
        snapshot_date = closest_tuesday(date_start)
        drought_class_series = assign_drought_class_for_window(snapshot_date, tile_centroids)
        if drought_class_series is None:
            continue
        per_window_df = pd.DataFrame({
            "tile_id":      drought_class_series.index,
            "year":         np.full(len(drought_class_series), row["year"], dtype=np.int16),
            "window_index": np.full(len(drought_class_series), row["window_index"], dtype=np.int16),
            "usdm_date":    np.full(len(drought_class_series), np.datetime64(snapshot_date)),
            "usdm_class":   drought_class_series.values,
        })
        rows.append(per_window_df)
        n_with_class = int(drought_class_series.notna().sum())
        max_class = (
            float(drought_class_series.dropna().max())
            if drought_class_series.notna().any()
            else float("nan")
        )
        print(f"[m05]   {row['window_index']:2d}  {snapshot_date}  "
              f"tiles_in_drought={n_with_class}/{len(drought_class_series)}  "
              f"max_class={max_class}",
              flush=True)
    if not rows:
        raise SystemExit("[m05] no USDM data fetched — abort")
    return pd.concat(rows, ignore_index=True)


def compute_validation_metrics(merged: pd.DataFrame) -> Dict[str, float]:
    """Compare labels vs USDM-derived drought flag."""
    valid_mask = merged["label_z15"].isin([0, 1])
    valid_df = merged.loc[valid_mask].copy()
    valid_df["usdm_drought_flag"] = (
        valid_df["usdm_class"].fillna(-1) >= DROUGHT_BINARY_CUTOFF_CLASS
    ).astype(np.int8)

    label_array = valid_df["label_z15"].to_numpy().astype(np.int8)
    usdm_flag_array = valid_df["usdm_drought_flag"].to_numpy().astype(np.int8)

    if usdm_flag_array.sum() == 0 and label_array.sum() == 0:
        return {
            "n": int(len(valid_df)),
            "note": "no positives in either signal — no usable metric",
        }

    tn, fp, fn, tp = confusion_matrix(label_array, usdm_flag_array, labels=[0, 1]).ravel()
    precision = precision_score(label_array, usdm_flag_array, zero_division=0)
    recall    = recall_score(label_array, usdm_flag_array, zero_division=0)
    f1        = f1_score(label_array, usdm_flag_array, zero_division=0)

    metrics = {
        "n_rows":              int(len(valid_df)),
        "n_label_positives":   int(label_array.sum()),
        "n_usdm_positives":    int(usdm_flag_array.sum()),
        "true_positive":  int(tp),
        "false_positive": int(fp),
        "true_negative":  int(tn),
        "false_negative": int(fn),
        "precision_when_usdm_is_truth": float(precision),
        "recall_when_usdm_is_truth":    float(recall),
        "f1_when_usdm_is_truth":        float(f1),
    }
    # If there's a continuous z-score available, also compute ROC against USDM class.
    if "ndvi_z" in merged.columns:
        z_valid = valid_df["ndvi_z"].notna() & valid_df["usdm_class"].notna()
        if z_valid.any():
            metrics["roc_auc_negative_ndvi_z_vs_usdm_d2plus"] = float(
                roc_auc_score(
                    (valid_df.loc[z_valid, "usdm_class"] >= DROUGHT_BINARY_CUTOFF_CLASS).astype(np.int8),
                    -valid_df.loc[z_valid, "ndvi_z"].to_numpy(),
                )
            )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-windows", type=int, default=None,
                        help="(debug) only process first N windows")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    window_table = load_window_table()
    if args.limit_windows is not None:
        window_table = window_table.head(args.limit_windows)
    print(f"[m05] {(window_table['year'] == SCORE_YEAR).sum()} {SCORE_YEAR} windows in label table")

    tile_centroids = load_tile_centroids_in_wgs84()
    print(f"[m05] {len(tile_centroids)} tile centroids loaded")

    usdm_per_tile_window = build_usdm_per_tile_window_table(window_table, tile_centroids)
    usdm_per_tile_window.to_parquet(USDM_PER_TILE_WINDOW_PATH, index=False)
    print(f"[m05] wrote {USDM_PER_TILE_WINDOW_PATH}  ({len(usdm_per_tile_window)} rows)")

    labels_df = pd.read_parquet(LABELS_PATH)
    merged = labels_df.merge(usdm_per_tile_window, on=["tile_id", "year", "window_index"], how="inner")
    print(f"[m05] {len(merged)} rows after merge with labels")

    metrics = compute_validation_metrics(merged)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"[m05] wrote {METRICS_PATH}")
    for k, v in metrics.items():
        print(f"        {k}: {v}")


if __name__ == "__main__":
    main()
