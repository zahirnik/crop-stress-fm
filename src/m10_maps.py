"""
Milestone 10 — Iowa 2024 stress maps.

For each 2024 window we render a clean per-tile choropleth coloured by
the Clay + GRU probability of stress (p_stressed). We also save:

  * one cumulative map of how many windows each tile was flagged stressed
    in 2024 (the model's "annual stress hotspots" view), and
  * one 3-panel comparison map for the Sep 16 event — NDVI baseline vs
    Clay + GRU vs proxy label — to make the headline finding visible.

Maps are rendered at 300 DPI, with the county boundary overlay, a clean
neutral background, and a colour bar.

Outputs go to:
    <SCRATCH_ROOT>/m10_outputs/maps/
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


PERSIST_ROOT = Path(
    "<PERSIST_ROOT>"
)
SCRATCH_ROOT = Path("<SCRATCH_ROOT>")

AOI_TILES_GPKG_PATH = PERSIST_ROOT / "m01_aoi_fields" / "central_iowa_tiles.gpkg"
COUNTIES_GPKG_PATH  = SCRATCH_ROOT / "m01_aoi_fields" / "counties_tiger" / "tl_2024_us_county.shp"
GRU_PREDS_PATH      = SCRATCH_ROOT / "m07_path_a_gru" / "path_a_predictions.parquet"
NDVI_PREDS_PATH     = SCRATCH_ROOT / "m06_baseline"   / "baseline_predictions.parquet"
LABELS_PATH         = SCRATCH_ROOT / "m04_proxy_labels" / "labels.parquet"

OUTPUT_DIR = SCRATCH_ROOT / "m10_outputs" / "maps"

PLOT_CRS_EPSG = "EPSG:32615"          # UTM 15N matches the chips
SCORE_YEAR = 2024
WINDOWS_PER_YEAR = 19

GRU_BEST_F1_THRESHOLD = 0.00176       # from m07 metrics
NDVI_BEST_F1_THRESHOLD = -0.107       # from m06 metrics (score = -ndvi)

MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
WINDOW_START_DATES = {  # 2024 calendar — biweekly growing season + monthly off-season
    1: dt.date(2024, 4, 1),
    2: dt.date(2024, 4, 15),
    3: dt.date(2024, 4, 29),
    4: dt.date(2024, 5, 13),
    5: dt.date(2024, 5, 27),
    6: dt.date(2024, 6, 10),
    7: dt.date(2024, 6, 24),
    8: dt.date(2024, 7, 8),
    9: dt.date(2024, 7, 22),
    10: dt.date(2024, 8, 5),
    11: dt.date(2024, 8, 19),
    12: dt.date(2024, 9, 2),
    13: dt.date(2024, 9, 16),
    14: dt.date(2024, 9, 30),
    15: dt.date(2024, 1, 1),
    16: dt.date(2024, 2, 1),
    17: dt.date(2024, 3, 1),
    18: dt.date(2024, 11, 1),
    19: dt.date(2024, 12, 1),
}

STRESS_CMAP = LinearSegmentedColormap.from_list(
    "stress",
    [(0.0, "#f7fcf5"), (0.25, "#a1d99b"), (0.5, "#fec44f"), (0.75, "#e34a33"), (1.0, "#67000d")],
)


def load_aoi_tiles_in_plot_crs() -> gpd.GeoDataFrame:
    tiles = gpd.read_file(AOI_TILES_GPKG_PATH)
    return tiles.to_crs(PLOT_CRS_EPSG)


def load_six_aoi_counties_in_plot_crs() -> Optional[gpd.GeoDataFrame]:
    if not COUNTIES_GPKG_PATH.exists():
        return None
    counties = gpd.read_file(COUNTIES_GPKG_PATH)
    iowa_county_fips = {
        "19169", "19079", "19015", "19127", "19083", "19153",   # Story, Hamilton, Boone, Marshall, Hardin, Polk
    }
    iowa_six = counties[counties["GEOID"].isin(iowa_county_fips)]
    return iowa_six.to_crs(PLOT_CRS_EPSG)


def apply_pretty_style(ax: plt.Axes, tiles: gpd.GeoDataFrame, counties: Optional[gpd.GeoDataFrame]) -> None:
    minx, miny, maxx, maxy = tiles.total_bounds
    padding = 5000.0
    ax.set_xlim(minx - padding, maxx + padding)
    ax.set_ylim(miny - padding, maxy + padding)
    ax.set_facecolor("#ffffff")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if counties is not None:
        counties.boundary.plot(ax=ax, color="#222222", linewidth=1.2, alpha=0.9, zorder=4)
        for _, county_row in counties.iterrows():
            centroid = county_row.geometry.centroid
            label_text = county_row.get("NAME", "")
            label = ax.text(
                centroid.x, centroid.y, label_text,
                ha="center", va="center",
                color="#111111", fontsize=8, fontweight="bold",
                zorder=5,
            )
            label.set_path_effects([
                patheffects.Stroke(linewidth=2.0, foreground="white"),
                patheffects.Normal(),
            ])


def draw_per_tile_choropleth(
    ax: plt.Axes,
    tiles: gpd.GeoDataFrame,
    counties: Optional[gpd.GeoDataFrame],
    value_column: str,
    value_min: float,
    value_max: float,
    title: str,
    cbar_label: str,
) -> None:
    apply_pretty_style(ax, tiles, counties)
    tiles.plot(
        ax=ax,
        column=value_column,
        cmap=STRESS_CMAP,
        vmin=value_min, vmax=value_max,
        edgecolor="#ffffff",
        linewidth=0.25,
        legend=False,
        zorder=2,
    )
    scalar_mappable = plt.cm.ScalarMappable(
        cmap=STRESS_CMAP,
        norm=plt.Normalize(vmin=value_min, vmax=value_max),
    )
    scalar_mappable.set_array([])
    cbar = plt.colorbar(scalar_mappable, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(cbar_label, color="#111111", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#111111")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#111111")
    ax.set_title(title, color="#111111", fontsize=11, pad=8)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tiles = load_aoi_tiles_in_plot_crs()
    counties = load_six_aoi_counties_in_plot_crs()
    if counties is None:
        print("[m10] WARNING: counties shapefile not at expected path — maps will have no county overlay")
    print(f"[m10] {len(tiles)} tiles loaded")

    gru_predictions  = pd.read_parquet(GRU_PREDS_PATH)
    ndvi_predictions = pd.read_parquet(NDVI_PREDS_PATH)
    labels_df        = pd.read_parquet(LABELS_PATH)
    print(f"[m10] {len(gru_predictions)} GRU preds  {len(ndvi_predictions)} NDVI preds  {len(labels_df)} labels")

    gru_predictions = gru_predictions[gru_predictions["year"] == SCORE_YEAR]
    ndvi_predictions = ndvi_predictions[ndvi_predictions["year"] == SCORE_YEAR]
    labels_df = labels_df[labels_df["year"] == SCORE_YEAR]

    # ---- per-window stress maps (GRU probability) ----
    n_windows_rendered = 0
    for window_index in sorted(WINDOW_START_DATES.keys()):
        window_predictions = gru_predictions[gru_predictions["window_index"] == window_index]
        if window_predictions.empty:
            continue
        tile_value_map = window_predictions.set_index("tile_id")["p_stressed"].astype(float)
        tiles_for_window = tiles.copy()
        tiles_for_window["p_stressed"] = tiles_for_window["tile_id"].map(tile_value_map)

        fig, ax = plt.subplots(figsize=(8, 8), facecolor="#ffffff")
        draw_per_tile_choropleth(
            ax, tiles_for_window, counties,
            value_column="p_stressed", value_min=0.0, value_max=1.0,
            title=f"Clay + GRU stress probability — {WINDOW_START_DATES[window_index].isoformat()}\n"
                  f"Central Iowa AOI, 5.12 km tiles",
            cbar_label="p(stressed)",
        )
        out_path = OUTPUT_DIR / f"stress_map_2024_w{window_index:02d}_{WINDOW_START_DATES[window_index].strftime('%b%d').lower()}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#ffffff")
        plt.close(fig)
        n_windows_rendered += 1
        print(f"[m10] wrote {out_path}")
    print(f"[m10] rendered {n_windows_rendered} per-window maps")

    # ---- annual stress count map ----
    gru_predictions = gru_predictions.assign(
        pred_stressed=(gru_predictions["p_stressed"] >= GRU_BEST_F1_THRESHOLD).astype(int)
    )
    n_stressed_per_tile = gru_predictions.groupby("tile_id")["pred_stressed"].sum()
    tiles_annual = tiles.copy()
    tiles_annual["n_stressed_windows"] = tiles_annual["tile_id"].map(n_stressed_per_tile).astype(float)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#ffffff")
    draw_per_tile_choropleth(
        ax, tiles_annual, counties,
        value_column="n_stressed_windows",
        value_min=0.0,
        value_max=float(tiles_annual["n_stressed_windows"].max() or 1.0),
        title=f"Annual stress hotspots — {SCORE_YEAR}\n"
              f"# of windows each tile flagged stressed by Clay + GRU",
        cbar_label="# windows flagged stressed",
    )
    out_path = OUTPUT_DIR / f"annual_stress_count_{SCORE_YEAR}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"[m10] wrote {out_path}")

    # ---- 3-panel comparison for the headline Sep 16 event ----
    headline_window_index = 13
    headline_date = WINDOW_START_DATES[headline_window_index].isoformat()

    gru_headline_row = gru_predictions[gru_predictions["window_index"] == headline_window_index].set_index("tile_id")
    ndvi_headline_row = ndvi_predictions[ndvi_predictions["window_index"] == headline_window_index].set_index("tile_id")
    label_headline_row = labels_df[labels_df["window_index"] == headline_window_index].set_index("tile_id")

    tiles_headline = tiles.copy()
    tiles_headline["clay_gru_p"] = tiles_headline["tile_id"].map(gru_headline_row["p_stressed"].astype(float))
    tiles_headline["ndvi_score"] = tiles_headline["tile_id"].map(ndvi_headline_row["score_neg_ndvi"].astype(float))
    tiles_headline["label_z15"]  = tiles_headline["tile_id"].map(label_headline_row["label_z15"].astype(float))

    fig, axes = plt.subplots(1, 3, figsize=(22, 8), facecolor="#ffffff")
    draw_per_tile_choropleth(
        axes[0], tiles_headline, counties,
        value_column="ndvi_score",
        value_min=float(np.nanpercentile(tiles_headline["ndvi_score"], 5)),
        value_max=float(np.nanpercentile(tiles_headline["ndvi_score"], 95)),
        title=f"NDVI baseline score (-NDVI)\n{headline_date}",
        cbar_label="-NDVI (higher = more stressed)",
    )
    draw_per_tile_choropleth(
        axes[1], tiles_headline, counties,
        value_column="clay_gru_p", value_min=0.0, value_max=1.0,
        title=f"Clay + GRU p(stressed)\n{headline_date}",
        cbar_label="p(stressed)",
    )
    draw_per_tile_choropleth(
        axes[2], tiles_headline, counties,
        value_column="label_z15", value_min=0.0, value_max=1.0,
        title=f"Proxy label (NDVI z < -1.5)\n{headline_date}",
        cbar_label="stressed (0/1)",
    )
    out_path = OUTPUT_DIR / f"comparison_2024_w{headline_window_index:02d}_sep16.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"[m10] wrote {out_path}")

    print(f"[m10] done — outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
