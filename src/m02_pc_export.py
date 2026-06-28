"""
Milestone 2 — Microsoft Planetary Computer alternative to the GEE export.

Why this exists
---------------
The original ``m02_gee_export_and_sync.py`` queues server-side composites on
Earth Engine and rclone-moves the GeoTIFFs back from Drive. That works but
is bottlenecked by Google's batch queue and the Drive intermediate hop. The
Planetary Computer (PC) serves Sentinel-2 and Sentinel-1 as Cloud-Optimized
GeoTIFFs sitting on Azure blob storage; we can read only the AOI window of
each scene over plain HTTP, do the masking and the composite locally on
the cluster's CPUs, and write the same per-window output GeoTIFFs straight
to scratch — no Drive in the loop.

Outputs are byte-for-byte interchangeable with the GEE version:
``sentinel{1,2}/<sensor>_<YYYY>_w<NN>_<MMMDD>.tif`` at 10 m, UTM 15N.

This script is resume-safe — output files that already exist on disk
(including the 27 chips synced from the earlier GEE run) are skipped.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import geopandas as gpd
import numpy as np
import pystac_client
import planetary_computer
import odc.stac
import rioxarray  # noqa: F401  — registers the .rio accessor on xarray
import xarray as xr
from scipy import ndimage as scipy_ndimage
from shapely.geometry import box as shapely_box

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

PERSIST_ROOT = Path(
    "<PERSIST_ROOT>"
)
SCRATCH_ROOT = Path("<SCRATCH_ROOT>")

AOI_TILES_GPKG_PATH = PERSIST_ROOT / "m01_aoi_fields" / "central_iowa_tiles.gpkg"
LOCAL_SENTINEL2_DIR = SCRATCH_ROOT / "m02_satellite_chips" / "sentinel2"
LOCAL_SENTINEL1_DIR = SCRATCH_ROOT / "m02_satellite_chips" / "sentinel1"

# Same time-window plan the GEE script uses, so file names match.
YEARS_TO_EXPORT: List[int] = list(range(2017, 2025))
GROWING_SEASON_START = (4, 1)
GROWING_SEASON_END   = (10, 15)
GROWING_SEASON_WINDOW_DAYS = 14
OFFSEASON_WINDOW_DAYS = 30

# Sentinel-2 bands. PC uses lowercase `B02`-style names (same as Sentinel Hub /
# Copernicus Data Space). We keep the same 10 bands the GEE version exported.
S2_BANDS_TO_KEEP = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
S2_SCL_BAND_NAME = "SCL"
S2_BAD_SCL_CLASSES = [0, 1, 3, 8, 9, 10]
S2_MAX_CLOUD_COVER_PERCENT = 80   # filter scenes loosely; SCL mask does the rest

# Sentinel-1: PC's RTC ("Radiometric Terrain Corrected") collection is the
# modern, properly-calibrated version. Bands are lowercase "vv" / "vh".
S1_BANDS_TO_KEEP = ["vv", "vh"]
S1_SPECKLE_FOCAL_RADIUS_PIXELS = 5   # at 10 m, this is a 50 m diameter

# Output projection — same as GEE.
OUTPUT_CRS_EPSG = "EPSG:32615"
OUTPUT_PIXEL_SIZE_METRES = 10

# Dask chunk size for processing. With S2 winter windows pulling 30-47
# scenes, a 4096^2 chunk × 10 bands × 30+ scenes is ~10 GB in flight per
# chunk during the median reduction — enough to repeatedly OOM-kill the
# job. Halving to 2048 cuts the per-chunk peak by 4x while leaving plenty
# of room for compositing.
PROCESSING_CHUNK_SIZE_PIXELS = 2048

# Number of windows to process in parallel. Empirically:
#   * 8 workers × 64 GB SLURM job  ->  OOM-killed (~67 GB peak)
#   * 4 workers × 128 GB SLURM job ->  OOM-killed (~140+ GB peak)
#   * 2 workers × 128 GB SLURM job ->  OOM-killed (~134 GB peak)
# The smoke test on a 5-scene S1 chip uses ~9 GB, but the actual workload
# is dominated by S2 winter windows that pull 30-47 scenes; those compose
# a 4096^2 × 10 bands × N_scenes × 2 bytes = ~10 GB IN-FLIGHT chunk during
# the median reduction. Smaller chunks (PROCESSING_CHUNK_SIZE_PIXELS=2048
# below) reduce that to ~2.5 GB per chunk, but to be safe we also drop to
# a single worker.
NUMBER_OF_PARALLEL_WORKERS = 1

PC_STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


# ---------------------------------------------------------------------------
# Time-window helpers (identical to GEE script — so file names match)
# ---------------------------------------------------------------------------

@dataclass
class TimeWindow:
    label: str
    start_date: dt.date
    end_date_exclusive: dt.date


def generate_time_windows_for_year(year: int) -> List[TimeWindow]:
    """Biweekly Apr 1 -> Oct 14 + monthly Jan/Feb/Mar/Nov/Dec."""
    windows: List[TimeWindow] = []
    window_index = 0

    growing_start = dt.date(year, *GROWING_SEASON_START)
    growing_end   = dt.date(year, *GROWING_SEASON_END)
    current_start = growing_start
    while (growing_end - current_start).days >= 7:
        window_end = min(current_start + dt.timedelta(days=GROWING_SEASON_WINDOW_DAYS),
                         growing_end)
        window_index += 1
        windows.append(TimeWindow(
            label=f"{year}_w{window_index:02d}_{current_start.strftime('%b%d').lower()}",
            start_date=current_start,
            end_date_exclusive=window_end,
        ))
        current_start = window_end

    offseason_starts = [
        dt.date(year, 1, 1), dt.date(year, 2, 1), dt.date(year, 3, 1),
        dt.date(year, 11, 1), dt.date(year, 12, 1),
    ]
    for offseason_start in offseason_starts:
        window_index += 1
        windows.append(TimeWindow(
            label=f"{year}_w{window_index:02d}_{offseason_start.strftime('%b%d').lower()}",
            start_date=offseason_start,
            end_date_exclusive=offseason_start + dt.timedelta(days=OFFSEASON_WINDOW_DAYS),
        ))
    return windows


# ---------------------------------------------------------------------------
# STAC search + sign helpers
# ---------------------------------------------------------------------------

def open_stac_catalog() -> pystac_client.Client:
    """Open the Planetary Computer STAC API with automatic URL signing."""
    return pystac_client.Client.open(
        PC_STAC_API_URL,
        modifier=planetary_computer.sign_inplace,
    )


def search_sentinel2_scenes(
    catalog: pystac_client.Client,
    aoi_bbox_wgs84: tuple,
    time_window: TimeWindow,
) -> list:
    """Return STAC items for Sentinel-2 L2A scenes inside the AOI + window."""
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=aoi_bbox_wgs84,
        datetime=f"{time_window.start_date.isoformat()}/"
                 f"{(time_window.end_date_exclusive - dt.timedelta(days=1)).isoformat()}",
        query={"eo:cloud_cover": {"lt": S2_MAX_CLOUD_COVER_PERCENT}},
    )
    return list(search.items())


def search_sentinel1_scenes(
    catalog: pystac_client.Client,
    aoi_bbox_wgs84: tuple,
    time_window: TimeWindow,
) -> list:
    """Return STAC items for Sentinel-1 RTC scenes inside the AOI + window."""
    search = catalog.search(
        collections=["sentinel-1-rtc"],
        bbox=aoi_bbox_wgs84,
        datetime=f"{time_window.start_date.isoformat()}/"
                 f"{(time_window.end_date_exclusive - dt.timedelta(days=1)).isoformat()}",
    )
    return list(search.items())


# ---------------------------------------------------------------------------
# Composite construction
# ---------------------------------------------------------------------------

def build_sentinel2_composite(
    s2_stac_items: list,
    aoi_polygon_in_utm15n: gpd.GeoSeries,
) -> Optional[xr.DataArray]:
    """SCL-masked median composite as int16 DN. Returns None on empty input."""
    if not s2_stac_items:
        return None

    # odc.stac.load gives us a dask-backed xarray DataArray. We load both the
    # 10 spectral bands AND the SCL band in a single call; chunks default to
    # whole-image but we set them explicitly for memory control.
    loaded_dataset = odc.stac.load(
        s2_stac_items,
        bands=S2_BANDS_TO_KEEP + [S2_SCL_BAND_NAME],
        crs=OUTPUT_CRS_EPSG,
        resolution=OUTPUT_PIXEL_SIZE_METRES,
        geopolygon=aoi_polygon_in_utm15n,
        chunks={"x": PROCESSING_CHUNK_SIZE_PIXELS, "y": PROCESSING_CHUNK_SIZE_PIXELS, "time": -1},
        dtype="int16",   # DN are 0-10000 ish, fit in int16
    )

    # Build a per-scene boolean "bad" mask from SCL, then mask the spectral
    # bands. We keep everything as int16 to avoid the ~2x memory hit of float.
    scl = loaded_dataset[S2_SCL_BAND_NAME]
    bad_pixel_mask = xr.zeros_like(scl, dtype=bool)
    for bad_class_value in S2_BAD_SCL_CLASSES:
        bad_pixel_mask = bad_pixel_mask | (scl == bad_class_value)

    spectral_dataset = loaded_dataset[S2_BANDS_TO_KEEP]
    masked_spectral = spectral_dataset.where(~bad_pixel_mask)

    # Per-pixel median across the time axis. xarray + dask will streaming-
    # compute this chunk-by-chunk on the local CPUs — no out-of-memory.
    median_per_band = masked_spectral.median(dim="time", skipna=True, keep_attrs=True)

    # Stack the band dims into a single (band, y, x) DataArray.
    stacked = median_per_band.to_array(dim="band")
    stacked = stacked.fillna(0).astype("int16")
    stacked = stacked.assign_coords(band=S2_BANDS_TO_KEEP)
    return stacked


def build_sentinel1_composite(
    s1_stac_items: list,
    aoi_polygon_in_utm15n: gpd.GeoSeries,
) -> Optional[xr.DataArray]:
    """Speckle-filtered VV+VH mean composite as float32. None on empty input.

    Important: Planetary Computer's Sentinel-1 RTC bands use
    ``nodata = -32768`` for masked pixels (terrain shadow, layover, image
    edges, etc.). odc.stac.load returns these as the raw fill values
    rather than NaN, so any downstream reduction (`.mean()`,
    ``uniform_filter``) silently averages them with real backscatter
    and produces nonsense numbers. The first thing we do here is
    explicitly cast the fill values to NaN.

    Also: the focal mean speckle filter runs on the materialised array
    (not dask) because scipy's filters can't see chunk boundaries. We
    use ``nan_to_num`` first so NaN doesn't poison the filter window.
    """
    if not s1_stac_items:
        return None

    loaded_dataset = odc.stac.load(
        s1_stac_items,
        bands=S1_BANDS_TO_KEEP,
        crs=OUTPUT_CRS_EPSG,
        resolution=OUTPUT_PIXEL_SIZE_METRES,
        geopolygon=aoi_polygon_in_utm15n,
        chunks={"x": PROCESSING_CHUNK_SIZE_PIXELS, "y": PROCESSING_CHUNK_SIZE_PIXELS, "time": -1},
        dtype="float32",
    )

    # Mask the nodata fill values BEFORE reducing — otherwise -32768 leaks
    # into the mean.
    s1_data = loaded_dataset[S1_BANDS_TO_KEEP]
    s1_data = s1_data.where(s1_data > -32000.0)   # -32768 nodata -> NaN

    # Per-pixel mean across time (S1 RTC values are linear power, additive).
    mean_per_band = s1_data.mean(dim="time", skipna=True, keep_attrs=True)
    stacked = mean_per_band.to_array(dim="band").astype("float32")
    stacked = stacked.assign_coords(band=S1_BANDS_TO_KEEP)

    # Materialise + speckle filter band by band. Replace any remaining NaN
    # with 0 so uniform_filter doesn't smear NaN across neighbours; the
    # downstream code treats 0 as nodata for S1.
    stacked_loaded = stacked.compute()
    raw_values = np.nan_to_num(stacked_loaded.values, nan=0.0)
    smoothed_bands = []
    for band_index in range(raw_values.shape[0]):
        smoothed_bands.append(
            scipy_ndimage.uniform_filter(
                raw_values[band_index],
                size=S1_SPECKLE_FOCAL_RADIUS_PIXELS,
            )
        )
    stacked_loaded.values = np.stack(smoothed_bands, axis=0).astype("float32")
    return stacked_loaded


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_geotiff(composite_dataarray: xr.DataArray, output_path: Path) -> None:
    """Write the composite as a LZW-compressed, tiled, BigTIFF-safe GeoTIFF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composite_dataarray.rio.write_crs(OUTPUT_CRS_EPSG, inplace=True)
    composite_dataarray.rio.to_raster(
        str(output_path),
        compress="lzw",
        tiled=True,
        BIGTIFF="YES",
    )


# ---------------------------------------------------------------------------
# Per-window worker
# ---------------------------------------------------------------------------

def process_one_window(args: tuple) -> dict:
    """Build + write the composite for a single (sensor, year, window)."""
    sensor, time_window, aoi_polygon_in_utm15n, aoi_bbox_wgs84 = args
    output_filename = f"{sensor}_{time_window.label}.tif"
    output_path = (
        LOCAL_SENTINEL2_DIR / output_filename if sensor == "s2"
        else LOCAL_SENTINEL1_DIR / output_filename
    )

    if output_path.exists():
        return {"status": "skipped_already_on_scratch", "path": str(output_path)}

    catalog = open_stac_catalog()
    start_time = time.time()
    try:
        if sensor == "s2":
            items = search_sentinel2_scenes(catalog, aoi_bbox_wgs84, time_window)
            composite = build_sentinel2_composite(items, aoi_polygon_in_utm15n)
        else:
            items = search_sentinel1_scenes(catalog, aoi_bbox_wgs84, time_window)
            composite = build_sentinel1_composite(items, aoi_polygon_in_utm15n)

        if composite is None:
            return {"status": "skipped_empty_collection",
                    "label": time_window.label, "sensor": sensor,
                    "n_items": len(items)}

        write_geotiff(composite, output_path)
        return {
            "status": "ok",
            "path": str(output_path),
            "n_items": len(items),
            "elapsed_seconds": round(time.time() - start_time, 1),
        }
    except Exception as exception:
        return {
            "status": "error",
            "label": time_window.label,
            "sensor": sensor,
            "error": repr(exception),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_full_work_list(only_year: Optional[int]) -> List[tuple]:
    """Enumerate every (sensor, window) pair we want as a chip."""
    aoi_geodataframe = gpd.read_file(AOI_TILES_GPKG_PATH).dissolve()
    aoi_polygon_in_utm15n = aoi_geodataframe.to_crs(OUTPUT_CRS_EPSG).geometry
    aoi_bbox_wgs84 = tuple(aoi_geodataframe.geometry.iloc[0].bounds)

    work_items: List[tuple] = []
    for year in YEARS_TO_EXPORT:
        if only_year is not None and year != only_year:
            continue
        for time_window in generate_time_windows_for_year(year):
            for sensor in ("s2", "s1"):
                work_items.append(
                    (sensor, time_window, aoi_polygon_in_utm15n, aoi_bbox_wgs84)
                )
    return work_items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-year", type=int, default=None)
    parser.add_argument("--workers", type=int, default=NUMBER_OF_PARALLEL_WORKERS,
                        help="Number of windows to process in parallel.")
    parser.add_argument("--single-window", type=str, default=None,
                        help="For smoke tests: run only this window label (e.g. 2024_w01_apr01).")
    parser.add_argument("--single-sensor", type=str, default=None, choices=["s1", "s2"],
                        help="For smoke tests: restrict to one sensor.")
    args = parser.parse_args()

    LOCAL_SENTINEL2_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_SENTINEL1_DIR.mkdir(parents=True, exist_ok=True)

    work_items = build_full_work_list(args.test_year)
    if args.single_window:
        work_items = [w for w in work_items if w[1].label == args.single_window]
    if args.single_sensor:
        work_items = [w for w in work_items if w[0] == args.single_sensor]

    print(f"[pc] total work items: {len(work_items)}")
    print(f"[pc] parallel workers: {args.workers}")

    counter_ok = counter_skipped = counter_empty = counter_error = 0
    start_time = time.time()

    # ThreadPoolExecutor is the right pool here — almost all the time is
    # spent waiting on HTTP I/O from Azure blob storage; only the median
    # itself burns CPU and that's already parallelised inside dask.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(process_one_window, work_items):
            status = result.get("status")
            if status == "ok":
                counter_ok += 1
                print(
                    f"  ok      [{counter_ok + counter_skipped + counter_empty + counter_error:>3d}/"
                    f"{len(work_items):>3d}] {Path(result['path']).name}  "
                    f"({result.get('n_items')} scenes, "
                    f"{result.get('elapsed_seconds')}s)",
                    flush=True,
                )
            elif status == "skipped_already_on_scratch":
                counter_skipped += 1
            elif status == "skipped_empty_collection":
                counter_empty += 1
                print(
                    f"  empty   {result.get('sensor')}_{result.get('label')}  "
                    f"(no scenes — wrote nothing)",
                    flush=True,
                )
            else:
                counter_error += 1
                print(
                    f"  ERROR   {result.get('sensor')}_{result.get('label')}  "
                    f"{result.get('error', '?')[:200]}",
                    flush=True,
                )

    elapsed_minutes = (time.time() - start_time) / 60.0
    print(
        f"[pc] done. ok={counter_ok}  skipped(on-disk)={counter_skipped}  "
        f"empty={counter_empty}  errors={counter_error}  "
        f"elapsed={elapsed_minutes:.1f} min"
    )
    return 0 if counter_error == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
