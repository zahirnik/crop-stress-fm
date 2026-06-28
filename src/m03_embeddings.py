"""
Milestone 3 — Clay v1.5 frozen-encoder embeddings.

Inputs (from M2):
    <SCRATCH_ROOT>/m02_satellite_chips/
        sentinel2/s2_<YYYY>_w<NN>_<MMMDD>.tif    (10 bands, 12k x 12k, AOI-wide)
        sentinel1/s1_<YYYY>_w<NN>_<MMMDD>.tif    (2 bands, same grid, linear power)

Outputs:
    <SCRATCH_ROOT>/m03_embeddings/
        sentinel2.zarr   -- shape [tile, window, 1024], float32
        sentinel1.zarr   -- shape [tile, window, 1024], float32

For each (sensor, window) composite:
  1. Window-read each of the 320 tile sub-chips (512x512 px @ 10 m).
  2. Quadrant-tile to 4x (256x256) — Clay v1.5 expects 256.
  3. Normalize per-band using Clay's metadata.yaml stats (S1: linear -> dB first).
  4. Build the Clay input dict (pixels, time, latlon, waves, gsd).
  5. Forward through the frozen encoder; mean-pool spatial tokens; mean across quadrants
     -> one (tile, window, 1024) embedding.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
import torch
import yaml
import zarr
from rasterio.windows import from_bounds as window_from_bounds


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

PERSIST_ROOT = Path(
    "<PERSIST_ROOT>"
)
SCRATCH_ROOT = Path("<SCRATCH_ROOT>")

AOI_TILES_GPKG_PATH = PERSIST_ROOT / "m01_aoi_fields" / "central_iowa_tiles.gpkg"
SENTINEL2_DIR = SCRATCH_ROOT / "m02_satellite_chips" / "sentinel2"
SENTINEL1_DIR = SCRATCH_ROOT / "m02_satellite_chips" / "sentinel1"

OUTPUT_DIR = SCRATCH_ROOT / "m03_embeddings"

CLAY_CHECKPOINT_PATH = PERSIST_ROOT / "clay_weights" / "clay-v1.5.ckpt"
CLAY_METADATA_PATH   = PERSIST_ROOT / "clay_weights" / "metadata.yaml"

# Clay v1.5 "large" model — embedding dim = 1024.
CLAY_MODEL_SIZE = "large"
CLAY_EMBED_DIM  = 1024
CLAY_CHIP_SIZE  = 256       # what Clay was trained on
TILE_PIXELS     = 512       # what M2 wrote per tile
N_QUADRANTS     = (TILE_PIXELS // CLAY_CHIP_SIZE) ** 2   # 4

# Output projection of M2.
COMPOSITE_CRS_EPSG = "EPSG:32615"

# Sentinel-2 band order (matches Clay's metadata.yaml exactly).
S2_BAND_ORDER = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
S2_CLAY_KEYS  = ["blue", "green", "red", "rededge1", "rededge2", "rededge3",
                 "nir", "nir08", "swir16", "swir22"]

# Sentinel-1 band order (matches Clay's metadata.yaml exactly).
S1_BAND_ORDER = ["VV", "VH"]
S1_CLAY_KEYS  = ["vv", "vh"]

# Nodata sentinels in M2 outputs.
S2_NODATA_VALUE_INT16   = 0          # produced by M2 nan_to_num
S1_NODATA_VALUE_FLOAT32 = 0.0        # produced by M2 nan_to_num


# ---------------------------------------------------------------------------
# Tile + window metadata helpers
# ---------------------------------------------------------------------------

CHIP_FILENAME_RE = re.compile(r"^s[12]_(\d{4})_w(\d{2})_([a-z]{3})(\d{2})\.tif$")

MONTH_NAMES = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]


def parse_window_date_from_filename(path: Path) -> Tuple[int, int, dt.date]:
    """('s2_2019_w17_mar01.tif') -> (year=2019, window_index=17, start_date=2019-03-01)."""
    m = CHIP_FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Unrecognised chip filename: {path.name}")
    year = int(m.group(1))
    window_index = int(m.group(2))
    month_name = m.group(3)
    day = int(m.group(4))
    month = MONTH_NAMES.index(month_name) + 1
    return year, window_index, dt.date(year, month, day)


def load_tile_table_in_utm() -> gpd.GeoDataFrame:
    """Tile GeoPackage in UTM 15N (matches composite CRS)."""
    tiles_in_wgs84 = gpd.read_file(AOI_TILES_GPKG_PATH)
    tiles_in_wgs84 = tiles_in_wgs84.sort_values("tile_id").reset_index(drop=True)
    return tiles_in_wgs84.to_crs(COMPOSITE_CRS_EPSG)


# ---------------------------------------------------------------------------
# Sub-chip extraction (per-tile window read from the big AOI composite)
# ---------------------------------------------------------------------------

def extract_per_tile_subchips(
    composite_path: Path,
    tiles_in_utm: gpd.GeoDataFrame,
    n_bands_expected: int,
) -> np.ndarray:
    """
    For each tile, window-read a TILE_PIXELS x TILE_PIXELS sub-chip from the
    composite. Returns ndarray shape [n_tiles, n_bands, TILE_PIXELS, TILE_PIXELS]
    (dtype = whatever the source file is).
    """
    n_tiles = len(tiles_in_utm)
    out: Optional[np.ndarray] = None
    bounds_xy = tiles_in_utm.geometry.bounds.to_numpy()  # (n_tiles, 4)  xmin,ymin,xmax,ymax

    with rasterio.open(composite_path) as src:
        if src.count != n_bands_expected:
            raise ValueError(
                f"{composite_path.name}: expected {n_bands_expected} bands, got {src.count}"
            )
        for tile_index in range(n_tiles):
            xmin, ymin, xmax, ymax = bounds_xy[tile_index]
            read_window = window_from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)
            chip = src.read(
                window=read_window,
                out_shape=(n_bands_expected, TILE_PIXELS, TILE_PIXELS),
                boundless=True,
                fill_value=0,
            )
            if out is None:
                out = np.empty((n_tiles, n_bands_expected, TILE_PIXELS, TILE_PIXELS),
                               dtype=chip.dtype)
            out[tile_index] = chip
    return out


def quadrant_tile(chips: np.ndarray) -> np.ndarray:
    """
    [n_tiles, C, 512, 512] -> [n_tiles*4, C, 256, 256]
    Order within each tile: (top-left, top-right, bottom-left, bottom-right).
    """
    n_tiles, n_bands, h, w = chips.shape
    half_h, half_w = h // 2, w // 2
    out = np.empty((n_tiles * 4, n_bands, half_h, half_w), dtype=chips.dtype)
    out[0::4] = chips[:, :, :half_h, :half_w]
    out[1::4] = chips[:, :, :half_h, half_w:]
    out[2::4] = chips[:, :, half_h:, :half_w]
    out[3::4] = chips[:, :, half_h:, half_w:]
    return out


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def load_clay_metadata() -> dict:
    with open(CLAY_METADATA_PATH) as f:
        return yaml.safe_load(f)


def s2_normalize(chips_int16: np.ndarray, metadata: dict) -> Tuple[np.ndarray, List[float]]:
    """[B,10,H,W] int16 reflectance DN -> normalized float32 + Clay wavelength list."""
    s2_meta = metadata["sentinel-2-l2a"]
    means = np.array([s2_meta["bands"]["mean"][k] for k in S2_CLAY_KEYS], dtype=np.float32)
    stds  = np.array([s2_meta["bands"]["std"][k]  for k in S2_CLAY_KEYS], dtype=np.float32)
    waves = [s2_meta["bands"]["wavelength"][k] for k in S2_CLAY_KEYS]

    pixels = chips_int16.astype(np.float32)
    pixels = (pixels - means[None, :, None, None]) / stds[None, :, None, None]
    return pixels, waves


def s1_normalize(chips_float32_db: np.ndarray, metadata: dict) -> Tuple[np.ndarray, List[float]]:
    """[B,2,H,W] float32 dB -> normalized + Clay wavelength list.

    The M2 PC pipeline writes RTC backscatter in dB (not linear power as an
    earlier comment claimed). NaN-marked nodata pixels (outside the AOI mask
    on rectangular chip bounds) are filled with the band mean so they become
    zero after normalization.
    """
    s1_meta = metadata["sentinel-1-rtc"]
    means = np.array([s1_meta["bands"]["mean"][k] for k in S1_CLAY_KEYS], dtype=np.float32)
    stds  = np.array([s1_meta["bands"]["std"][k]  for k in S1_CLAY_KEYS], dtype=np.float32)
    waves = [s1_meta["bands"]["wavelength"][k] for k in S1_CLAY_KEYS]

    pixels_db = chips_float32_db.astype(np.float32, copy=True)
    for band_index in range(pixels_db.shape[1]):
        nan_mask = ~np.isfinite(pixels_db[:, band_index])
        if nan_mask.any():
            pixels_db[:, band_index][nan_mask] = means[band_index]
    pixels = (pixels_db - means[None, :, None, None]) / stds[None, :, None, None]
    return pixels, waves


# ---------------------------------------------------------------------------
# Clay metadata batch encodings
# ---------------------------------------------------------------------------

def build_time_vector(window_start_date: dt.date) -> Tuple[float, float, float, float]:
    """Clay's `time` is [week_norm_sin, week_norm_cos, hour_norm_sin, hour_norm_cos]."""
    week_of_year = window_start_date.isocalendar().week
    week_sin = math.sin(2 * math.pi * week_of_year / 52.0)
    week_cos = math.cos(2 * math.pi * week_of_year / 52.0)
    # No useful hour info for our daily composites — pick local solar noon.
    hour_sin = math.sin(2 * math.pi * 12.0 / 24.0)
    hour_cos = math.cos(2 * math.pi * 12.0 / 24.0)
    return week_sin, week_cos, hour_sin, hour_cos


def build_latlon_vector(centroid_lat_deg: float, centroid_lon_deg: float) -> Tuple[float, float, float, float]:
    """Clay's `latlon` is [lat_sin, lat_cos, lon_sin, lon_cos] of the chip centroid."""
    lat_rad = math.radians(centroid_lat_deg)
    lon_rad = math.radians(centroid_lon_deg)
    return math.sin(lat_rad), math.cos(lat_rad), math.sin(lon_rad), math.cos(lon_rad)


# ---------------------------------------------------------------------------
# Clay model wrapper
# ---------------------------------------------------------------------------

def load_clay_encoder(device: torch.device):
    """Returns a callable that takes a batch dict and returns the [B, T, D] patch tensor."""
    from claymodel.module import ClayMAEModule
    module = ClayMAEModule.load_from_checkpoint(
        checkpoint_path=str(CLAY_CHECKPOINT_PATH),
        model_size=CLAY_MODEL_SIZE,
        metadata_path=str(CLAY_METADATA_PATH),
        dolls=[16, 32, 64, 128, 256, 768, 1024],
        doll_weights=[1, 1, 1, 1, 1, 1, 1],
        mask_ratio=0.0,
        shuffle=False,
    )
    module.eval()
    module = module.to(device)
    return module


def embed_quadrants(
    encoder_module,
    pixels_normalized: np.ndarray,
    waves: List[float],
    gsd: float,
    time_vectors: np.ndarray,
    latlon_vectors: np.ndarray,
    device: torch.device,
    forward_batch_size: int = 32,
) -> np.ndarray:
    """
    pixels_normalized: [B, C, 256, 256] float32 (B = n_tiles * 4 quadrants).
    time_vectors:      [B, 4] float32   (already replicated to per-quadrant)
    latlon_vectors:    [B, 4] float32   (already replicated to per-quadrant)
    Returns: [B, embed_dim] float32 mean-pooled patch tokens.
    """
    n_inputs = pixels_normalized.shape[0]
    out = np.empty((n_inputs, CLAY_EMBED_DIM), dtype=np.float32)

    pixels_tensor = torch.from_numpy(pixels_normalized)
    time_tensor   = torch.from_numpy(time_vectors)
    latlon_tensor = torch.from_numpy(latlon_vectors)
    waves_tensor  = torch.tensor(waves, dtype=torch.float32, device=device)
    gsd_tensor    = torch.tensor(float(gsd), dtype=torch.float32, device=device)

    with torch.no_grad():
        for start in range(0, n_inputs, forward_batch_size):
            stop = min(start + forward_batch_size, n_inputs)
            batch_dict = {
                "pixels": pixels_tensor[start:stop].to(device, non_blocking=True),
                "time":   time_tensor[start:stop].to(device,  non_blocking=True),
                "latlon": latlon_tensor[start:stop].to(device, non_blocking=True),
                "waves":  waves_tensor,
                "gsd":    gsd_tensor,
            }
            patch_tokens, *_ = encoder_module.model.encoder(batch_dict)
            # Mean-pool over (cls + patches); shape [b, embed_dim].
            embeddings = patch_tokens.mean(dim=1).detach().cpu().float().numpy()
            out[start:stop] = embeddings
    return out


# ---------------------------------------------------------------------------
# Per-(sensor, year, window) embedding step
# ---------------------------------------------------------------------------

def embed_one_composite(
    composite_path: Path,
    sensor_key: str,
    tiles_in_utm: gpd.GeoDataFrame,
    centroid_lats: np.ndarray,
    centroid_lons: np.ndarray,
    metadata: dict,
    encoder_module,
    device: torch.device,
) -> np.ndarray:
    """Returns [n_tiles, 1024] float32 — one embedding per tile."""
    if sensor_key == "s2":
        n_bands = len(S2_BAND_ORDER)
        chips_int16 = extract_per_tile_subchips(composite_path, tiles_in_utm, n_bands)
        quadrants_int16 = quadrant_tile(chips_int16)        # [n_tiles*4, 10, 256, 256]
        pixels_normalized, waves = s2_normalize(quadrants_int16, metadata)
        gsd = metadata["sentinel-2-l2a"]["gsd"]
    elif sensor_key == "s1":
        n_bands = len(S1_BAND_ORDER)
        chips_f32 = extract_per_tile_subchips(composite_path, tiles_in_utm, n_bands)
        quadrants_f32 = quadrant_tile(chips_f32)
        pixels_normalized, waves = s1_normalize(quadrants_f32, metadata)
        gsd = metadata["sentinel-1-rtc"]["gsd"]
    else:
        raise ValueError(f"Unknown sensor: {sensor_key}")

    year, _, window_start_date = parse_window_date_from_filename(composite_path)
    time_vector_4d  = np.array(build_time_vector(window_start_date), dtype=np.float32)
    n_inputs = pixels_normalized.shape[0]
    time_vectors = np.broadcast_to(time_vector_4d, (n_inputs, 4)).copy()

    n_tiles = len(tiles_in_utm)
    latlon_per_tile = np.array(
        [build_latlon_vector(lat, lon) for lat, lon in zip(centroid_lats, centroid_lons)],
        dtype=np.float32,
    )
    latlon_vectors = np.repeat(latlon_per_tile, repeats=N_QUADRANTS, axis=0)

    quadrant_embeddings = embed_quadrants(
        encoder_module, pixels_normalized, waves, gsd,
        time_vectors, latlon_vectors, device,
    )
    per_tile_embeddings = quadrant_embeddings.reshape(n_tiles, N_QUADRANTS, CLAY_EMBED_DIM).mean(axis=1)
    return per_tile_embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def expected_window_count_per_year() -> int:
    """Sanity number — biweekly Apr1-Oct14 (14) + monthly off-season (5) = 19."""
    return 19


def list_composites(sensor_dir: Path, sensor_prefix: str, year_filter: Optional[int]) -> List[Path]:
    paths = sorted(sensor_dir.glob(f"{sensor_prefix}_*.tif"))
    if year_filter is not None:
        paths = [p for p in paths if f"_{year_filter}_" in p.name]
    return paths


def build_or_open_zarr(
    sensor_store_path: Path,
    n_tiles: int,
    expected_n_windows_per_year: int,
    years: List[int],
) -> Tuple[zarr.Group, np.ndarray, np.ndarray]:
    """Returns (root_group, embedding_array, mask_array)."""
    n_windows_total = expected_n_windows_per_year * len(years)
    sensor_store_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(sensor_store_path), mode="a")

    if "embedding" not in root:
        root.create_array(
            name="embedding",
            shape=(n_tiles, n_windows_total, CLAY_EMBED_DIM),
            chunks=(n_tiles, 1, CLAY_EMBED_DIM),
            dtype="float32",
            fill_value=0.0,
        )
    if "mask" not in root:
        root.create_array(
            name="mask",
            shape=(n_tiles, n_windows_total),
            chunks=(n_tiles, expected_n_windows_per_year),
            dtype="bool",
            fill_value=False,
        )
    return root, root["embedding"], root["mask"]


def window_global_index(year: int, window_index_in_year: int, base_year: int, per_year: int) -> int:
    return (year - base_year) * per_year + (window_index_in_year - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", choices=["s2", "s1", "both"], default="both")
    parser.add_argument("--year",   type=int, default=None,
                        help="Restrict to one year (smoke test).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--forward-batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"[m03] device:  {device}")
    print(f"[m03] loading tile table: {AOI_TILES_GPKG_PATH}")
    tiles_in_utm = load_tile_table_in_utm()
    n_tiles = len(tiles_in_utm)
    print(f"[m03] tiles:   {n_tiles}")

    centroid_lats = tiles_in_utm["centroid_lat"].to_numpy(dtype=np.float64)
    centroid_lons = tiles_in_utm["centroid_lon"].to_numpy(dtype=np.float64)

    metadata = load_clay_metadata()
    print(f"[m03] loading Clay v1.5 ({CLAY_MODEL_SIZE}) from {CLAY_CHECKPOINT_PATH}")
    encoder_module = load_clay_encoder(device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sensors_to_run = ["s2", "s1"] if args.sensor == "both" else [args.sensor]
    years = [args.year] if args.year is not None else list(range(2017, 2025))
    base_year = min(years)
    per_year = expected_window_count_per_year()

    for sensor_key in sensors_to_run:
        sensor_dir = SENTINEL2_DIR if sensor_key == "s2" else SENTINEL1_DIR
        prefix     = "s2" if sensor_key == "s2" else "s1"

        composites = list_composites(sensor_dir, prefix, args.year)
        print(f"[m03] {sensor_key}: {len(composites)} composites to embed")

        zarr_path = OUTPUT_DIR / f"{ 'sentinel2' if sensor_key=='s2' else 'sentinel1' }.zarr"
        root, embed_arr, mask_arr = build_or_open_zarr(zarr_path, n_tiles, per_year, years)

        for composite_path in composites:
            year, window_index_in_year, _ = parse_window_date_from_filename(composite_path)
            global_index = window_global_index(year, window_index_in_year, base_year, per_year)

            if bool(mask_arr[0, global_index]):
                continue   # already done

            # Skip corrupt / zero-byte chips from interrupted M2 runs — M2's
            # follow-up pass will refill them when we delete them.
            if composite_path.stat().st_size < 1024:
                print(f"[m03] SKIP  {composite_path.name}  (size={composite_path.stat().st_size} B — corrupt)",
                      flush=True)
                continue

            t_start = time.time()
            try:
                per_tile_embeddings = embed_one_composite(
                    composite_path, sensor_key, tiles_in_utm,
                    centroid_lats, centroid_lons, metadata,
                    encoder_module, device,
                )
            except Exception as exc:
                print(f"[m03] SKIP  {composite_path.name}  ({type(exc).__name__}: {exc})",
                      flush=True)
                continue
            embed_arr[:, global_index, :] = per_tile_embeddings
            mask_arr[:, global_index] = True
            elapsed = time.time() - t_start
            print(f"[m03] {composite_path.name}  -> [{n_tiles}, {CLAY_EMBED_DIM}]  ({elapsed:.1f}s)",
                  flush=True)

    print("[m03] done.")


if __name__ == "__main__":
    main()
