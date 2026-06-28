"""
Milestone 1 — build the AOI and per-tile metadata table (grid-tile version).

Pipeline
--------
1. Download the US Census TIGER 2024 counties shapefile, extract polygons
   for the six central-Iowa counties listed in TARGET_COUNTY_FIPS_FULL.
2. Union the six county polygons into a single AOI polygon.
3. For each year in YEARS, fetch that year's USDA Cropland Data Layer
   clipped to the AOI via the CropScape REST API:
       https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile
4. Generate a regular grid of TILE_SIZE_METRES × TILE_SIZE_METRES tiles
   covering the AOI's bounding box in CDL's native Albers CRS (EPSG:5070).
5. Drop tiles that don't intersect the AOI polygon.
6. For each surviving tile, count CDL pixels by class per year and emit:
      tile_id, geometry, centroid (lon, lat),
      corn_frac_<year>, soy_frac_<year>, crop_or_soy_frac_<year>,
      mean_corn_or_soy_frac (averaged across years),
      modal_crop_<year>  (the dominant of {corn, soy} per year, or 0)
7. Drop tiles whose mean corn-or-soy fraction across years is below
   MIN_MEAN_CROP_FRACTION (default 30 %).
8. Save the surviving tile table as a GeoPackage on <PERSIST_ROOT>
   (persistent), with geometry in WGS84 lon / lat so M2's Sentinel chip
   extraction lines up directly.

Why uniform tiles instead of CDL connected-components?
------------------------------------------------------
The 30 m CDL pixel can't resolve Midwest field roads (~10 m wide), so
connected components on "corn or soy in any year" produces giant blobs
covering ~90 % of the county. Even aggressive morphological opening
can't break the blobs because adjacent fields share pixel borders
directly. Uniform 5.12 km tiles sidestep the problem entirely: each
tile holds a mix of fields, towns, and roads, and the Clay encoder
later sees that mix as one chip. The downstream stress signal is
"per-tile" rather than "per-field", which matches the granularity at
which Clay actually operates.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from xml.etree import ElementTree

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
import rasterio.mask
import requests
from rasterio.transform import Affine
from shapely.geometry import box, shape as shapely_shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CONFIG — every knob the script exposes.
# ---------------------------------------------------------------------------

# Years to download from the USDA Cropland Data Layer.
YEARS: List[int] = list(range(2017, 2025))

# Six central-Iowa counties (FIPS = state + county).
TARGET_COUNTY_FIPS_FULL: List[str] = [
    "19169",  # Story
    "19079",  # Hamilton
    "19015",  # Boone
    "19127",  # Marshall
    "19083",  # Hardin
    "19153",  # Polk
]
AOI_HUMAN_NAME = "Central-Iowa 6-county corn belt"

# USDA CDL crop codes.
CDL_CODE_CORN = 1
CDL_CODE_SOYBEAN = 5

# Tile size in metres. 512 px × 10 m Sentinel-2 = 5,120 m.
TILE_SIZE_METRES = 5120

# A tile must have AT LEAST this fraction of corn-or-soy averaged across
# the 8 years to be considered an agricultural tile. Tiles below this
# threshold are dropped (likely contain mostly urban / forest / water).
MIN_MEAN_CROP_FRACTION = 0.30

# Output paths.
SCRATCH_ROOT = Path("<SCRATCH_ROOT>")
PERSIST_ROOT = Path(
    "<PERSIST_ROOT>"
)

COUNTIES_DOWNLOAD_DIR = SCRATCH_ROOT / "m01_aoi_fields" / "counties_tiger"
CDL_DOWNLOAD_DIR      = SCRATCH_ROOT / "m01_aoi_fields" / "cdl_raw"

TILES_GPKG_PATH = PERSIST_ROOT / "m01_aoi_fields" / "central_iowa_tiles.gpkg"


# ---------------------------------------------------------------------------
# 1. County boundaries from TIGER 2024
# ---------------------------------------------------------------------------

TIGER_COUNTIES_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/tl_2024_us_county.zip"
)


def download_and_extract_zip(url: str, destination_directory: Path) -> None:
    """Download a zip and unpack it into the destination directory."""
    destination_directory.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=120, stream=True)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(destination_directory)


def load_aoi_polygon(county_fips_codes: Sequence[str]) -> gpd.GeoDataFrame:
    """Return a 1-row GeoDataFrame whose geometry is the union of the
    requested counties."""
    if not (COUNTIES_DOWNLOAD_DIR / "tl_2024_us_county.shp").exists():
        print(f"[counties] downloading TIGER counties to {COUNTIES_DOWNLOAD_DIR}")
        download_and_extract_zip(TIGER_COUNTIES_URL, COUNTIES_DOWNLOAD_DIR)
    counties_geodataframe = gpd.read_file(
        COUNTIES_DOWNLOAD_DIR / "tl_2024_us_county.shp"
    )
    matched_counties = counties_geodataframe[
        counties_geodataframe["GEOID"].isin(county_fips_codes)
    ]
    missing_codes = set(county_fips_codes) - set(matched_counties["GEOID"])
    if missing_codes:
        raise RuntimeError(f"FIPS codes not found: {missing_codes}")
    aoi_union_geometry: BaseGeometry = unary_union(matched_counties.geometry.values)
    return gpd.GeoDataFrame(
        {"name": [AOI_HUMAN_NAME], "geometry": [aoi_union_geometry]},
        crs=matched_counties.crs,
    ).to_crs("EPSG:4326")


# ---------------------------------------------------------------------------
# 2. Per-year CDL download via CropScape API
# ---------------------------------------------------------------------------

CROPSCAPE_API_URL = (
    "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile"
)


def fetch_cdl_geotiff_for_county(year: int, county_fips_full: str) -> Path:
    """Download the per-year per-county CDL GeoTIFF."""
    output_geotiff = CDL_DOWNLOAD_DIR / f"cdl_{year}_county_{county_fips_full}.tif"
    if output_geotiff.exists():
        return output_geotiff
    CDL_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Ask CropScape to prepare the per-county clip and return its URL.
    response = requests.get(
        CROPSCAPE_API_URL,
        params={"year": str(year), "fips": county_fips_full},
        timeout=120,
    )
    response.raise_for_status()
    xml_root = ElementTree.fromstring(response.text)
    return_url_node = next(
        (e for e in xml_root.iter() if e.tag.endswith("returnURL")), None
    )
    if return_url_node is None or not return_url_node.text:
        raise RuntimeError(
            f"CropScape returned no URL for year={year} fips={county_fips_full}"
        )
    geotiff_url = return_url_node.text.strip()

    # Stream the TIF to disk.
    tif_response = requests.get(geotiff_url, timeout=300, stream=True)
    tif_response.raise_for_status()
    with open(output_geotiff, "wb") as handle:
        for chunk in tif_response.iter_content(chunk_size=1024 * 1024):
            handle.write(chunk)
    return output_geotiff


def read_cdl_clipped_to_polygon(
    geotiff_path: Path,
    polygon_geodataframe_in_cdl_crs: gpd.GeoDataFrame,
) -> Tuple[np.ndarray, Affine]:
    """Open one CDL GeoTIFF and crop it to a polygon (in the CDL's CRS)."""
    with rasterio.open(geotiff_path) as dataset:
        cropped_array, cropped_transform = rasterio.mask.mask(
            dataset,
            polygon_geodataframe_in_cdl_crs.geometry,
            crop=True,
            filled=True,
            nodata=0,
        )
    return cropped_array[0], cropped_transform


def stitch_yearly_cdl_for_aoi(
    year: int,
    county_fips_codes: Sequence[str],
    aoi_polygon_in_cdl_crs: gpd.GeoDataFrame,
) -> Tuple[np.ndarray, Affine]:
    """Per-county fetch + mosaic into one AOI-wide CDL array for the year.

    CropScape only serves CDL per single county at a time, so we fetch each
    county, then mosaic them onto a common grid aligned to CDL's 30 m
    Albers raster. Pixels outside the AOI polygon get 0.
    """
    if len(county_fips_codes) == 0:
        raise ValueError("Need at least one county")

    # Fetch one TIF per county.
    per_county_arrays: List[np.ndarray] = []
    per_county_transforms: List[Affine] = []
    common_crs = None
    for fips_code in county_fips_codes:
        single_county_tif = fetch_cdl_geotiff_for_county(year, fips_code)
        with rasterio.open(single_county_tif) as dataset:
            if common_crs is None:
                common_crs = dataset.crs
            per_county_arrays.append(dataset.read(1))
            per_county_transforms.append(dataset.transform)

    # Compute the union bounding box across all per-county arrays, then
    # write each one into a single AOI-sized empty array. We use the same
    # 30 m grid origin as the first county's transform.
    pixel_size = abs(per_county_transforms[0].a)
    if not all(abs(t.a) == pixel_size for t in per_county_transforms):
        raise RuntimeError("CDL counties have mismatched pixel sizes")

    bounds_each = []
    for arr, t in zip(per_county_arrays, per_county_transforms):
        h, w = arr.shape
        x0, y_top = t.c, t.f
        x1 = x0 + w * t.a
        y_bottom = y_top + h * t.e
        bounds_each.append((x0, y_bottom, x1, y_top))
    union_x0 = min(b[0] for b in bounds_each)
    union_y0 = min(b[1] for b in bounds_each)
    union_x1 = max(b[2] for b in bounds_each)
    union_y1 = max(b[3] for b in bounds_each)

    # Align the union grid to the first county's origin so pixel centres
    # match the source TIF exactly (avoids half-pixel resampling artefacts).
    first_x0, first_y1 = per_county_transforms[0].c, per_county_transforms[0].f
    col_offset = int(np.round((union_x0 - first_x0) / pixel_size))
    row_offset = int(np.round((first_y1 - union_y1) / pixel_size))
    union_x0_aligned = first_x0 + col_offset * pixel_size
    union_y1_aligned = first_y1 - row_offset * pixel_size

    union_width  = int(np.ceil((union_x1 - union_x0_aligned) / pixel_size))
    union_height = int(np.ceil((union_y1_aligned - union_y0) / pixel_size))

    aoi_cdl_array = np.zeros((union_height, union_width), dtype=np.uint8)
    aoi_transform = Affine(pixel_size, 0, union_x0_aligned,
                           0, -pixel_size, union_y1_aligned)

    # Paste each county's pixels at the correct offset in the union array.
    for arr, t in zip(per_county_arrays, per_county_transforms):
        h, w = arr.shape
        col_off = int(np.round((t.c - union_x0_aligned) / pixel_size))
        row_off = int(np.round((union_y1_aligned - t.f) / pixel_size))
        # Pixels can overlap on county borders — keep the larger label
        # (typically corn/soy beats 0/background).
        target_slice = aoi_cdl_array[row_off:row_off + h, col_off:col_off + w]
        np.copyto(target_slice, np.maximum(target_slice, arr))

    # Mask everything outside the AOI polygon to 0.
    with rasterio.io.MemoryFile() as memory_file:
        with memory_file.open(
            driver="GTiff", height=union_height, width=union_width,
            count=1, dtype="uint8", crs=common_crs, transform=aoi_transform,
        ) as in_memory_dataset:
            in_memory_dataset.write(aoi_cdl_array, 1)
            masked_array, masked_transform = rasterio.mask.mask(
                in_memory_dataset,
                aoi_polygon_in_cdl_crs.geometry,
                crop=True, filled=True, nodata=0,
            )

    return masked_array[0], masked_transform


# ---------------------------------------------------------------------------
# 3. Generate the regular tile grid
# ---------------------------------------------------------------------------

def generate_tile_grid_within_aoi(
    aoi_polygon_in_projected_crs: gpd.GeoDataFrame,
    tile_size_metres: int,
    projected_crs: str | dict,
) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame of square tiles that intersect the AOI."""
    aoi_geometry = aoi_polygon_in_projected_crs.geometry.iloc[0]
    minimum_x, minimum_y, maximum_x, maximum_y = aoi_geometry.bounds

    # Snap the grid origin to a multiple of tile_size_metres so different
    # AOIs share consistent global tile centres.
    snapped_origin_x = (int(minimum_x) // tile_size_metres) * tile_size_metres
    snapped_origin_y = (int(minimum_y) // tile_size_metres) * tile_size_metres

    tile_records: List[dict] = []
    tile_id_counter = 1
    current_y = snapped_origin_y
    while current_y < maximum_y:
        current_x = snapped_origin_x
        while current_x < maximum_x:
            tile_polygon = box(current_x, current_y,
                               current_x + tile_size_metres,
                               current_y + tile_size_metres)
            if tile_polygon.intersects(aoi_geometry):
                tile_records.append({
                    "tile_id": tile_id_counter,
                    "geometry": tile_polygon,
                })
                tile_id_counter += 1
            current_x += tile_size_metres
        current_y += tile_size_metres

    return gpd.GeoDataFrame(tile_records, crs=projected_crs)


# ---------------------------------------------------------------------------
# 4. Per-tile CDL statistics
# ---------------------------------------------------------------------------

def compute_per_tile_crop_fractions(
    tiles_geodataframe_in_cdl_crs: gpd.GeoDataFrame,
    yearly_cdl_arrays: Dict[int, np.ndarray],
    cdl_affine_transform: Affine,
) -> gpd.GeoDataFrame:
    """For each tile, compute per-year corn / soy fractions.

    We do this by rasterising each tile into the CDL pixel grid and counting
    classes inside the burned-in raster. This is O(n_tiles × n_pixels_per_tile)
    which for ~400 tiles × ~290k pixels each is fine on the login node.
    """
    cdl_height, cdl_width = next(iter(yearly_cdl_arrays.values())).shape

    corn_fractions_per_year: Dict[int, List[float]] = {y: [] for y in yearly_cdl_arrays}
    soy_fractions_per_year:  Dict[int, List[float]] = {y: [] for y in yearly_cdl_arrays}
    modal_crops_per_year:    Dict[int, List[int]]   = {y: [] for y in yearly_cdl_arrays}

    for _, tile_row in tqdm(
        tiles_geodataframe_in_cdl_crs.iterrows(),
        total=len(tiles_geodataframe_in_cdl_crs),
        desc="per-tile CDL stats",
    ):
        # Rasterise this tile polygon into a mask aligned to the CDL grid.
        tile_mask = rasterio.features.geometry_mask(
            [tile_row.geometry],
            out_shape=(cdl_height, cdl_width),
            transform=cdl_affine_transform,
            invert=True,   # True INSIDE the polygon
            all_touched=False,
        )
        tile_pixel_total = int(tile_mask.sum())
        if tile_pixel_total == 0:
            # The tile fell into all-nodata CDL — shouldn't happen since we
            # filtered tiles by AOI intersection, but be safe.
            for year in yearly_cdl_arrays:
                corn_fractions_per_year[year].append(0.0)
                soy_fractions_per_year[year].append(0.0)
                modal_crops_per_year[year].append(0)
            continue

        for year, cdl_array in yearly_cdl_arrays.items():
            tile_cdl_values = cdl_array[tile_mask]
            corn_count = int((tile_cdl_values == CDL_CODE_CORN).sum())
            soy_count  = int((tile_cdl_values == CDL_CODE_SOYBEAN).sum())
            corn_fractions_per_year[year].append(corn_count / tile_pixel_total)
            soy_fractions_per_year[year].append(soy_count / tile_pixel_total)

            # Modal crop for this year inside this tile: corn / soy / 0.
            if corn_count > soy_count and corn_count > 0:
                modal_crops_per_year[year].append(CDL_CODE_CORN)
            elif soy_count > 0:
                modal_crops_per_year[year].append(CDL_CODE_SOYBEAN)
            else:
                modal_crops_per_year[year].append(0)

    # Attach the new columns onto the GeoDataFrame.
    for year in yearly_cdl_arrays:
        tiles_geodataframe_in_cdl_crs[f"corn_frac_{year}"] = corn_fractions_per_year[year]
        tiles_geodataframe_in_cdl_crs[f"soy_frac_{year}"]  = soy_fractions_per_year[year]
        tiles_geodataframe_in_cdl_crs[f"crop_frac_{year}"] = (
            np.asarray(corn_fractions_per_year[year])
            + np.asarray(soy_fractions_per_year[year])
        )
        tiles_geodataframe_in_cdl_crs[f"modal_crop_{year}"] = modal_crops_per_year[year]

    # Mean fraction across years is what we filter on.
    mean_crop_fraction = np.mean(
        np.column_stack(
            [tiles_geodataframe_in_cdl_crs[f"crop_frac_{y}"].values for y in yearly_cdl_arrays]
        ),
        axis=1,
    )
    tiles_geodataframe_in_cdl_crs["mean_corn_or_soy_frac"] = mean_crop_fraction

    return tiles_geodataframe_in_cdl_crs


# ---------------------------------------------------------------------------
# 5. Centroid lat/lon and final cleanup
# ---------------------------------------------------------------------------

def add_centroid_columns_in_wgs84(
    tiles_geodataframe_in_projected_crs: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Add centroid_lon / centroid_lat columns in EPSG:4326."""
    projected_centroids = tiles_geodataframe_in_projected_crs.geometry.centroid
    wgs84_centroids = (
        gpd.GeoSeries(projected_centroids, crs=tiles_geodataframe_in_projected_crs.crs)
        .to_crs("EPSG:4326")
    )
    tiles_geodataframe_in_projected_crs["centroid_lon"] = wgs84_centroids.x.values
    tiles_geodataframe_in_projected_crs["centroid_lat"] = wgs84_centroids.y.values
    return tiles_geodataframe_in_projected_crs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"[m01] AOI:           {AOI_HUMAN_NAME}")
    print(f"[m01] counties:      {TARGET_COUNTY_FIPS_FULL}")
    print(f"[m01] years:         {YEARS[0]}..{YEARS[-1]}")
    print(f"[m01] tile size:     {TILE_SIZE_METRES} m  ({TILE_SIZE_METRES // 10} px @ 10 m)")
    print(f"[m01] output GPKG:   {TILES_GPKG_PATH}")

    # 1. AOI polygon in WGS84.
    aoi_polygon_wgs84 = load_aoi_polygon(TARGET_COUNTY_FIPS_FULL)
    aoi_area_km2 = float(aoi_polygon_wgs84.to_crs(5070).area.iloc[0]) / 1e6
    print(f"[m01] AOI area:      {aoi_area_km2:.1f} km^2")

    # 2. Per-year CDL stitched across the 6 counties.
    yearly_cdl_arrays: Dict[int, np.ndarray] = {}
    cdl_affine_transform: Affine | None = None
    cdl_crs = None
    for year in tqdm(YEARS, desc="CDL"):
        # Need the AOI polygon in CDL's CRS. We learn CDL's CRS on the
        # first fetch.
        if cdl_crs is None:
            first_tif = fetch_cdl_geotiff_for_county(year, TARGET_COUNTY_FIPS_FULL[0])
            with rasterio.open(first_tif) as ds:
                cdl_crs = ds.crs
        aoi_polygon_in_cdl_crs = aoi_polygon_wgs84.to_crs(cdl_crs)
        cdl_array, cdl_transform = stitch_yearly_cdl_for_aoi(
            year, TARGET_COUNTY_FIPS_FULL, aoi_polygon_in_cdl_crs
        )
        yearly_cdl_arrays[year] = cdl_array
        if cdl_affine_transform is None:
            cdl_affine_transform = cdl_transform

    # Normalize all years to a common shape — rasterio.mask.mask(..., crop=True)
    # can compute slightly different bounding boxes year-to-year when the
    # underlying CDL tiles have drifted by one pixel. We crop every array
    # to the minimum height/width and keep the first year's transform
    # (which corresponds to the matching origin pixel).
    common_height = min(arr.shape[0] for arr in yearly_cdl_arrays.values())
    common_width  = min(arr.shape[1] for arr in yearly_cdl_arrays.values())
    for year_key in YEARS:
        yearly_cdl_arrays[year_key] = yearly_cdl_arrays[year_key][
            :common_height, :common_width
        ]
    height, width = common_height, common_width
    print(f"[m01] CDL mosaic:    ({height}, {width}) pixels  CRS: {cdl_crs}")

    # 3. Tile grid in CDL's projected CRS.
    aoi_polygon_in_cdl_crs = aoi_polygon_wgs84.to_crs(cdl_crs)
    tiles_in_cdl_crs = generate_tile_grid_within_aoi(
        aoi_polygon_in_cdl_crs,
        tile_size_metres=TILE_SIZE_METRES,
        projected_crs=cdl_crs,
    )
    print(f"[m01] raw tiles:     {len(tiles_in_cdl_crs)} (before crop-fraction filter)")

    # 4. CDL stats per tile.
    tiles_in_cdl_crs = compute_per_tile_crop_fractions(
        tiles_in_cdl_crs, yearly_cdl_arrays, cdl_affine_transform
    )

    # 5. Filter by mean corn-or-soy fraction.
    before_filter_count = len(tiles_in_cdl_crs)
    tiles_in_cdl_crs = tiles_in_cdl_crs[
        tiles_in_cdl_crs["mean_corn_or_soy_frac"] >= MIN_MEAN_CROP_FRACTION
    ].copy().reset_index(drop=True)
    print(
        f"[m01] kept tiles:    {len(tiles_in_cdl_crs)} / {before_filter_count}  "
        f"(mean_corn_or_soy_frac >= {MIN_MEAN_CROP_FRACTION:.2f})"
    )
    tiles_in_cdl_crs["tile_id"] = np.arange(1, len(tiles_in_cdl_crs) + 1)

    # 6. Centroid lon/lat + reproject geometry to WGS84.
    tiles_in_cdl_crs = add_centroid_columns_in_wgs84(tiles_in_cdl_crs)
    tiles_in_wgs84 = tiles_in_cdl_crs.to_crs("EPSG:4326")

    column_order = [
        "tile_id", "mean_corn_or_soy_frac",
        "centroid_lon", "centroid_lat",
        *[f"corn_frac_{y}" for y in YEARS],
        *[f"soy_frac_{y}" for y in YEARS],
        *[f"crop_frac_{y}" for y in YEARS],
        *[f"modal_crop_{y}" for y in YEARS],
        "geometry",
    ]
    tiles_in_wgs84 = tiles_in_wgs84[column_order]

    # 7. Save GeoPackage.
    TILES_GPKG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tiles_in_wgs84.to_file(TILES_GPKG_PATH, driver="GPKG")
    print(f"[m01] wrote {TILES_GPKG_PATH}")

    # Quick summary of dominant crop in the last year.
    last_year = YEARS[-1]
    summary_lines = []
    summary_lines.append(f"  total tiles:                     {len(tiles_in_wgs84):>6d}")
    summary_lines.append(
        f"  mean of mean_corn_or_soy_frac:   "
        f"{tiles_in_wgs84['mean_corn_or_soy_frac'].mean():>6.3f}"
    )
    summary_lines.append(
        f"  median mean_corn_or_soy_frac:    "
        f"{tiles_in_wgs84['mean_corn_or_soy_frac'].median():>6.3f}"
    )
    last_year_modal_counts = tiles_in_wgs84[f"modal_crop_{last_year}"].value_counts().to_dict()
    summary_lines.append(
        f"  {last_year} modal crop counts (1=corn, 5=soy, 0=other): "
        f"{last_year_modal_counts}"
    )
    print("[m01] summary:")
    for line in summary_lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
