"""
Milestone 2 — pull Sentinel-2 + Sentinel-1 composites from GEE to the cluster.

Pipeline
--------
For every time-window in 2017-2024:

  1. Build a single AOI-wide Sentinel-2 SR composite (cloud-masked + median).
  2. Build a single AOI-wide Sentinel-1 GRD composite (speckle-filtered + mean).
  3. Queue an Earth Engine Export.image.toDrive task for each composite.
  4. As tasks finish on Google's side, rclone-move the GeoTIFF from Drive
     into <SCRATCH_ROOT>/m02_satellite_chips/.
     The "move" deletes the file from Drive after a successful copy, so
     personal-Drive storage never piles up past a few GB.

Why AOI-wide composites instead of per-tile?
-------------------------------------------
A per-tile export schedule would queue ~97,000 EE tasks. GEE's task
scheduler caps at a few thousand active tasks, and the tasks-per-second
limit makes the queueing itself slow. AOI-wide composites cut the job
count to ~304 (152 windows x 2 sensors). The per-tile splitting happens
locally on the cluster after download — see ``m02b_split_to_tiles.py``.

Resume-safe
-----------
Every export task is recorded in ``m02_state.csv`` with its task id and
status. If the script (or the SLURM job) is killed mid-run, re-launching
it picks up exactly where it left off: tasks already DONE are skipped,
tasks still RUNNING are re-polled, only un-queued windows are submitted.

Run
---
This script is meant to run inside the SLURM job ``m02_export.slurm``,
but can also be launched directly from the login node for the smoke
test (e.g. ``--test-year 2024``).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import ee
import geopandas as gpd
import google.auth


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Paths.
PERSIST_ROOT = Path(
    "<PERSIST_ROOT>"
)
SCRATCH_ROOT = Path("<SCRATCH_ROOT>")

AOI_TILES_GPKG_PATH = PERSIST_ROOT / "m01_aoi_fields" / "central_iowa_tiles.gpkg"
LOCAL_SENTINEL2_DIR = SCRATCH_ROOT / "m02_satellite_chips" / "sentinel2"
LOCAL_SENTINEL1_DIR = SCRATCH_ROOT / "m02_satellite_chips" / "sentinel1"
STATE_CSV_PATH      = SCRATCH_ROOT / "m02_satellite_chips" / "m02_state.csv"

# Drive folders Export.image.toDrive writes into. We use TWO flat folder
# names (no slash inside the names) because GEE replaces "/" with the
# full-width "／" (U+FF0F) when constructing the Drive folder, which makes
# rclone unable to navigate the path. One folder per sensor.
DRIVE_SENTINEL2_FOLDER = "clay_veg_stress_s2"
DRIVE_SENTINEL1_FOLDER = "clay_veg_stress_s1"

# GEE project (same as for the AlphaEarth scripts).
GEE_CLOUD_PROJECT = "bamboo-creek-269221"

# Years to export.
YEARS_TO_EXPORT: List[int] = list(range(2017, 2025))

# Growing-season biweekly windows: April 1 to October 15.
GROWING_SEASON_START = (4, 1)     # (month, day)
GROWING_SEASON_END   = (10, 15)
GROWING_SEASON_WINDOW_DAYS = 14   # biweekly

# Off-season (winter) monthly windows: October 16 to March 31.
OFFSEASON_WINDOW_DAYS = 30

# Sentinel-2 bands we keep (10 m + 20 m resampled to 10 m).
S2_BANDS_TO_KEEP = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

# SCL classes that mean "bad pixel" — drop them from each scene before
# building the median composite.
S2_BAD_SCL_CLASSES = [0, 1, 3, 8, 9, 10]

# Drop scenes that have >60 % of their pixels marked as bad SCL.
S2_MAX_BAD_PIXEL_FRACTION_PER_SCENE = 0.60

# Sentinel-1: small focal-mean speckle filter applied AT GEE, before mean.
S1_SPECKLE_FOCAL_RADIUS_METRES = 50

# Output projection — UTM zone 15N is the right zone for central Iowa.
OUTPUT_CRS_EPSG = "EPSG:32615"
OUTPUT_PIXEL_SIZE_METRES = 10

# Batch behaviour for the streaming pipeline.
# Lowered from 12 to 6 after we exhausted Drive: even with permanent-delete
# in place, keeping the in-flight transit volume small means a single slow
# rclone sync doesn't push Drive over the limit.
MAX_ACTIVELY_QUEUED_EXPORT_TASKS = 6    # how many tasks to keep in flight
POLL_INTERVAL_SECONDS            = 60   # how often to re-check task status
DRIVE_FOLDER_REMOTE_NAME         = "drive"   # the rclone remote name

# rclone binary on the cluster.
RCLONE_EXECUTABLE = Path("<RCLONE_BINARY_PATH>")


# ---------------------------------------------------------------------------
# Earth Engine initialisation using gcloud ADC.
# ---------------------------------------------------------------------------

def initialize_earth_engine_with_adc() -> None:
    """Initialize EE with the gcloud Application Default Credentials.

    Used because this venv inherits an older `earthengine-api` from the
    parent venv that doesn't auto-fall-back to ADC when the EE-specific
    credentials file is absent.
    """
    credentials, _ = google.auth.default(scopes=[
        "https://www.googleapis.com/auth/earthengine",
        "https://www.googleapis.com/auth/cloud-platform",
    ])
    ee.Initialize(credentials=credentials, project=GEE_CLOUD_PROJECT)


# ---------------------------------------------------------------------------
# Time-window planning
# ---------------------------------------------------------------------------

@dataclass
class TimeWindow:
    """One export composite — a date range + a human-readable label."""

    label: str         # e.g. "2024_w14_jul02"
    start_date: dt.date
    end_date_exclusive: dt.date

    def as_ee_date_range(self) -> Tuple[ee.Date, ee.Date]:
        return ee.Date(self.start_date.isoformat()), ee.Date(self.end_date_exclusive.isoformat())


def generate_time_windows_for_year(year: int) -> List[TimeWindow]:
    """Generate the biweekly + monthly windows for one calendar year.

    Convention: a window is named by its first day, in `YYYY_wNN_DDDmm` format
    where NN is a 2-digit sequence number within the year.
    """
    windows: List[TimeWindow] = []
    window_index = 0

    # ---- growing-season biweekly windows ----
    growing_start = dt.date(year, *GROWING_SEASON_START)
    growing_end   = dt.date(year, *GROWING_SEASON_END)
    current_start = growing_start
    # Require at least a week of room before adding another window — avoids
    # 1-day stub windows at the tail (e.g. Oct 14 -> Oct 15).
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

    # ---- off-season monthly windows ----
    # Five windows covering the months wholly OUTSIDE growing season:
    # Jan, Feb, Mar, Nov, Dec. We deliberately skip Oct 16-31 because it
    # is just a tail of growing season and contributes little new signal.
    offseason_starts = [
        dt.date(year, 1, 1), dt.date(year, 2, 1), dt.date(year, 3, 1),
        dt.date(year, 11, 1), dt.date(year, 12, 1),
    ]
    for offseason_start in offseason_starts:
        offseason_end = offseason_start + dt.timedelta(days=OFFSEASON_WINDOW_DAYS)
        window_index += 1
        windows.append(TimeWindow(
            label=f"{year}_w{window_index:02d}_{offseason_start.strftime('%b%d').lower()}",
            start_date=offseason_start,
            end_date_exclusive=offseason_end,
        ))

    return windows


# ---------------------------------------------------------------------------
# Sentinel-2 composite construction
# ---------------------------------------------------------------------------

def build_sentinel2_composite(
    aoi_polygon: ee.Geometry,
    time_window: TimeWindow,
) -> Optional[ee.Image]:
    """Cloud-masked median Sentinel-2 SR composite over the AOI.

    Returns None if no usable scenes fall in the window.
    """
    start_date, end_date = time_window.as_ee_date_range()
    s2_scenes = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi_polygon)
        .filterDate(start_date, end_date)
    )

    def _mask_bad_scl(scene_image: ee.Image) -> ee.Image:
        scl_band = scene_image.select("SCL")
        # Build a "good pixel" boolean mask by chaining .neq for each bad class.
        good_mask = ee.Image.constant(1).rename("good")
        for bad_class in S2_BAD_SCL_CLASSES:
            good_mask = good_mask.And(scl_band.neq(bad_class))
        return scene_image.updateMask(good_mask)

    s2_scenes_masked = s2_scenes.map(_mask_bad_scl)

    # Take the per-pixel median across all valid pixels in the window and
    # keep the data as int16 DN. We DO NOT compute NDVI/NDWI here, do not
    # divide by 10000, and do not switch to float — all of that adds bytes
    # per pixel and pushes the GeoTIFF past GEE's per-file ~1.5 GB limit,
    # which would force GEE to auto-tile the export into 4+ pieces. Doing
    # the scaling and the derived indices on the cluster after download is
    # near-free and keeps each export to a single GeoTIFF file.
    #
    # Pixels with zero unmasked scenes get masked out automatically — those
    # will become 0 / nodata in the exported GeoTIFF and the cluster code
    # treats them as missing.
    nonempty_composite = (
        s2_scenes_masked.select(S2_BANDS_TO_KEEP).median().toInt16()
    )

    # Empty-collection guard: identical pattern to the Sentinel-1 helper.
    # Some winter windows can contain zero scenes (cloud + edge-of-season
    # gaps); without this guard ``.median()`` of an empty collection
    # crashes the export with "Image has no bands".
    nodata_placeholder = (
        ee.Image.constant([0] * len(S2_BANDS_TO_KEEP))
        .rename(S2_BANDS_TO_KEEP)
        .toInt16()
        .updateMask(0)
    )
    s2_median_composite = ee.Image(
        ee.Algorithms.If(
            s2_scenes.size().gt(0),
            nonempty_composite,
            nodata_placeholder,
        )
    ).clip(aoi_polygon)

    return s2_median_composite


# ---------------------------------------------------------------------------
# Sentinel-1 composite construction
# ---------------------------------------------------------------------------

def build_sentinel1_composite(
    aoi_polygon: ee.Geometry,
    time_window: TimeWindow,
) -> Optional[ee.Image]:
    """Speckle-filtered VV + VH mean composite over the AOI.

    Robustness fixes (applied after early smoke-test failures):

    1. Empty-collection guard. Some windows contain zero Sentinel-1 scenes
       because of S1 orbit-cycle gaps (especially after S1B failed in
       December 2021). When the collection is empty, ``.mean()`` produces
       a zero-band image and the subsequent ``.focal_mean()`` crashes with
       ``"Can't get band number 0"``. We use ``ee.Algorithms.If`` to fall
       back to a fully-masked 2-band placeholder, which exports as nodata
       everywhere — the cluster-side loader treats that as "missing".

    2. Cast to float32 (``.toFloat()``). The native Sentinel-1 GRD bands
       come back as float32 from Google, but ``.focal_mean`` silently
       up-casts to float64. That doubles the file size (~2.1 GB instead
       of ~1 GB). Casting back to float32 at the end halves Drive churn.
    """
    start_date, end_date = time_window.as_ee_date_range()
    s1_scenes = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi_polygon)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
    )

    # Placeholder image for windows with no scenes — all bands masked off,
    # exports as nodata everywhere, has the right band names so downstream
    # code doesn't need a special case.
    nodata_placeholder = (
        ee.Image.constant([0, 0]).rename(["VV", "VH"]).toFloat().updateMask(0)
    )

    # Speckle filter then mean — order matters; mean THEN speckle would
    # double-smooth.
    nonempty_composite = (
        s1_scenes
        .mean()
        .focal_mean(S1_SPECKLE_FOCAL_RADIUS_METRES, "circle", "meters")
    )

    s1_mean_composite = ee.Image(
        ee.Algorithms.If(
            s1_scenes.size().gt(0),
            nonempty_composite,
            nodata_placeholder,
        )
    ).toFloat().clip(aoi_polygon)

    return s1_mean_composite


# ---------------------------------------------------------------------------
# Export task book-keeping
# ---------------------------------------------------------------------------

@dataclass
class ExportRecord:
    """One row in m02_state.csv."""

    sensor: str          # 's2' | 's1'
    year: int
    window_label: str
    drive_filename: str  # without folder prefix
    task_id: str         # GEE task id, may be empty until queued
    status: str          # 'PENDING' | 'RUNNING' | 'DONE' | 'FAILED' | 'SYNCED'
    local_path: str      # populated once rclone-moved


# CSV column order — used by every read/write of m02_state.csv.
STATE_CSV_FIELDS = [
    "sensor", "year", "window_label", "drive_filename",
    "task_id", "status", "local_path",
]


def read_state_csv(state_csv_path: Path) -> List[ExportRecord]:
    """Load the resume-state CSV. Returns an empty list if the file is new."""
    if not state_csv_path.exists():
        return []
    records = []
    with open(state_csv_path, "r", newline="") as handle:
        for raw_row in csv.DictReader(handle):
            records.append(ExportRecord(
                sensor=raw_row["sensor"],
                year=int(raw_row["year"]),
                window_label=raw_row["window_label"],
                drive_filename=raw_row["drive_filename"],
                task_id=raw_row["task_id"],
                status=raw_row["status"],
                local_path=raw_row["local_path"],
            ))
    return records


def write_state_csv(state_csv_path: Path, records: List[ExportRecord]) -> None:
    state_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATE_CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({k: getattr(record, k) for k in STATE_CSV_FIELDS})


# ---------------------------------------------------------------------------
# Building the planned-export table
# ---------------------------------------------------------------------------

def build_planned_export_records(
    years_to_export: Iterable[int],
    only_year: Optional[int] = None,
) -> List[ExportRecord]:
    """Enumerate every (sensor, year, window) combination we want to export."""
    planned_records: List[ExportRecord] = []
    for year in years_to_export:
        if only_year is not None and year != only_year:
            continue
        for time_window in generate_time_windows_for_year(year):
            for sensor in ("s2", "s1"):
                drive_filename = f"{sensor}_{time_window.label}"
                planned_records.append(ExportRecord(
                    sensor=sensor,
                    year=year,
                    window_label=time_window.label,
                    drive_filename=drive_filename,
                    task_id="",
                    status="PENDING",
                    local_path="",
                ))
    return planned_records


# ---------------------------------------------------------------------------
# Queue + monitor + sync loop
# ---------------------------------------------------------------------------

def queue_export_task(
    aoi_polygon: ee.Geometry,
    aoi_export_region: ee.Geometry,
    record: ExportRecord,
) -> str:
    """Build the composite, fire off Export.image.toDrive, return task id."""
    time_window = TimeWindow(
        label=record.window_label,
        start_date=_parse_window_label_to_start_date(record.window_label),
        end_date_exclusive=_parse_window_label_to_end_date(record.window_label, record.sensor),
    )

    if record.sensor == "s2":
        composite_image = build_sentinel2_composite(aoi_polygon, time_window)
        drive_folder    = DRIVE_SENTINEL2_FOLDER
    elif record.sensor == "s1":
        composite_image = build_sentinel1_composite(aoi_polygon, time_window)
        drive_folder    = DRIVE_SENTINEL1_FOLDER
    else:
        raise ValueError(f"Unknown sensor {record.sensor!r}")

    task = ee.batch.Export.image.toDrive(
        image=composite_image,
        description=record.drive_filename,
        folder=drive_folder,
        fileNamePrefix=record.drive_filename,
        region=aoi_export_region,
        scale=OUTPUT_PIXEL_SIZE_METRES,
        crs=OUTPUT_CRS_EPSG,
        maxPixels=int(1e10),
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    return task.id


def _parse_window_label_to_start_date(window_label: str) -> dt.date:
    """`2024_w14_jul02` -> date(2024, 7, 2)."""
    year_str, _window_idx_str, monthday_str = window_label.split("_")
    return dt.datetime.strptime(f"{year_str}_{monthday_str}", "%Y_%b%d").date()


def _parse_window_label_to_end_date(window_label: str, sensor: str) -> dt.date:
    """End is start + GROWING/OFF_SEASON_WINDOW_DAYS depending on month."""
    start_date = _parse_window_label_to_start_date(window_label)
    growing_start = dt.date(start_date.year, *GROWING_SEASON_START)
    growing_end   = dt.date(start_date.year, *GROWING_SEASON_END)
    if growing_start <= start_date < growing_end:
        return start_date + dt.timedelta(days=GROWING_SEASON_WINDOW_DAYS)
    return start_date + dt.timedelta(days=OFFSEASON_WINDOW_DAYS)


def poll_active_task_statuses(records: List[ExportRecord]) -> Dict[str, str]:
    """Look up the current EE-side status of every RUNNING record's task."""
    task_id_to_status: Dict[str, str] = {}
    for record in records:
        if record.status != "RUNNING":
            continue
        try:
            task_info = ee.data.getTaskStatus(record.task_id)[0]
            task_id_to_status[record.task_id] = task_info.get("state", "UNKNOWN")
        except Exception:  # network blip / transient; will retry next loop
            task_id_to_status[record.task_id] = "UNKNOWN"
    return task_id_to_status


def _list_drive_folder_ids_by_name(folder_name: str) -> List[str]:
    """Return every Drive folder ID that has the given name at the root.

    Drive allows duplicate folder names at the same level; GEE creates a
    fresh ``clay_veg_stress_s*`` folder for nearly every Export task. The
    cluster-side rclone command can only navigate to ONE folder per name,
    so we enumerate IDs directly and probe each.
    """
    list_command = [
        str(RCLONE_EXECUTABLE), "lsjson", f"{DRIVE_FOLDER_REMOTE_NAME}:",
        "--dirs-only",
    ]
    list_result = subprocess.run(list_command, capture_output=True, text=True)
    if list_result.returncode != 0:
        return []
    try:
        import json
        all_folders = json.loads(list_result.stdout)
    except Exception:
        return []
    return [folder["ID"] for folder in all_folders if folder["Name"] == folder_name]


def rclone_move_file_from_drive(drive_filename: str, sensor: str,
                                local_destination_dir: Path) -> Optional[Path]:
    """rclone-copy a finished GeoTIFF from Drive to the cluster, then delete.

    Handles the "Drive has multiple folders with identical names" problem:
    GEE auto-creates a brand-new ``clay_veg_stress_s*`` folder for nearly
    every Export task, so there is usually a handful of duplicate folders
    by the time we're trying to download. We enumerate every folder ID
    matching the expected name and probe each in turn (using
    ``--drive-root-folder-id``) until the file is found, copied, and
    deleted.

    GEE may auto-split a large export into ``<file>-NNNNNNNN-NNNNNNNN.tif``
    siblings if the in-memory size exceeds its per-file cap. The
    ``<file>*.tif`` glob covers both single-file and multi-file exports.
    """
    drive_folder = (
        DRIVE_SENTINEL2_FOLDER if sensor == "s2" else DRIVE_SENTINEL1_FOLDER
    )
    local_destination_dir.mkdir(parents=True, exist_ok=True)

    candidate_folder_ids = _list_drive_folder_ids_by_name(drive_folder)
    if not candidate_folder_ids:
        print(
            f"  no Drive folders named {drive_folder!r} found for {drive_filename}",
            flush=True,
        )
        return None

    for candidate_id in candidate_folder_ids:
        copy_command = [
            str(RCLONE_EXECUTABLE), "copy", f"{DRIVE_FOLDER_REMOTE_NAME}:",
            str(local_destination_dir),
            "--drive-root-folder-id", candidate_id,
            "--include", f"{drive_filename}*.tif",
            "--ignore-existing",
        ]
        subprocess.run(copy_command, capture_output=True, text=True)
        matching_local_files = sorted(
            local_destination_dir.glob(f"{drive_filename}*.tif")
        )
        if not matching_local_files:
            continue   # file not in this folder; try the next ID

        # File landed — PERMANENTLY delete the Drive copy. The default
        # rclone-delete-on-Drive routes files to Trash, where they continue
        # to occupy quota until trash is auto-purged 30 days later. With a
        # streaming pipeline that produces ~150 GB of GeoTIFFs over a few
        # hours, Drive trash fills up long before auto-purge runs. The
        # `--drive-use-trash=false` flag makes the delete bypass trash and
        # free the bytes immediately.
        delete_command = [
            str(RCLONE_EXECUTABLE), "delete", f"{DRIVE_FOLDER_REMOTE_NAME}:",
            "--drive-root-folder-id", candidate_id,
            "--include", f"{drive_filename}*.tif",
            "--drive-use-trash=false",
        ]
        subprocess.run(delete_command, capture_output=True, text=True)
        return matching_local_files[0]

    print(
        f"  rclone copy FAILED for {drive_filename}: "
        f"file not found in any of {len(candidate_folder_ids)} candidate "
        f"folder IDs named {drive_folder!r}",
        flush=True,
    )
    return None


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_streaming_export_loop(
    aoi_polygon_wgs84: gpd.GeoDataFrame,
    planned_records: List[ExportRecord],
) -> List[ExportRecord]:
    """Queue + poll + sync until every planned record is SYNCED or FAILED."""
    aoi_geometry = aoi_polygon_wgs84.geometry.iloc[0]
    # Use the ACTUAL AOI polygon (not its bbox) as both the compute region
    # and the export region. The 6-county AOI is L-shaped; its bbox covers
    # ~15,000 km² but the AOI itself is only ~9,000 km². Exporting the bbox
    # wasted ~40% of every file as nodata and pushed S2 GeoTIFFs to ~2.2 GB
    # (close to GEE's auto-split threshold). The polygon export brings each
    # file to ~1.3 GB and avoids splits.
    aoi_compute_polygon = ee.Geometry(aoi_geometry.__geo_interface__)
    aoi_export_polygon  = aoi_compute_polygon

    print(f"[m02] planned records: {len(planned_records)}")

    while True:
        # 1. Refresh statuses of any still-running tasks.
        statuses = poll_active_task_statuses(planned_records)
        for record in planned_records:
            if record.status == "RUNNING" and record.task_id in statuses:
                ee_state = statuses[record.task_id]
                if ee_state == "COMPLETED":
                    record.status = "DONE"
                elif ee_state in ("FAILED", "CANCELLED"):
                    record.status = "FAILED"

        # 2. Move every DONE record's GeoTIFF from Drive to scratch.
        for record in planned_records:
            if record.status != "DONE":
                continue
            local_destination_dir = (
                LOCAL_SENTINEL2_DIR if record.sensor == "s2" else LOCAL_SENTINEL1_DIR
            )
            moved_path = rclone_move_file_from_drive(
                record.drive_filename, record.sensor, local_destination_dir
            )
            if moved_path is not None:
                record.local_path = str(moved_path)
                record.status = "SYNCED"
                print(f"  synced: {record.drive_filename} -> {moved_path}", flush=True)
            else:
                print(f"  sync failed (will retry next pass): {record.drive_filename}",
                      flush=True)

        # 3. Queue more tasks up to the max-in-flight cap.
        in_flight = sum(1 for r in planned_records if r.status == "RUNNING")
        for record in planned_records:
            if in_flight >= MAX_ACTIVELY_QUEUED_EXPORT_TASKS:
                break
            if record.status != "PENDING":
                continue
            try:
                task_id = queue_export_task(
                    aoi_compute_polygon, aoi_export_polygon, record
                )
                record.task_id = task_id
                record.status = "RUNNING"
                in_flight += 1
                print(f"  queued: {record.drive_filename} -> task {task_id}", flush=True)
            except Exception as e:
                print(f"  queueing FAILED for {record.drive_filename}: {e}", flush=True)
                record.status = "FAILED"

        # 4. Persist state so we can resume.
        write_state_csv(STATE_CSV_PATH, planned_records)

        # 5. Stop when nothing more to do.
        remaining_states = {r.status for r in planned_records
                            if r.status not in ("SYNCED", "FAILED")}
        if not remaining_states:
            print("[m02] all records terminal (SYNCED or FAILED) — done.", flush=True)
            break

        # 6. Wait before next pass.
        print(
            f"[m02] in-flight={in_flight}  "
            f"synced={sum(1 for r in planned_records if r.status == 'SYNCED')}  "
            f"pending={sum(1 for r in planned_records if r.status == 'PENDING')}  "
            f"failed={sum(1 for r in planned_records if r.status == 'FAILED')}",
            flush=True,
        )
        time.sleep(POLL_INTERVAL_SECONDS)

    return planned_records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-year", type=int, default=None,
        help="Only export this year (smoke-test mode).",
    )
    parser.add_argument(
        "--force-restart", action="store_true",
        help="Delete m02_state.csv and start fresh instead of resuming.",
    )
    args = parser.parse_args()

    initialize_earth_engine_with_adc()
    print(f"[m02] EE initialised with project {GEE_CLOUD_PROJECT}")

    aoi_polygon_wgs84 = gpd.read_file(AOI_TILES_GPKG_PATH).dissolve()
    print(f"[m02] AOI loaded from {AOI_TILES_GPKG_PATH}")

    # Resume / fresh-start logic.
    if args.force_restart and STATE_CSV_PATH.exists():
        STATE_CSV_PATH.unlink()

    # Always build the FULL plan first, then merge in any existing state.
    # Earlier versions short-circuited and used only the existing state if
    # the CSV existed — that meant we couldn't expand from a 2024 smoke
    # test to a full 2017-2024 run without first deleting the CSV and
    # losing the work already synced.
    full_plan = build_planned_export_records(
        YEARS_TO_EXPORT, only_year=args.test_year
    )
    existing_state = read_state_csv(STATE_CSV_PATH)
    existing_by_key = {
        (r.sensor, r.year, r.window_label): r for r in existing_state
    }

    records: List[ExportRecord] = []
    reused_count = 0
    reset_failed_count = 0
    for planned in full_plan:
        key = (planned.sensor, planned.year, planned.window_label)
        existing = existing_by_key.get(key)
        if existing is None:
            records.append(planned)
            continue
        reused_count += 1
        # FAILED records were FAILED under an older script version; rerun
        # them under the current code path (which has the empty-collection
        # guard) so they don't perma-fail.
        if existing.status == "FAILED":
            existing.status = "PENDING"
            existing.task_id = ""
            reset_failed_count += 1
        records.append(existing)
    print(
        f"[m02] planned {len(full_plan)} exports; "
        f"reused {reused_count} from CSV "
        f"(reset {reset_failed_count} previously-FAILED back to PENDING)"
    )

    if not records:
        print("[m02] nothing to do.")
        return 0

    final_records = run_streaming_export_loop(aoi_polygon_wgs84, records)

    failed = [r for r in final_records if r.status == "FAILED"]
    synced = [r for r in final_records if r.status == "SYNCED"]
    print(f"[m02] final: {len(synced)} synced, {len(failed)} failed.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
