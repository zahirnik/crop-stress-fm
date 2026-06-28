"""
Milestone 8 — Path B: Clay embedding-distance anomaly detector.

For each (tile, calendar-window-index) build a historical mean Clay
embedding from 2017-2023. Score every 2024 (tile, window) by the cosine
distance between its embedding and the matching historical mean.

Three score variants:
    s2_dist             — Sentinel-2 cosine distance
    s1_dist             — Sentinel-1 cosine distance
    combined_dist       — concatenated [S2 | S1] cosine distance

Each is evaluated against the M4 label_z15 with the same metrics used
in M6 (NDVI baseline): ROC AUC, PR AUC, best-F1 threshold + the
precision / recall / F1 at it.

Outputs:
    m08_path_b_distance/
        path_b_metrics.json
        path_b_predictions.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import zarr
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


SCRATCH_ROOT = Path("<SCRATCH_ROOT>")
EMBEDDINGS_DIR = SCRATCH_ROOT / "m03_embeddings"
S2_ZARR_PATH = EMBEDDINGS_DIR / "sentinel2.zarr"
S1_ZARR_PATH = EMBEDDINGS_DIR / "sentinel1.zarr"

LABELS_PATH = SCRATCH_ROOT / "m04_proxy_labels" / "labels.parquet"

OUTPUT_DIR = SCRATCH_ROOT / "m08_path_b_distance"
METRICS_PATH     = OUTPUT_DIR / "path_b_metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "path_b_predictions.parquet"

YEARS_IN_CUBE = list(range(2017, 2025))
WINDOWS_PER_YEAR = 19
SCORE_YEAR = 2024
HISTORY_YEARS = list(range(2017, 2024))


def load_zarr_cube(zarr_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (embedding_array [tile, window, dim], mask_array [tile, window])."""
    root = zarr.open_group(str(zarr_path), mode="r")
    return root["embedding"][:], root["mask"][:]


def reshape_to_year_window(cube: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    cube shape: [n_tiles, n_year*n_window, dim]
    -> [n_tiles, n_years, n_windows_per_year, dim]
    """
    n_tiles, _, embed_dim = cube.shape
    n_years = len(YEARS_IN_CUBE)
    reshaped_cube = cube.reshape(n_tiles, n_years, WINDOWS_PER_YEAR, embed_dim)
    reshaped_mask = mask.reshape(n_tiles, n_years, WINDOWS_PER_YEAR)
    return reshaped_cube, reshaped_mask


def historical_mean_per_tile_window(reshaped_cube: np.ndarray, reshaped_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns per-(tile, window-of-year) mean embedding + a count of history
    years that contributed.
    """
    history_index = [YEARS_IN_CUBE.index(y) for y in HISTORY_YEARS]
    history_cube = reshaped_cube[:, history_index, :, :]
    history_mask = reshaped_mask[:, history_index, :]
    contributing_counts = history_mask.sum(axis=1).astype(np.int16)

    history_zero_filled = history_cube * history_mask[..., None]
    sum_over_years = history_zero_filled.sum(axis=1)
    safe_counts = np.maximum(contributing_counts, 1)
    mean_embedding = sum_over_years / safe_counts[..., None]
    return mean_embedding.astype(np.float32), contributing_counts


def cosine_distance_per_tile_window(
    query_cube: np.ndarray,
    reference_means: np.ndarray,
    query_mask: np.ndarray,
    reference_count_per_window: np.ndarray,
) -> np.ndarray:
    """
    query_cube:      [n_tiles, n_windows_per_year, dim]   — one year
    reference_means: [n_tiles, n_windows_per_year, dim]
    Returns [n_tiles, n_windows_per_year] cosine distance, NaN where
    either side is missing.
    """
    eps = 1e-8
    query_norm     = np.linalg.norm(query_cube, axis=-1, keepdims=True)
    reference_norm = np.linalg.norm(reference_means, axis=-1, keepdims=True)
    cosine_similarity = (query_cube * reference_means).sum(axis=-1) / (query_norm.squeeze(-1) * reference_norm.squeeze(-1) + eps)
    cosine_distance = 1.0 - cosine_similarity

    missing = (~query_mask) | (reference_count_per_window <= 0)
    cosine_distance = np.where(missing, np.nan, cosine_distance)
    return cosine_distance.astype(np.float32)


def score_distance_table(
    sensor_label: str,
    distances_for_score_year: np.ndarray,   # [n_tiles, n_windows_per_year]
) -> pd.DataFrame:
    """Returns long DataFrame: tile_id, year, window_index, sensor_dist column."""
    n_tiles, n_windows = distances_for_score_year.shape
    tile_id_column = np.repeat(np.arange(1, n_tiles + 1, dtype=np.int32), n_windows)
    window_index_column = np.tile(np.arange(1, n_windows + 1, dtype=np.int16), n_tiles)
    year_column = np.full(n_tiles * n_windows, SCORE_YEAR, dtype=np.int16)
    distance_column = distances_for_score_year.reshape(-1)
    return pd.DataFrame({
        "tile_id":      tile_id_column,
        "year":         year_column,
        "window_index": window_index_column,
        f"{sensor_label}_dist": distance_column,
    })


def threshold_best_f1(label_array: np.ndarray, score_array: np.ndarray) -> Dict[str, float]:
    precision_arr, recall_arr, thresholds = precision_recall_curve(label_array, score_array)
    f1_arr = 2.0 * precision_arr * recall_arr / np.maximum(precision_arr + recall_arr, 1e-12)
    best_index = int(np.nanargmax(f1_arr))
    best_threshold = float(thresholds[min(best_index, len(thresholds) - 1)]) if len(thresholds) else float("nan")
    return {
        "best_threshold": best_threshold,
        "precision":      float(precision_arr[best_index]),
        "recall":         float(recall_arr[best_index]),
        "f1":             float(f1_arr[best_index]),
    }


def evaluate_one_score(label_array: np.ndarray, score_array: np.ndarray, name: str) -> Dict[str, float]:
    return {
        f"{name}_roc_auc": float(roc_auc_score(label_array, score_array)),
        f"{name}_pr_auc":  float(average_precision_score(label_array, score_array)),
        **{
            f"{name}_{k}": v for k, v in threshold_best_f1(label_array, score_array).items()
        },
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[m08] loading S2 cube: {S2_ZARR_PATH}")
    s2_cube, s2_mask = load_zarr_cube(S2_ZARR_PATH)
    s2_cube, s2_mask = reshape_to_year_window(s2_cube, s2_mask)
    print(f"[m08]   shape {s2_cube.shape}  mask_true {int(s2_mask.sum())} / {s2_mask.size}")

    print(f"[m08] loading S1 cube: {S1_ZARR_PATH}")
    s1_cube, s1_mask = load_zarr_cube(S1_ZARR_PATH)
    s1_cube, s1_mask = reshape_to_year_window(s1_cube, s1_mask)
    print(f"[m08]   shape {s1_cube.shape}  mask_true {int(s1_mask.sum())} / {s1_mask.size}")

    print(f"[m08] building historical mean ({HISTORY_YEARS[0]}-{HISTORY_YEARS[-1]})")
    s2_history_mean, s2_history_counts = historical_mean_per_tile_window(s2_cube, s2_mask)
    s1_history_mean, s1_history_counts = historical_mean_per_tile_window(s1_cube, s1_mask)

    score_year_index = YEARS_IN_CUBE.index(SCORE_YEAR)
    s2_query     = s2_cube[:, score_year_index, :, :]
    s2_query_msk = s2_mask[:, score_year_index, :]
    s1_query     = s1_cube[:, score_year_index, :, :]
    s1_query_msk = s1_mask[:, score_year_index, :]

    s2_dist = cosine_distance_per_tile_window(s2_query, s2_history_mean, s2_query_msk, s2_history_counts)
    s1_dist = cosine_distance_per_tile_window(s1_query, s1_history_mean, s1_query_msk, s1_history_counts)

    combined_query = np.concatenate([s2_query, s1_query], axis=-1)
    combined_mean  = np.concatenate([s2_history_mean, s1_history_mean], axis=-1)
    combined_mask  = s2_query_msk & s1_query_msk
    combined_count = np.minimum(s2_history_counts, s1_history_counts)
    combined_dist  = cosine_distance_per_tile_window(combined_query, combined_mean, combined_mask, combined_count)

    s2_table       = score_distance_table("s2",       s2_dist)
    s1_table       = score_distance_table("s1",       s1_dist)
    combined_table = score_distance_table("combined", combined_dist)

    distances_df = s2_table.merge(s1_table, on=["tile_id", "year", "window_index"], how="outer")
    distances_df = distances_df.merge(combined_table, on=["tile_id", "year", "window_index"], how="outer")

    labels_df = pd.read_parquet(LABELS_PATH)
    merged = labels_df.merge(distances_df, on=["tile_id", "year", "window_index"], how="left")
    score_mask = (merged["year"] == SCORE_YEAR) & (merged["label_z15"].isin([0, 1]))
    score_df = merged.loc[score_mask].copy()
    print(f"[m08] {len(score_df)} rows usable in {SCORE_YEAR}")

    label_array = score_df["label_z15"].to_numpy().astype(np.int8)
    n_pos = int(label_array.sum())
    metrics: Dict[str, float] = {
        "n_total":     int(len(score_df)),
        "n_positives": n_pos,
        "prevalence":  float(n_pos / max(len(score_df), 1)),
    }

    for column in ("s2_dist", "s1_dist", "combined_dist"):
        valid_mask = score_df[column].notna()
        if int(valid_mask.sum()) == 0:
            continue
        sub_metrics = evaluate_one_score(
            label_array[valid_mask.values],
            score_df.loc[valid_mask, column].to_numpy(),
            name=column.removesuffix("_dist"),
        )
        metrics.update(sub_metrics)
        print(f"[m08] {column} metrics:")
        for k, v in sub_metrics.items():
            print(f"        {k}: {v}")

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"[m08] wrote {METRICS_PATH}")
    score_df.to_parquet(PREDICTIONS_PATH, index=False)
    print(f"[m08] wrote {PREDICTIONS_PATH}  ({len(score_df)} rows)")


if __name__ == "__main__":
    main()
