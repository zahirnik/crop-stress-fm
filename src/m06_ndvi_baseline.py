"""
Milestone 6 — NDVI / EVI threshold baseline (the number to beat).

Predict "stressed" purely from the raw vegetation index value, no
foundation model, no temporal head, no history. This is the naive
strawman every later method (M7 Clay + GRU, M8 Clay-distance anomaly)
must out-perform.

Score function: -ndvi (so high score = stressed). Same for -evi.

Evaluate on 2024 only (the score year), restricted to rows where the
M4 label is defined (label_z15 in {0, 1}).

Outputs:
    <SCRATCH_ROOT>/m06_baseline/
        baseline_metrics.json          summary metrics
        baseline_predictions.parquet   per-row scores + binary preds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


SCRATCH_ROOT = Path("<SCRATCH_ROOT>")
LABELS_PATH  = SCRATCH_ROOT / "m04_proxy_labels" / "labels.parquet"
OUTPUT_DIR   = SCRATCH_ROOT / "m06_baseline"
METRICS_PATH     = OUTPUT_DIR / "baseline_metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "baseline_predictions.parquet"

SCORE_YEAR = 2024


def threshold_best_f1(label_array: np.ndarray, score_array: np.ndarray) -> Dict[str, float]:
    """Sweep thresholds, return precision/recall/F1 at the one that maximizes F1."""
    precision_arr, recall_arr, thresholds = precision_recall_curve(label_array, score_array)
    f1_arr = 2.0 * precision_arr * recall_arr / np.maximum(precision_arr + recall_arr, 1e-12)
    best_index = int(np.nanargmax(f1_arr))
    # precision_recall_curve returns thresholds shorter by 1 than precision/recall arrays.
    best_threshold = float(thresholds[min(best_index, len(thresholds) - 1)]) if len(thresholds) else float("nan")
    return {
        "best_threshold": best_threshold,
        "precision": float(precision_arr[best_index]),
        "recall":    float(recall_arr[best_index]),
        "f1":        float(f1_arr[best_index]),
    }


def evaluate_one_index(label_array: np.ndarray, score_array: np.ndarray, name: str) -> Dict[str, float]:
    """Compute ROC AUC, PR AUC, best-F1 metrics for a single index."""
    roc_auc = float(roc_auc_score(label_array, score_array))
    pr_auc  = float(average_precision_score(label_array, score_array))
    best = threshold_best_f1(label_array, score_array)
    fixed_threshold_metrics = {
        f"{name}_threshold_-1.5_z_equiv_precision": float(precision_score(
            label_array, score_array >= best["best_threshold"], zero_division=0)),
    }
    return {
        f"{name}_roc_auc": roc_auc,
        f"{name}_pr_auc":  pr_auc,
        f"{name}_best_threshold": best["best_threshold"],
        f"{name}_precision_at_best_f1": best["precision"],
        f"{name}_recall_at_best_f1":    best["recall"],
        f"{name}_f1_at_best_f1":        best["f1"],
        **fixed_threshold_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="(debug) only first N rows")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[m06] loading labels: {LABELS_PATH}")
    labels_df = pd.read_parquet(LABELS_PATH)
    print(f"[m06] {len(labels_df)} rows total")

    scoring_mask = (labels_df["year"] == SCORE_YEAR) & (labels_df["label_z15"].isin([0, 1]))
    scoring_df = labels_df.loc[scoring_mask].copy()
    if args.limit is not None:
        scoring_df = scoring_df.head(args.limit)
    print(f"[m06] {len(scoring_df)} rows usable in {SCORE_YEAR}")

    if scoring_df.empty:
        raise SystemExit("[m06] no usable rows in scoring year — abort")

    label_array = scoring_df["label_z15"].to_numpy().astype(np.int8)
    n_pos = int(label_array.sum())
    n_neg = int(len(label_array) - n_pos)
    print(f"[m06] class balance: positives={n_pos}  negatives={n_neg}  prevalence={n_pos / len(label_array):.4f}")

    metrics: Dict[str, float] = {
        "n_total":      int(len(scoring_df)),
        "n_positives":  n_pos,
        "n_negatives":  n_neg,
        "prevalence":   float(n_pos / len(scoring_df)),
        "score_year":   SCORE_YEAR,
    }

    ndvi_valid_mask = scoring_df["ndvi"].notna()
    evi_valid_mask  = scoring_df["evi"].notna()
    if ndvi_valid_mask.sum() > 0:
        ndvi_metrics = evaluate_one_index(
            label_array=label_array[ndvi_valid_mask.values],
            score_array=(-scoring_df.loc[ndvi_valid_mask, "ndvi"]).to_numpy(),
            name="ndvi",
        )
        metrics.update(ndvi_metrics)
        print("[m06] NDVI baseline:")
        for k, v in ndvi_metrics.items():
            print(f"        {k}: {v}")
    if evi_valid_mask.sum() > 0:
        evi_metrics = evaluate_one_index(
            label_array=label_array[evi_valid_mask.values],
            score_array=(-scoring_df.loc[evi_valid_mask, "evi"]).to_numpy(),
            name="evi",
        )
        metrics.update(evi_metrics)
        print("[m06] EVI baseline:")
        for k, v in evi_metrics.items():
            print(f"        {k}: {v}")

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"[m06] wrote {METRICS_PATH}")

    scoring_df["score_neg_ndvi"] = -scoring_df["ndvi"]
    scoring_df["score_neg_evi"]  = -scoring_df["evi"]
    if "ndvi_best_threshold" in metrics:
        scoring_df["pred_ndvi"] = (scoring_df["score_neg_ndvi"] >= metrics["ndvi_best_threshold"]).astype(np.int8)
    if "evi_best_threshold" in metrics:
        scoring_df["pred_evi"]  = (scoring_df["score_neg_evi"]  >= metrics["evi_best_threshold"]).astype(np.int8)

    scoring_df.to_parquet(PREDICTIONS_PATH, index=False)
    print(f"[m06] wrote {PREDICTIONS_PATH}  ({len(scoring_df)} rows)")


if __name__ == "__main__":
    main()
