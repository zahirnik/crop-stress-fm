"""
Milestone 9 — three-way comparison: NDVI baseline (M6) vs Clay distance
(M8) vs Clay + GRU (M7).

Loads the predictions from each milestone, aligns them on (tile_id,
year, window_index) for 2024 valid rows, then produces:

    comparison_metrics.json     consolidated AUC + best-F1 numbers
    roc_curves.png              ROC curves overlaid
    pr_curves.png               Precision-Recall curves overlaid
    confusion_matrices.png      one panel per method at their best F1 threshold
    per_window_recall.png       recall by window for each method
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


SCRATCH_ROOT = Path("<SCRATCH_ROOT>")

LABELS_PATH      = SCRATCH_ROOT / "m04_proxy_labels"     / "labels.parquet"
NDVI_PREDS_PATH  = SCRATCH_ROOT / "m06_baseline"         / "baseline_predictions.parquet"
DIST_PREDS_PATH  = SCRATCH_ROOT / "m08_path_b_distance"  / "path_b_predictions.parquet"
GRU_PREDS_PATH   = SCRATCH_ROOT / "m07_path_a_gru"       / "path_a_predictions.parquet"

OUTPUT_DIR = SCRATCH_ROOT / "m09_comparison"
METRICS_PATH               = OUTPUT_DIR / "comparison_metrics.json"
ROC_PLOT_PATH              = OUTPUT_DIR / "roc_curves.png"
PR_PLOT_PATH               = OUTPUT_DIR / "pr_curves.png"
CONFUSION_PLOT_PATH        = OUTPUT_DIR / "confusion_matrices.png"
PER_WINDOW_RECALL_PATH     = OUTPUT_DIR / "per_window_recall.png"

SCORE_YEAR = 2024


def best_f1_threshold(label_array: np.ndarray, score_array: np.ndarray) -> Tuple[float, Dict[str, float]]:
    precision_arr, recall_arr, thresholds = precision_recall_curve(label_array, score_array)
    f1_arr = 2.0 * precision_arr * recall_arr / np.maximum(precision_arr + recall_arr, 1e-12)
    best_index = int(np.nanargmax(f1_arr))
    best_threshold = float(thresholds[min(best_index, len(thresholds) - 1)]) if len(thresholds) else float("nan")
    return best_threshold, {
        "precision": float(precision_arr[best_index]),
        "recall":    float(recall_arr[best_index]),
        "f1":        float(f1_arr[best_index]),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[m09] loading predictions")
    labels_df = pd.read_parquet(LABELS_PATH)[
        ["tile_id", "year", "window_index", "label_z15"]
    ]
    ndvi_df = pd.read_parquet(NDVI_PREDS_PATH)[
        ["tile_id", "year", "window_index", "score_neg_ndvi"]
    ].rename(columns={"score_neg_ndvi": "ndvi_score"})
    dist_df = pd.read_parquet(DIST_PREDS_PATH)[
        ["tile_id", "year", "window_index", "s2_dist"]
    ].rename(columns={"s2_dist": "clay_distance_score"})
    gru_df = pd.read_parquet(GRU_PREDS_PATH)[
        ["tile_id", "year", "window_index", "p_stressed"]
    ].rename(columns={"p_stressed": "clay_gru_score"})

    merged = (
        labels_df
        .merge(ndvi_df, on=["tile_id", "year", "window_index"], how="left")
        .merge(dist_df, on=["tile_id", "year", "window_index"], how="left")
        .merge(gru_df,  on=["tile_id", "year", "window_index"], how="left")
    )
    score_mask = (
        (merged["year"] == SCORE_YEAR)
        & (merged["label_z15"].isin([0, 1]))
        & merged["ndvi_score"].notna()
        & merged["clay_distance_score"].notna()
        & merged["clay_gru_score"].notna()
    )
    score_df = merged.loc[score_mask].copy()
    print(f"[m09] {len(score_df)} rows usable in {SCORE_YEAR} (intersected across methods)")

    label_array = score_df["label_z15"].to_numpy().astype(np.int8)
    methods = {
        "NDVI baseline (M6)":         score_df["ndvi_score"].to_numpy(),
        "Clay distance (M8)":         score_df["clay_distance_score"].to_numpy(),
        "Clay + GRU (M7)":            score_df["clay_gru_score"].to_numpy(),
    }

    consolidated: Dict[str, Dict[str, float]] = {}
    plt.figure(figsize=(7, 6))
    for method_name, score_array in methods.items():
        roc_auc = roc_auc_score(label_array, score_array)
        fpr, tpr, _ = roc_curve(label_array, score_array)
        plt.plot(fpr, tpr, label=f"{method_name}  AUC={roc_auc:.3f}")
        consolidated[method_name] = {"roc_auc": float(roc_auc)}
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4, label="chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curves on {SCORE_YEAR} (n={len(label_array)}, positives={int(label_array.sum())})")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROC_PLOT_PATH, dpi=120)
    plt.close()
    print(f"[m09] wrote {ROC_PLOT_PATH}")

    plt.figure(figsize=(7, 6))
    chance_rate = float(label_array.mean())
    for method_name, score_array in methods.items():
        precision_arr, recall_arr, _ = precision_recall_curve(label_array, score_array)
        pr_auc = average_precision_score(label_array, score_array)
        plt.plot(recall_arr, precision_arr, label=f"{method_name}  PR-AUC={pr_auc:.3f}")
        consolidated[method_name]["pr_auc"] = float(pr_auc)
    plt.axhline(chance_rate, color="k", linestyle="--", linewidth=0.8, alpha=0.4, label=f"chance (prevalence={chance_rate:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall on {SCORE_YEAR}")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PR_PLOT_PATH, dpi=120)
    plt.close()
    print(f"[m09] wrote {PR_PLOT_PATH}")

    best_thresholds: Dict[str, float] = {}
    fig, axes = plt.subplots(1, len(methods), figsize=(14, 4), sharey=True)
    for axis, (method_name, score_array) in zip(axes, methods.items()):
        threshold, metrics_at_best = best_f1_threshold(label_array, score_array)
        predictions = (score_array >= threshold).astype(np.int8)
        confusion = confusion_matrix(label_array, predictions, labels=[0, 1])
        consolidated[method_name].update({
            "best_threshold": threshold,
            **metrics_at_best,
        })
        best_thresholds[method_name] = threshold

        axis.imshow(confusion, cmap="Blues")
        axis.set_xticks([0, 1])
        axis.set_yticks([0, 1])
        axis.set_xticklabels(["pred 0", "pred 1"])
        axis.set_yticklabels(["true 0", "true 1"])
        axis.set_title(f"{method_name}\nF1={metrics_at_best['f1']:.3f}  prec={metrics_at_best['precision']:.3f}  rec={metrics_at_best['recall']:.3f}", fontsize=9)
        for i in range(2):
            for j in range(2):
                axis.text(j, i, str(confusion[i, j]), ha="center", va="center",
                          color="white" if confusion[i, j] > confusion.max() / 2 else "black",
                          fontsize=11)
    plt.tight_layout()
    plt.savefig(CONFUSION_PLOT_PATH, dpi=120)
    plt.close()
    print(f"[m09] wrote {CONFUSION_PLOT_PATH}")

    plt.figure(figsize=(10, 5))
    window_index_array = score_df["window_index"].to_numpy()
    label_per_window = (
        score_df.assign(label=label_array)
        .groupby("window_index")["label"].sum()
    )
    for method_name, score_array in methods.items():
        threshold = best_thresholds[method_name]
        per_window_recall = []
        windows_sorted = sorted(score_df["window_index"].unique().tolist())
        for window_value in windows_sorted:
            window_mask = window_index_array == window_value
            if not window_mask.any():
                per_window_recall.append(np.nan)
                continue
            window_labels = label_array[window_mask]
            if window_labels.sum() == 0:
                per_window_recall.append(np.nan)
                continue
            window_predictions = (score_array[window_mask] >= threshold).astype(np.int8)
            true_positive = int(((window_labels == 1) & (window_predictions == 1)).sum())
            per_window_recall.append(true_positive / float(window_labels.sum()))
        plt.plot(windows_sorted, per_window_recall, marker="o", label=method_name)
    plt.xticks(sorted(score_df["window_index"].unique().tolist()))
    plt.xlabel("Window index (1=Apr1, 19=Dec1)")
    plt.ylabel("Recall (TP / actual positives in window)")
    plt.title(f"Per-window recall on {SCORE_YEAR} at each method's best-F1 threshold")
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PER_WINDOW_RECALL_PATH, dpi=120)
    plt.close()
    print(f"[m09] wrote {PER_WINDOW_RECALL_PATH}")

    consolidated["meta"] = {
        "n_total":     int(len(score_df)),
        "n_positives": int(label_array.sum()),
        "prevalence":  float(label_array.mean()),
    }
    METRICS_PATH.write_text(json.dumps(consolidated, indent=2))
    print(f"[m09] wrote {METRICS_PATH}")
    print(json.dumps(consolidated, indent=2))


if __name__ == "__main__":
    main()
