"""
Milestone 7 — Path A: small GRU on Clay v1.5 S2 embeddings.

Treat each tile's 19-step S2 embedding sequence for one year as the
input; predict a per-timestep "stressed" probability against the M4
NDVI z-score label. Train on 2017-2023, validate on 2024.

Model:
    nn.GRU(input_dim=1024, hidden_dim=128, num_layers=1, batch_first=True)
    -> Linear(128, 1) -> BCEWithLogits per timestep (masked for missing labels)

Outputs:
    m07_path_a_gru/
        path_a_metrics.json
        path_a_predictions.parquet
        path_a_model.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
S2_ZARR_PATH = SCRATCH_ROOT / "m03_embeddings" / "sentinel2.zarr"
LABELS_PATH  = SCRATCH_ROOT / "m04_proxy_labels" / "labels.parquet"

OUTPUT_DIR = SCRATCH_ROOT / "m07_path_a_gru"
METRICS_PATH     = OUTPUT_DIR / "path_a_metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "path_a_predictions.parquet"
MODEL_PATH       = OUTPUT_DIR / "path_a_model.pt"

YEARS_IN_CUBE = list(range(2017, 2025))
WINDOWS_PER_YEAR = 19
HISTORY_YEARS = list(range(2017, 2024))
SCORE_YEAR = 2024

HIDDEN_DIM = 128
NUM_EPOCHS = 80
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 64
RANDOM_SEED = 1


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = HIDDEN_DIM, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_output, _ = self.gru(x)         # [B, T, H]
        logits = self.head(gru_output).squeeze(-1)   # [B, T]
        return logits


def load_zarr_cube(zarr_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    root = zarr.open_group(str(zarr_path), mode="r")
    return root["embedding"][:], root["mask"][:]


def build_year_sequences(
    s2_cube: np.ndarray,
    s2_mask: np.ndarray,
    labels_df: pd.DataFrame,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Returns {year: {tile_features [n_tiles, 19, 1024], tile_labels [n_tiles, 19], valid_label_mask [n_tiles, 19]}}.
    valid_label_mask = True where label_z15 in {0, 1} (missing labels get -1).
    """
    n_tiles, _, embed_dim = s2_cube.shape
    sequences: Dict[int, Dict[str, np.ndarray]] = {}
    for year_offset, year in enumerate(YEARS_IN_CUBE):
        year_features = s2_cube.reshape(n_tiles, len(YEARS_IN_CUBE), WINDOWS_PER_YEAR, embed_dim)[:, year_offset]
        year_mask = s2_mask.reshape(n_tiles, len(YEARS_IN_CUBE), WINDOWS_PER_YEAR)[:, year_offset]

        labels_for_year = labels_df[labels_df["year"] == year].pivot_table(
            index="tile_id", columns="window_index", values="label_z15", aggfunc="first"
        ).sort_index().reindex(np.arange(1, n_tiles + 1))
        label_array = labels_for_year.to_numpy(dtype=np.float32, na_value=-1.0)
        valid_label_mask = np.isin(label_array, [0.0, 1.0]) & year_mask
        label_array = np.where(valid_label_mask, label_array, 0.0)

        sequences[year] = {
            "features":   year_features.astype(np.float32),
            "labels":     label_array.astype(np.float32),
            "valid_mask": valid_label_mask.astype(bool),
        }
    return sequences


def stack_train_arrays(sequences: Dict[int, Dict[str, np.ndarray]], train_years: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_list, label_list, mask_list = [], [], []
    for year in train_years:
        feature_list.append(sequences[year]["features"])
        label_list.append(sequences[year]["labels"])
        mask_list.append(sequences[year]["valid_mask"])
    return (
        np.concatenate(feature_list, axis=0),
        np.concatenate(label_list, axis=0),
        np.concatenate(mask_list, axis=0),
    )


def threshold_best_f1(label_array: np.ndarray, score_array: np.ndarray) -> Dict[str, float]:
    precision_arr, recall_arr, thresholds = precision_recall_curve(label_array, score_array)
    f1_arr = 2.0 * precision_arr * recall_arr / np.maximum(precision_arr + recall_arr, 1e-12)
    best_index = int(np.nanargmax(f1_arr))
    best_threshold = float(thresholds[min(best_index, len(thresholds) - 1)]) if len(thresholds) else float("nan")
    return {
        "best_threshold": best_threshold,
        "precision": float(precision_arr[best_index]),
        "recall":    float(recall_arr[best_index]),
        "f1":        float(f1_arr[best_index]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"[m07] device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[m07] loading S2 cube: {S2_ZARR_PATH}")
    s2_cube, s2_mask = load_zarr_cube(S2_ZARR_PATH)
    print(f"[m07] {s2_cube.shape}  mask_true {int(s2_mask.sum())}/{s2_mask.size}")

    print(f"[m07] loading labels: {LABELS_PATH}")
    labels_df = pd.read_parquet(LABELS_PATH)

    sequences = build_year_sequences(s2_cube, s2_mask, labels_df)

    train_features, train_labels, train_valid_mask = stack_train_arrays(sequences, HISTORY_YEARS)
    val_features   = sequences[SCORE_YEAR]["features"]
    val_labels     = sequences[SCORE_YEAR]["labels"]
    val_valid_mask = sequences[SCORE_YEAR]["valid_mask"]
    print(f"[m07] train sequences: {train_features.shape}   val sequences: {val_features.shape}")

    positive_count = int(train_labels[train_valid_mask].sum())
    negative_count = int(train_valid_mask.sum() - positive_count)
    pos_weight_value = max(negative_count, 1) / max(positive_count, 1)
    print(f"[m07] train class balance: pos={positive_count}  neg={negative_count}  pos_weight={pos_weight_value:.2f}")

    train_features_tensor   = torch.from_numpy(train_features).to(device)
    train_labels_tensor     = torch.from_numpy(train_labels).to(device)
    train_valid_mask_tensor = torch.from_numpy(train_valid_mask).to(device)
    val_features_tensor     = torch.from_numpy(val_features).to(device)
    val_labels_tensor       = torch.from_numpy(val_labels).to(device)
    val_valid_mask_tensor   = torch.from_numpy(val_valid_mask).to(device)

    model = GRUClassifier(input_dim=s2_cube.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(pos_weight_value, device=device))

    n_train = train_features_tensor.shape[0]
    print(f"[m07] training for {args.epochs} epochs over {n_train} sequences, batch={BATCH_SIZE}")

    best_val_auc = -1.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(n_train, device=device)
        running_loss = 0.0
        for start in range(0, n_train, BATCH_SIZE):
            stop = min(start + BATCH_SIZE, n_train)
            batch_indices = permutation[start:stop]
            batch_features = train_features_tensor[batch_indices]
            batch_labels   = train_labels_tensor[batch_indices]
            batch_mask     = train_valid_mask_tensor[batch_indices]
            optimizer.zero_grad()
            logits = model(batch_features)
            per_position_loss = loss_fn(logits, batch_labels)
            masked_loss_sum = (per_position_loss * batch_mask.float()).sum()
            masked_count    = batch_mask.float().sum().clamp_min(1.0)
            loss = masked_loss_sum / masked_count
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * batch_features.shape[0]
        epoch_train_loss = running_loss / max(n_train, 1)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_features_tensor)
            val_probabilities = torch.sigmoid(val_logits).cpu().numpy()
        val_labels_np = val_labels_tensor.cpu().numpy()
        val_mask_np   = val_valid_mask_tensor.cpu().numpy()
        if val_mask_np.any():
            try:
                val_auc = float(roc_auc_score(val_labels_np[val_mask_np].astype(np.int8), val_probabilities[val_mask_np]))
            except ValueError:
                val_auc = float("nan")
        else:
            val_auc = float("nan")

        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"[m07] epoch {epoch:3d}  train_loss {epoch_train_loss:.4f}  val_auc {val_auc:.4f}  best_auc {best_val_auc:.4f}", flush=True)

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)
    torch.save(best_state, MODEL_PATH)
    print(f"[m07] saved best model to {MODEL_PATH}")

    model.eval()
    with torch.no_grad():
        val_logits = model(val_features_tensor)
        val_probabilities = torch.sigmoid(val_logits).cpu().numpy()
    val_labels_np = val_labels_tensor.cpu().numpy()
    val_mask_np   = val_valid_mask_tensor.cpu().numpy()

    if not val_mask_np.any():
        raise SystemExit("[m07] no valid val rows — abort")

    label_array = val_labels_np[val_mask_np].astype(np.int8)
    score_array = val_probabilities[val_mask_np]
    metrics = {
        "n_total":     int(len(label_array)),
        "n_positives": int(label_array.sum()),
        "roc_auc":     float(roc_auc_score(label_array, score_array)),
        "pr_auc":      float(average_precision_score(label_array, score_array)),
        **{f"best_{k}": v for k, v in threshold_best_f1(label_array, score_array).items()},
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"[m07] wrote {METRICS_PATH}")
    for k, v in metrics.items():
        print(f"        {k}: {v}")

    n_tiles, _, _ = s2_cube.shape
    tile_id_column = np.repeat(np.arange(1, n_tiles + 1, dtype=np.int32), WINDOWS_PER_YEAR)
    window_index_column = np.tile(np.arange(1, WINDOWS_PER_YEAR + 1, dtype=np.int16), n_tiles)
    year_column = np.full(n_tiles * WINDOWS_PER_YEAR, SCORE_YEAR, dtype=np.int16)
    predictions_df = pd.DataFrame({
        "tile_id":      tile_id_column,
        "year":         year_column,
        "window_index": window_index_column,
        "label_z15":    val_labels_np.reshape(-1).astype(np.int8),
        "valid_mask":   val_mask_np.reshape(-1).astype(bool),
        "p_stressed":   val_probabilities.reshape(-1).astype(np.float32),
    })
    predictions_df.to_parquet(PREDICTIONS_PATH, index=False)
    print(f"[m07] wrote {PREDICTIONS_PATH}  ({len(predictions_df)} rows)")


if __name__ == "__main__":
    main()
