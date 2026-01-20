# src/eval/recon_auc_ap.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import nn


# -----------------------
# Config
# -----------------------
@dataclass(frozen=True)
class ReconEvalConfig:
    batch_size: int = 1
    num_workers: int = 2

    # edge cleanup to match training behavior
    remove_self_loops: bool = True
    keep_upper_triangle: bool = True  # keep i<j

    # if True, skip graphs without recon_neg_edge_index
    skip_if_no_neg: bool = True


# -----------------------
# Utils
# -----------------------
def clean_edge_index(
    edge_index: torch.Tensor,
    *,
    remove_self_loops: bool,
    keep_upper_triangle: bool,
) -> torch.Tensor:
    ei = edge_index
    if remove_self_loops:
        m = ei[0] != ei[1]
        ei = ei[:, m]
    if keep_upper_triangle:
        m2 = ei[0] < ei[1]
        ei = ei[:, m2]
    return ei


def load_state_dict_any(weight_path: str | Path, device: torch.device) -> Dict[str, torch.Tensor]:
    obj = torch.load(str(weight_path), map_location=device)
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state" in obj and isinstance(obj["model_state"], dict):
            return obj["model_state"]
        if all(torch.is_tensor(v) for v in obj.values()):
            return obj
    raise ValueError(f"Unsupported checkpoint format: {weight_path}")


# -----------------------
# Core
# -----------------------
@torch.no_grad()
def evaluate_reconstruction_auc_ap(
    *,
    model: nn.Module,
    loader,
    device: torch.device,
    cfg: ReconEvalConfig,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluate AUC/AP for reconstruction:
      - positives: data.edge_index
      - negatives: data.recon_neg_edge_index (must exist)

    Assumptions:
      - model.encode(x, edge_index, batch_vec) -> (z_mod, g)
      - model.main_dec(z_mod, edge_index) -> logits (E,)
    """
    model.eval()

    rows: List[dict] = []
    for batch in loader:
        data = batch.to(device)

        # file/name bookkeeping if present
        file_name = getattr(data, "file_name", None)
        if isinstance(file_name, (list, tuple)) and file_name:
            file_name = file_name[0]
        file_name = str(file_name) if file_name is not None else "unknown"

        # edge cleaning (same as training)
        if hasattr(data, "edge_index") and data.edge_index is not None:
            data.edge_index = clean_edge_index(
                data.edge_index,
                remove_self_loops=cfg.remove_self_loops,
                keep_upper_triangle=cfg.keep_upper_triangle,
            )

        if not hasattr(data, "recon_neg_edge_index"):
            if cfg.skip_if_no_neg:
                if verbose:
                    print(f"[skip] no recon_neg_edge_index: {file_name}")
                continue
            raise AttributeError(f"{file_name} has no recon_neg_edge_index")

        neg_edge = data.recon_neg_edge_index
        if neg_edge is None or neg_edge.numel() == 0:
            if cfg.skip_if_no_neg:
                if verbose:
                    print(f"[skip] empty recon_neg_edge_index: {file_name}")
                continue
            raise ValueError(f"{file_name} recon_neg_edge_index is empty")

        # batch vec
        batch_vec = getattr(data, "batch", None)
        if batch_vec is None:
            batch_vec = torch.zeros(int(data.num_nodes), dtype=torch.long, device=device)

        # forward
        z_mod, _ = model.encode(data.x, data.edge_index, batch_vec)

        pos_logits = model.main_dec(z_mod, data.edge_index)
        neg_logits = model.main_dec(z_mod, neg_edge.to(device))

        pos_prob = torch.sigmoid(pos_logits).detach().cpu().numpy()
        neg_prob = torch.sigmoid(neg_logits).detach().cpu().numpy()

        y_true = np.concatenate([np.ones_like(pos_prob), np.zeros_like(neg_prob)])
        y_pred = np.concatenate([pos_prob, neg_prob])

        auc = roc_auc_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)

        # optional attrs
        cell_id = getattr(data, "cell_id", None)
        if isinstance(cell_id, (list, tuple)) and cell_id:
            cell_id = cell_id[0]
        cell_id = str(cell_id) if cell_id is not None else ""

        chrom = getattr(data, "chrom", None)
        if isinstance(chrom, (list, tuple)) and chrom:
            chrom = chrom[0]
        chrom = str(chrom) if chrom is not None else ""

        rows.append(
            dict(
                file=file_name,
                cell_id=cell_id,
                chrom=chrom,
                auc=float(auc),
                ap=float(ap),
                n_pos=int(len(pos_prob)),
                n_neg=int(len(neg_prob)),
            )
        )

    return pd.DataFrame(rows)


def summarize_auc_ap(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame([{"mean_auc": np.nan, "mean_ap": np.nan, "n_graphs": 0}])
    return pd.DataFrame(
        [
            {
                "mean_auc": float(df["auc"].mean()),
                "mean_ap": float(df["ap"].mean()),
                "n_graphs": int(len(df)),
            }
        ]
    )

