# src/inference/loops.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm.auto import tqdm


# ----------------------------
# Config
# ----------------------------
@dataclass(frozen=True)
class LoopPredictConfig:
    resolution: int = 10000
    min_bin_dist: int = 2
    max_bin_dist: int = 20
    threshold: float = 0.7

    # output
    out_format: str = "tsv"   # "tsv" or "bedpe" (tsv recommended; bedpe is same columns)
    save_per_sample: bool = True
    merge_all: bool = False   # if True -> write merged file at end (needs buffering)


# ----------------------------
# Candidate generation
# ----------------------------
@torch.no_grad()
def make_candidate_edges(
    num_bins: int,
    min_bin_dist: int,
    max_bin_dist: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate all candidate (i,j) with i<j and min_dist<=j-i<=max_dist.
    Returns edge_index shape (2, E) on device.
    """
    src: List[int] = []
    dst: List[int] = []

    for i in range(num_bins):
        j_min = i + int(min_bin_dist)
        j_max = min(i + int(max_bin_dist), num_bins - 1)
        if j_min <= j_max:
            js = range(j_min, j_max + 1)
            src.extend([i] * (j_max - j_min + 1))
            dst.extend(js)

    if len(src) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    return torch.tensor([src, dst], dtype=torch.long, device=device)


def filter_edges_by_distance(
    edges: torch.Tensor,
    min_bin_dist: int,
    max_bin_dist: int,
) -> torch.Tensor:
    if edges.numel() == 0:
        return edges
    i = edges[0]
    j = edges[1]
    dist = (j - i).abs()
    mask = (dist >= int(min_bin_dist)) & (dist <= int(max_bin_dist))
    return edges[:, mask]


# ----------------------------
# Helpers: ID, chrom
# ----------------------------
def _as_str(x) -> str:
    if isinstance(x, (list, tuple)) and len(x) > 0:
        x = x[0]
    if torch.is_tensor(x):
        x = x.item()
    return str(x)


def get_cell_id(data: Data) -> str:
    return _as_str(getattr(data, "cell_id", "UNKNOWN"))


def get_chrom(data: Data) -> str:
    return _as_str(getattr(data, "chrom", "chr?"))


def to_bedpe_df(
    cell_id: str,
    chrom: str,
    i_bins: np.ndarray,
    j_bins: np.ndarray,
    scores: np.ndarray,
    logits: np.ndarray,
    resolution: int,
) -> pd.DataFrame:
    x1 = i_bins * resolution
    x2 = x1 + resolution
    y1 = j_bins * resolution
    y2 = y1 + resolution

    return pd.DataFrame(
        {
            "cell_id": cell_id,
            "chrom1": chrom,
            "start1": x1,
            "end1": x2,
            "chrom2": chrom,
            "start2": y1,
            "end2": y2,
            "score": scores,
            "logit": logits,
            "bin1": i_bins,
            "bin2": j_bins,
        }
    )


# ----------------------------
# Main inference
# ----------------------------
@torch.no_grad()
def predict_loops(
    *,
    model,
    loader: Iterable[Data],
    out_dir: str | Path,
    cfg: LoopPredictConfig,
    mode: str = "full",  # "full" or "existing"
    device: torch.device | str = "cuda:0",
) -> Optional[pd.DataFrame]:
    """
    Predict loops with a loop-head fine-tuned model.

    Parameters
    ----------
    mode:
      - "existing": score only existing edges in data.edge_index (filtered by distance)
      - "full": score all candidate pairs within [min_bin_dist, max_bin_dist]

    Returns
    -------
    If cfg.merge_all=True: returns merged dataframe. Else returns None.
    """
    device = torch.device(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    merged_rows: List[pd.DataFrame] = []

    for data in tqdm(loader, desc=f"Loop inference ({mode})"):
        data = data.to(device)

        cell_id = get_cell_id(data)
        chrom = get_chrom(data)

        # encode (FiLM)
        num_nodes = int(data.num_nodes)
        batch_vec = torch.zeros(num_nodes, dtype=torch.long, device=device)
        z_mod, _g = model.encode(data.x, data.edge_index, batch_vec)

        # candidate edges
        if mode == "existing":
            cand = filter_edges_by_distance(data.edge_index, cfg.min_bin_dist, cfg.max_bin_dist)
        elif mode == "full":
            cand = make_candidate_edges(num_nodes, cfg.min_bin_dist, cfg.max_bin_dist, device=device)
        else:
            raise ValueError("mode must be 'existing' or 'full'")

        if cand.numel() == 0:
            continue

        # score
        logits = model.forward_loop(z_mod, cand)             # (E,)
        probs = torch.sigmoid(logits)                        # (E,)
        keep = probs > float(cfg.threshold)

        if keep.sum().item() == 0:
            continue

        cand_kept = cand[:, keep]
        probs_kept = probs[keep]
        logits_kept = logits[keep]

        # to df
        i_bins = cand_kept[0].detach().cpu().numpy()
        j_bins = cand_kept[1].detach().cpu().numpy()
        scores = probs_kept.detach().cpu().numpy()
        logitv = logits_kept.detach().cpu().numpy()

        df = to_bedpe_df(
            cell_id=cell_id,
            chrom=chrom,
            i_bins=i_bins,
            j_bins=j_bins,
            scores=scores,
            logits=logitv,
            resolution=int(cfg.resolution),
        )

        # write per-sample
        if cfg.save_per_sample:
            fn = out_dir / f"{cell_id}_{chrom}.{cfg.out_format}"
            sep = "\t" if cfg.out_format in ("tsv", "bedpe") else "\t"
            df.to_csv(fn, sep=sep, index=False)

        if cfg.merge_all:
            merged_rows.append(df)

    if cfg.merge_all:
        merged = pd.concat(merged_rows, ignore_index=True) if merged_rows else pd.DataFrame()
        merged_fn = out_dir / f"predicted_loops_merged.{cfg.out_format}"
        merged.to_csv(merged_fn, sep="\t", index=False)
        return merged

    return None

