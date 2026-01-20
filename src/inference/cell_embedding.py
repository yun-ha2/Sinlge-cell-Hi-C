# src/inference/cell_embedding.py
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm


# -----------------------------
# Config
# -----------------------------
@dataclass
class CellEmbeddingConfig:
    """
    Extract cell-level embeddings by aggregating chromosome-wise graph embeddings.

    Embedding definition:
      - For each (cell_id, chromosome) graph, compute graph embedding g
      - For each cell_id, mean-pool g across chromosomes

    Outputs:
      - cell_embeddings.npy  : (n_cells, z_dim)
      - cell_names.txt       : one column CSV (cell_id)
      - metadata.tsv         : per-cell stats (num_graphs, chrom list, etc.)
      - config.json          : (optional) saved by caller
    """
    batch_size: int = 1
    num_workers: int = 2

    # aggregation over chromosome graphs
    agg: str = "mean"  # {"mean", "sum"} (mean recommended)

    # edge cleaning
    remove_self_loops: bool = True
    keep_upper_triangle: bool = True  # keep i<j

    # filtering
    min_graphs_per_cell: int = 1  # at least this many chr graphs to produce cell embedding

    # i/o
    save_metadata: bool = True


# -----------------------------
# Helpers
# -----------------------------
def _to_str(x: Any) -> str:
    if isinstance(x, (list, tuple)) and len(x) > 0:
        x = x[0]
    if isinstance(x, torch.Tensor):
        # scalar tensor
        return str(x.item()) if x.ndim == 0 else str(x.detach().cpu().numpy().tolist())
    return str(x)


def _clean_edge_index(edge_index: torch.Tensor, *, remove_self_loops: bool, keep_upper_triangle: bool) -> torch.Tensor:
    ei = edge_index
    if remove_self_loops:
        m = ei[0] != ei[1]
        ei = ei[:, m]
    if keep_upper_triangle:
        m2 = ei[0] < ei[1]
        ei = ei[:, m2]
    return ei


def _infer_cell_and_chrom_from_filename(path: str) -> Tuple[str, str]:
    """
    Fallback when Data has no cell_id/chrom attributes.
    Expected patterns:
      - <cell>_chr1.pt
      - <cell>_chrX.pt
    """
    stem = Path(path).stem  # without .pt
    if "_chr" not in stem:
        return stem, "unknown"
    cell, chrom = stem.rsplit("_chr", 1)
    chrom = chrom if chrom.startswith("chr") else f"chr{chrom}"
    return cell, chrom


def load_state_dict_any(weight_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Supports:
      (A) torch.save(model.state_dict())
      (B) checkpoint dict with keys: 'state_dict' or 'model_state'
    """
    obj = torch.load(weight_path, map_location=device)
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state" in obj and isinstance(obj["model_state"], dict):
            return obj["model_state"]
        # already a state_dict
        if all(torch.is_tensor(v) for v in obj.values()):
            return obj
    raise ValueError(f"Unsupported weight format: {weight_path}")


# -----------------------------
# Core API
# -----------------------------
@torch.no_grad()
def extract_cell_embeddings(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: CellEmbeddingConfig,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Parameters
    ----------
    model:
        Must expose encode(x, edge_index, batch) -> (z_mod, g) OR
        forward to return g. We try encode() first.
    loader:
        yields PyG Data or Batch objects.
    cfg:
        extraction config.

    Returns
    -------
    X: (n_cells, z_dim) numpy float32
    cell_ids: list[str] in same order as X
    meta: DataFrame with per-cell info
    """
    model.eval()

    # storage: cell_id -> list[g]
    pool: Dict[str, List[torch.Tensor]] = {}
    chroms: Dict[str, List[str]] = {}

    it = tqdm(loader, desc="Extract cell embeddings", disable=not verbose)

    for batch in it:
        data = batch.to(device)

        # ---- cell_id / chrom ----
        if hasattr(data, "cell_id"):
            cell_id = _to_str(getattr(data, "cell_id"))
        else:
            # if dataset provides file path in data, try it; else unknown
            path = getattr(data, "__file_path__", None)
            cell_id, _ = _infer_cell_and_chrom_from_filename(str(path)) if path else ("unknown", "unknown")

        if hasattr(data, "chrom"):
            chrom = _to_str(getattr(data, "chrom"))
        else:
            path = getattr(data, "__file_path__", None)
            _, chrom = _infer_cell_and_chrom_from_filename(str(path)) if path else ("unknown", "unknown")

        # ---- edge cleaning (optional but makes behavior consistent with training) ----
        if hasattr(data, "edge_index") and data.edge_index is not None:
            data.edge_index = _clean_edge_index(
                data.edge_index,
                remove_self_loops=cfg.remove_self_loops,
                keep_upper_triangle=cfg.keep_upper_triangle,
            )

        # ---- batch vector ----
        batch_vec = getattr(data, "batch", None)
        if batch_vec is None:
            # batch_size=1 or Data without batch attr
            batch_vec = torch.zeros(int(data.num_nodes), dtype=torch.long, device=device)

        # ---- forward: prefer encode() ----
        if hasattr(model, "encode"):
            _, g = model.encode(data.x, data.edge_index, batch_vec)
        else:
            # fallback: model(data) should return g
            g = model(data)

        # In case g is (B, D) with B=1
        if isinstance(g, torch.Tensor) and g.ndim == 2 and g.size(0) == 1:
            g = g.squeeze(0)

        if cell_id not in pool:
            pool[cell_id] = []
            chroms[cell_id] = []

        pool[cell_id].append(g.detach().cpu())
        chroms[cell_id].append(chrom)

    # ---- aggregate per cell ----
    cell_ids = sorted(pool.keys())
    rows = []
    embeds = []

    for cid in cell_ids:
        gs = torch.stack(pool[cid], dim=0)  # (n_graphs, D)
        if cfg.agg == "sum":
            g_cell = gs.sum(dim=0)
        else:
            g_cell = gs.mean(dim=0)

        if gs.size(0) < cfg.min_graphs_per_cell:
            continue

        embeds.append(g_cell.numpy())
        rows.append(
            {
                "cell_id": cid,
                "n_graphs": int(gs.size(0)),
                "chroms": ",".join(chroms[cid]),
            }
        )

    if len(embeds) == 0:
        raise RuntimeError("No cell embeddings produced. Check dataset cell_id/chrom and filtering rules.")

    X = np.vstack(embeds).astype(np.float32)
    meta = pd.DataFrame(rows)

    # Important: meta row order must match X order
    meta = meta.sort_values("cell_id").reset_index(drop=True)
    cell_ids_sorted = meta["cell_id"].astype(str).tolist()

    # Reorder X accordingly (since we appended in sorted(pool.keys()) but also filtered)
    # Here we built embeds in that same loop, but after filtering, safest to reorder explicitly:
    cid_to_vec = {r["cell_id"]: v for r, v in zip(rows, embeds)}
    X = np.vstack([cid_to_vec[cid] for cid in cell_ids_sorted]).astype(np.float32)

    return X, cell_ids_sorted, meta


def save_cell_embeddings(
    *,
    out_dir: str | Path,
    X: np.ndarray,
    cell_ids: List[str],
    meta: Optional[pd.DataFrame] = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "cell_embeddings.npy", X.astype(np.float32))
    pd.Series(cell_ids, name="cell_id").to_csv(out_dir / "cell_names.txt", index=False)

    if meta is not None:
        meta.to_csv(out_dir / "metadata.tsv", sep="\t", index=False)


# -----------------------------
# Convenience: full pipeline
# -----------------------------
def run_cell_embedding_inference(
    *,
    model: nn.Module,
    weight_path: Optional[str],
    loader: DataLoader,
    out_dir: str | Path,
    device: torch.device,
    cfg: Optional[CellEmbeddingConfig] = None,
    strict: bool = True,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Typical usage in scripts:
      - instantiate model architecture
      - load weights
      - run extraction + save

    Returns X, cell_ids, meta.
    """
    cfg = cfg or CellEmbeddingConfig()

    if weight_path is not None:
        sd = load_state_dict_any(weight_path, device=device)
        model.load_state_dict(sd, strict=strict)

    model = model.to(device)

    X, cell_ids, meta = extract_cell_embeddings(
        model=model,
        loader=loader,
        device=device,
        cfg=cfg,
        verbose=True,
    )
    save_cell_embeddings(out_dir=out_dir, X=X, cell_ids=cell_ids, meta=meta)
    return X, cell_ids, meta

