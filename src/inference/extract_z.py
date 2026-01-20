# src/inference/extract_z.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm


# =========================================================
# Config
# =========================================================

@dataclass
class ZExtractionConfig:
    """
    Extract per-graph node embeddings (z) from a trained model and save as .npz.

    For each chromosome-wise graph (cell_id, chrom):
      - run model.encode(x, edge_index, batch) -> (z_mod, g)
      - save z_mod to compressed .npz

    Output:
      out_dir/
        z_npz/
          <cell_id>.<chrom>.z.npz   (key: 'z', optional 'g')
        metadata.tsv
        config.json
    """
    batch_size: int = 1
    num_workers: int = 2

    # edge cleaning (match training convention)
    remove_self_loops: bool = True
    keep_upper_triangle: bool = True  # keep i < j

    # output format
    filename_sep: str = "."
    save_g: bool = False  # additionally save graph embedding g into same npz

    # save
    save_metadata: bool = True


# =========================================================
# Helpers
# =========================================================

def _to_str(x: Any) -> str:
    if isinstance(x, (list, tuple)) and len(x) > 0:
        x = x[0]
    if isinstance(x, torch.Tensor):
        if x.ndim == 0:
            return str(x.item())
        return str(x.detach().cpu().numpy().tolist())
    return str(x)


def _normalize_cell_id(cid: Any) -> str:
    s = _to_str(cid)
    return s.replace(".", "_").replace(" ", "_")


def _normalize_chrom(ch: Any) -> str:
    return _to_str(ch)


def _clean_edge_index(edge_index: torch.Tensor, *, remove_self_loops: bool, keep_upper_triangle: bool) -> torch.Tensor:
    ei = edge_index
    if remove_self_loops:
        m = ei[0] != ei[1]
        ei = ei[:, m]
    if keep_upper_triangle:
        m2 = ei[0] < ei[1]
        ei = ei[:, m2]
    return ei


def _infer_cell_and_chrom_from_stem(stem: str) -> Tuple[str, str]:
    """
    Expected: <cell>_chr<chrom>  (e.g., A10_AD001_L23_chr1)
    """
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
        if all(torch.is_tensor(v) for v in obj.values()):
            return obj
    raise ValueError(f"Unsupported weight format: {weight_path}")


# =========================================================
# Core
# =========================================================

@torch.no_grad()
def extract_and_save_z(
    *,
    model: nn.Module,
    loader: DataLoader,
    out_dir: str | Path,
    device: torch.device,
    cfg: Optional[ZExtractionConfig] = None,
    weight_path: Optional[str] = None,
    strict: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Extract z for each graph and save as compressed .npz.

    Returns
    -------
    meta_df: DataFrame with per-graph info
    """
    cfg = cfg or ZExtractionConfig()
    out_dir = Path(out_dir)
    z_dir = out_dir / "z_npz"
    z_dir.mkdir(parents=True, exist_ok=True)

    if weight_path is not None:
        sd = load_state_dict_any(weight_path, device=device)
        model.load_state_dict(sd, strict=strict)

    model = model.to(device).eval()

    rows: List[dict] = []
    it = tqdm(loader, desc="Extract z (.npz)", disable=not verbose)

    for batch in it:
        data = batch.to(device)

        # --- cell/chrom ---
        if hasattr(data, "cell_id"):
            cell_id = _normalize_cell_id(getattr(data, "cell_id"))
        else:
            stem = str(getattr(data, "__file_stem__", "unknown"))
            cell_id, _ = _infer_cell_and_chrom_from_stem(stem)
            cell_id = _normalize_cell_id(cell_id)

        if hasattr(data, "chrom"):
            chrom = _normalize_chrom(getattr(data, "chrom"))
        else:
            stem = str(getattr(data, "__file_stem__", "unknown"))
            _, chrom = _infer_cell_and_chrom_from_stem(stem)
            chrom = _normalize_chrom(chrom)

        # --- edge cleaning ---
        if hasattr(data, "edge_index") and data.edge_index is not None:
            data.edge_index = _clean_edge_index(
                data.edge_index,
                remove_self_loops=cfg.remove_self_loops,
                keep_upper_triangle=cfg.keep_upper_triangle,
            )

        # --- batch vector ---
        batch_vec = getattr(data, "batch", None)
        if batch_vec is None:
            batch_vec = torch.zeros(int(data.num_nodes), dtype=torch.long, device=device)

        # --- forward (require encode) ---
        if not hasattr(model, "encode"):
            raise AttributeError("Model must implement encode(x, edge_index, batch_vec) -> (z, g).")

        z_mod, g = model.encode(data.x, data.edge_index, batch_vec)

        z_np = z_mod.detach().cpu().numpy().astype(np.float32)

        # save: <cell>.<chrom>.z.npz
        fname = f"{cell_id}{cfg.filename_sep}{chrom}{cfg.filename_sep}z.npz"
        save_path = z_dir / fname

        if cfg.save_g:
            g_np = g.detach().cpu().numpy().astype(np.float32)
            np.savez_compressed(save_path, z=z_np, g=g_np)
        else:
            np.savez_compressed(save_path, z=z_np)

        rows.append(
            {
                "cell_id": cell_id,
                "chrom": chrom,
                "num_nodes": int(z_np.shape[0]),
                "z_dim": int(z_np.shape[1]),
                "z_path": str(save_path),
            }
        )

    meta_df = pd.DataFrame(rows).sort_values(["cell_id", "chrom"]).reset_index(drop=True)

    # save metadata/config
    if cfg.save_metadata:
        meta_df.to_csv(out_dir / "metadata.tsv", sep="\t", index=False)

    with open(out_dir / "config.json", "w") as f:
        json.dump(
            {"out_dir": str(out_dir), **asdict(cfg), "weight_path": weight_path},
            f,
            indent=2,
        )

    return meta_df

