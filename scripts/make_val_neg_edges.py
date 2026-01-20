#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================
# 🧬 make_val_neg_edges.py
#
# Precompute and store negative edges into VAL .pt graphs:
#   - recon_neg_edge_index : negatives for reconstruction loss
#   - ref_neg_edge_index   : negatives for loop(head) loss (uses ref loops)
#
# This script reads .pt files (PyG Data) and writes updated .pt files.
# Supports Lee / Nagano via src.model.ref_loops provider.
# ==============================================================

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from tqdm.auto import tqdm

from src.model.samplers import sample_negatives_mixed
from src.model.ref_loops import build_ref_loop_provider, SamplerArgs

# optional dataset loaders (if you want to ensure attributes exist)
from src.dataset.lee import LeeDataset
from src.dataset.nagano import NaganoDataset
from src.dataset.pt_dataset import PTGraphDataset


# -----------------------
# helpers
# -----------------------
def mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def normalize_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    """
    - remove self-loops
    - normalize to i<j (undirected unique)
    """
    if edge_index.numel() == 0:
        return edge_index.long()
    ei = edge_index.long()
    mask = ei[0] != ei[1]
    ei = ei[:, mask]
    a = torch.minimum(ei[0], ei[1])
    b = torch.maximum(ei[0], ei[1])
    ei = torch.stack([a, b], dim=0)
    # unique
    keys = ei[0] * (ei.max().item() + 1) + ei[1]
    uniq = torch.unique(keys)
    ei = torch.stack([uniq // (ei.max().item() + 1), uniq % (ei.max().item() + 1)], dim=0)
    return ei


def infer_dataset(dataset: str, data_dir: str, label_path: Optional[str]):
    dataset = dataset.lower()
    if dataset == "lee":
        return LeeDataset(data_dir=data_dir, label_path=label_path)
    if dataset == "nagano":
        return NaganoDataset(data_dir=data_dir, label_path=label_path)
    # fallback generic
    return PTGraphDataset(split_dir=data_dir, label_path=label_path)


def get_cell_meta_fallback(data, dataset: str, filename: str) -> None:
    """
    If your saved .pt doesn't contain cell_id/cell_type/chrom,
    try to derive from filename conventions.
    """
    if not hasattr(data, "cell_id"):
        stem = Path(filename).stem
        if "_chr" in stem:
            data.cell_id = stem.rsplit("_chr", 1)[0]
        else:
            data.cell_id = stem

    if not hasattr(data, "chrom"):
        stem = Path(filename).stem
        if "_chr" in stem:
            chrom_part = stem.rsplit("_chr", 1)[1]
            chrom_part = chrom_part.replace("chr", "")
            data.chrom = f"chr{chrom_part}"
        else:
            data.chrom = "chr?"

    # cell_type is dataset-specific; if absent, provider may return empty loops
    if not hasattr(data, "cell_type"):
        data.cell_type = "UNKNOWN"


@torch.no_grad()
def build_recon_negs(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    *,
    neg_pos_ratio: float,
    near_lower: int,
    near_upper: int,
    near_ratio: float,
    oversample_factor: int,
    max_rounds: int,
) -> torch.Tensor:
    pos = normalize_edge_index(edge_index).to(device)
    num_pos = int(pos.size(1))
    num_neg = int(num_pos * neg_pos_ratio)
    if num_neg <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    neg = sample_negatives_mixed(
        pos_edge_index=pos,
        num_nodes=int(num_nodes),
        num_neg=num_neg,
        device=device,
        near_lower=near_lower,
        near_upper=near_upper,
        near_ratio=near_ratio,
        oversample_factor=oversample_factor,
        max_rounds=max_rounds,
    )
    return neg.long()


@torch.no_grad()
def build_ref_negs(
    data,
    pos_loop: torch.Tensor,
    device: torch.device,
    sampler_args: SamplerArgs,
) -> torch.Tensor:
    """
    For loop supervision:
      - if pos_loop empty -> empty
      - else sample negatives based on pos_loop
    """
    if pos_loop is None or pos_loop.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    pos_loop = normalize_edge_index(pos_loop).to(device)

    num_loop = int(pos_loop.size(1))
    num_neg = int(num_loop * sampler_args.neg_pos_ratio)
    if num_neg <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    neg = sample_negatives_mixed(
        pos_edge_index=pos_loop,
        num_nodes=int(data.num_nodes),
        num_neg=num_neg,
        device=device,
        near_lower=sampler_args.near_lower,
        near_upper=sampler_args.near_upper,
        near_ratio=sampler_args.near_ratio,
        oversample_factor=sampler_args.oversample_factor,
        max_rounds=sampler_args.max_rounds,
    )
    return neg.long()


# -----------------------
# main
# -----------------------
def main():
    p = argparse.ArgumentParser()

    # input/output
    p.add_argument("--dataset", required=True, choices=["lee", "nagano", "generic"])
    p.add_argument("--val_dir", required=True, help="directory containing VAL .pt files")
    p.add_argument("--label_path", default=None, help="used by Lee/Nagano Dataset classes")
    p.add_argument("--out_dir", default=None, help="if not set, overwrite in-place")
    p.add_argument("--overwrite", action="store_true", help="overwrite files in val_dir")

    # fields
    p.add_argument("--recon_field", default="recon_neg_edge_index")
    p.add_argument("--ref_field", default="ref_neg_edge_index")

    # toggles
    p.add_argument("--make_recon", action="store_true", help="create recon negatives")
    p.add_argument("--make_ref", action="store_true", help="create ref(loop) negatives")
    p.add_argument("--force", action="store_true", help="recompute even if field exists")

    # ref loops
    p.add_argument("--ref_loop_path", default=None, help="required if --make_ref")
    p.add_argument(
        "--val_neg_field",
        default="ref_neg_edge_index",
        help="provider uses this name at validation time; keep consistent",
    )
    p.add_argument(
        "--neuron_types",
        nargs="*",
        default=["Sst", "L23", "L4", "L5", "L6", "Vip", "Pvalb", "Ndnf"],
        help="Lee only: neuron subtype list",
    )

    # sampler (shared)
    p.add_argument("--neg_pos_ratio", type=float, default=1.0)
    p.add_argument("--near_lower", type=int, default=5)
    p.add_argument("--near_upper", type=int, default=100)
    p.add_argument("--near_ratio", type=float, default=0.7)
    p.add_argument("--oversample_factor", type=int, default=3)
    p.add_argument("--max_rounds", type=int, default=4)

    # device
    p.add_argument("--device", default="cuda:0")

    args = p.parse_args()

    if not args.make_recon and not args.make_ref:
        raise ValueError("Nothing to do. Use --make_recon and/or --make_ref")

    if args.make_ref and args.ref_loop_path is None:
        raise ValueError("--make_ref requires --ref_loop_path")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    val_dir = Path(args.val_dir)
    if not val_dir.exists():
        raise FileNotFoundError(f"val_dir not found: {val_dir}")

    if args.out_dir is None:
        if not args.overwrite:
            raise ValueError("Set --out_dir OR use --overwrite for in-place update.")
        out_dir = val_dir
    else:
        out_dir = Path(args.out_dir)
        mkdir(str(out_dir))

    # dataset wrapper to get file_list + metadata parsing
    ds = infer_dataset(args.dataset, str(val_dir), args.label_path)

    # ref loop provider
    provider = None
    if args.make_ref:
        provider = build_ref_loop_provider(
            dataset=args.dataset,
            ref_loop_path=args.ref_loop_path,
            device=device,
            neuron_types=set(args.neuron_types) if args.dataset == "lee" else None,
            val_neg_field=args.val_neg_field,
        )

    sargs = SamplerArgs(
        neg_pos_ratio=args.neg_pos_ratio,
        near_lower=args.near_lower,
        near_upper=args.near_upper,
        near_ratio=args.near_ratio,
        oversample_factor=args.oversample_factor,
        max_rounds=args.max_rounds,
    )

    updated = 0

    for i, fname in enumerate(tqdm(ds.file_list, desc="Make VAL neg edges")):
        src_path = val_dir / fname
        dst_path = out_dir / fname

        # load through dataset to ensure attributes are set as your dataset class defines
        data = ds[i]

        # fallback if needed
        get_cell_meta_fallback(data, args.dataset, fname)

        # ensure edge_index normalized for recon
        if hasattr(data, "edge_index"):
            data.edge_index = normalize_edge_index(data.edge_index)

        # -------- recon negs --------
        if args.make_recon:
            has_field = hasattr(data, args.recon_field)
            if args.force or (not has_field):
                neg_edge = build_recon_negs(
                    data.edge_index,
                    int(data.num_nodes),
                    device=device,
                    neg_pos_ratio=args.neg_pos_ratio,
                    near_lower=args.near_lower,
                    near_upper=args.near_upper,
                    near_ratio=args.near_ratio,
                    oversample_factor=args.oversample_factor,
                    max_rounds=args.max_rounds,
                )
                setattr(data, args.recon_field, neg_edge.cpu())

        # -------- ref(loop) negs --------
        if args.make_ref:
            has_field = hasattr(data, args.ref_field)
            if args.force or (not has_field):
                pos_loop = provider.get_pos(data)  # (2, E) or empty
                neg_loop = build_ref_negs(data, pos_loop, device=device, sampler_args=sargs)
                setattr(data, args.ref_field, neg_loop.cpu())

        # save
        torch.save(data, dst_path)
        updated += 1

    print(f"✅ Done. Updated {updated} files.")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()

