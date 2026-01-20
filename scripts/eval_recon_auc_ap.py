#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scripts/eval_recon_auc_ap.py

Compute reconstruction AUC/AP on validation graphs.

Requirements:
- val_dir contains .pt graphs
- each graph has:
    - x, edge_index
    - recon_neg_edge_index  (precomputed by make_val_neg_edges.py)
- model exposes:
    - encode(x, edge_index, batch_vec) -> (z_mod, g)
    - main_dec(z_mod, edge_index) -> logits

This script is dataset-agnostic (Lee/Nagano).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

# project
from src.evaluation.recon_auc_ap import (
    ReconEvalConfig,
    evaluate_reconstruction_auc_ap,
    summarize_auc_ap,
    load_state_dict_any,
)

# ------------------------------------------------------------
# Minimal dataset: load *.pt files as-is
# ------------------------------------------------------------
class PTGraphFolder(Dataset):
    def __init__(self, data_dir: str | Path):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.files = sorted([p for p in self.data_dir.iterdir() if p.suffix == ".pt"])
        if len(self.files) == 0:
            raise RuntimeError(f"No .pt files found in: {self.data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        data = torch.load(p)
        # help evaluator identify source file
        data.file_name = p.name
        return data


# ------------------------------------------------------------
# IMPORTANT:
# Replace this import with your actual model class path.
# ------------------------------------------------------------
def build_model_from_args(in_dim: int, z_dim: int, hidden: int, dropout: float, target_dim: int):
    """
    Keep model definition in src/model in your repo.
    Here we import from your project modules.
    """
    from src.model.encoder_gcn import GCNEncoder
    from src.model.decoders import MLPDecoder  
    from src.model.gae_film import GAEFiLM     

    enc = GCNEncoder(in_dim=in_dim, hidden_dim=hidden, z_dim=z_dim, dropout=dropout)
    dec = MLPDecoder(z_dim=z_dim)
    model = GAEFiLM(encoder=enc, main_dec=dec, z_dim=z_dim, target_dim=target_dim)
    return model


def infer_in_dim_from_dir(val_dir: str | Path) -> int:
    val_dir = Path(val_dir)
    first = next(p for p in sorted(val_dir.iterdir()) if p.suffix == ".pt")
    sample = torch.load(first)
    return int(sample.x.shape[1])


def main():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--val_dir", required=True)
    p.add_argument("--out_csv", required=True)

    # checkpoint
    p.add_argument("--ckpt", required=True, help="model .pt (state_dict or checkpoint dict)")

    # model hyperparams (must match training)
    p.add_argument("--z_dim", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--target_dim", type=int, default=50)

    # eval config
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no_upper_triangle", action="store_true")
    p.add_argument("--keep_self_loops", action="store_true")
    p.add_argument("--strict", action="store_true", help="strict state_dict load")

    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # dataset/loader
    ds = PTGraphFolder(args.val_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # model
    in_dim = infer_in_dim_from_dir(args.val_dir)
    model = build_model_from_args(
        in_dim=in_dim,
        z_dim=args.z_dim,
        hidden=args.hidden_dim,
        dropout=args.dropout,
        target_dim=args.target_dim,
    ).to(device)

    sd = load_state_dict_any(args.ckpt, device=device)
    model.load_state_dict(sd, strict=args.strict)

    # eval
    cfg = ReconEvalConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        remove_self_loops=not args.keep_self_loops,
        keep_upper_triangle=not args.no_upper_triangle,
        skip_if_no_neg=True,
    )

    df = evaluate_reconstruction_auc_ap(model=model, loader=loader, device=device, cfg=cfg, verbose=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    summ = summarize_auc_ap(df)
    summ_path = out_csv.with_name(out_csv.stem + "_summary.csv")
    summ.to_csv(summ_path, index=False)

    print(f"[OK] saved per-graph: {out_csv}")
    print(f"[OK] saved summary : {summ_path}")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()

