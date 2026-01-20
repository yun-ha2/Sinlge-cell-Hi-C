#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract node embeddings (z) from chromosome-wise scHi-C graphs (.pt) using a trained model,
and save each graph's z into compressed .npz files.

Outputs:
  out_dir/
    z_npz/
      <cell_id>.<chrom>.z.npz  (key: 'z', optional 'g')
    metadata.tsv
    config.json

Example (Lee):
  python scripts/inference/extract_z.py \
    --dataset lee \
    --pt_dir /path/to/pt_dir \
    --label_path /path/to/Lee/cell_type.txt \
    --weight_path /path/to/model/model.pt \
    --out_dir /path/to/out/z_lee \
    --device cuda:0

Example (Nagano):
  python scripts/inference/extract_z.py \
    --dataset nagano \
    --pt_dir /path/to/pt_dir \
    --label_path /path/to/Nagano/cell_type.txt \
    --weight_path /path/to/model.pt \
    --out_dir /path/to/out/z_nagano \
    --device cuda:0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GlobalAttention

# project imports
from src.inference.extract_z import extract_and_save_z, ZExtractionConfig


# =========================================================
# Dataset
# =========================================================

def _clean_edges(data):
    # remove self-loop + keep i<j
    ei = data.edge_index
    m1 = ei[0] != ei[1]
    ei = ei[:, m1]
    m2 = ei[0] < ei[1]
    data.edge_index = ei[:, m2]
    return data


class LeePTDataset(Dataset):
    """
    Expects pt filenames like:
      <cell_id>_chr1.pt
      A10_AD008_Pvalb_chr1.pt
    label_path has columns: file_name, cell_type, ...
    """
    def __init__(self, pt_dir: str, label_path: str):
        self.pt_dir = Path(pt_dir)
        self.files = sorted([p for p in self.pt_dir.glob("*.pt")])

        df = pd.read_csv(label_path, sep="\t")
        # file_name should match <cell_id> (without _chr*)
        self.name2type = dict(zip(df["file_name"], df["cell_type"]))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = torch.load(path)

        stem = path.stem  # no .pt
        if "_chr" in stem:
            cell_id, chrom = stem.rsplit("_chr", 1)
            chrom = chrom if chrom.startswith("chr") else f"chr{chrom}"
        else:
            cell_id, chrom = stem, "unknown"

        data.cell_id = str(cell_id)
        data.chrom = str(chrom)

        # (optional) carry label if needed later
        data.cell_type = self.name2type.get(cell_id, "UNKNOWN")

        # for fallback inference in src (if needed)
        data.__file_stem__ = stem

        return _clean_edges(data)


class NaganoPTDataset(Dataset):
    """
    Expects pt filenames like:
      1CDX1.101_chr1.pt  or 1CDX1_101_chr1.pt (your code often normalizes later)
    label_path has columns: cell_nm, cell_type, ...
    """
    def __init__(self, pt_dir: str, label_path: str):
        self.pt_dir = Path(pt_dir)
        self.files = sorted([p for p in self.pt_dir.glob("*.pt")])

        df = pd.read_csv(label_path, sep="\t")
        self.cell2type = dict(zip(df["cell_nm"], df["cell_type"]))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = torch.load(path)

        stem = path.stem
        if "_chr" in stem:
            raw_id, chrom = stem.rsplit("_chr", 1)
            chrom = chrom if chrom.startswith("chr") else f"chr{chrom}"
        else:
            raw_id, chrom = stem, "unknown"

        # match your previous convention: "." -> "_"
        cell_nm = str(raw_id).replace(".", "_")

        data.cell_id = cell_nm
        data.chrom = str(chrom)
        data.cell_type = self.cell2type.get(cell_nm, "UNKNOWN")
        data.__file_stem__ = stem

        return _clean_edges(data)


# =========================================================
# Model (must match training)
# =========================================================

class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, h: int = 128, z_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.c1 = GCNConv(in_dim, h)
        self.c2 = GCNConv(h, h)
        self.out = GCNConv(h, z_dim)
        self.drop = dropout

    def forward(self, x, edge):
        x = self.norm(x)
        x = F.relu(self.c1(x, edge))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.c2(x, edge))
        return self.out(x, edge)


class MLPDecoder(nn.Module):
    def __init__(self, z_dim: int, h: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(z_dim * 2, h)
        self.fc2 = nn.Linear(h, 1)

    def forward(self, z, edge):
        zi = z[edge[0]]
        zj = z[edge[1]]
        h = torch.cat([zi, zj], dim=-1)
        h = F.relu(self.fc1(h))
        return self.fc2(h).squeeze(-1)


class GAE_FiLM(nn.Module):
    def __init__(self, encoder: nn.Module, main_dec: nn.Module, z_dim: int = 32, target_dim: int = 50):
        super().__init__()
        self.encoder = encoder
        self.main_dec = main_dec
        self.attn = GlobalAttention(nn.Linear(z_dim, 1))
        self.proj = nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(), nn.Linear(64, target_dim))
        self.film = nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(), nn.Linear(64, z_dim * 2))

    def encode(self, x, edge, batch_vec):
        z_local = self.encoder(x, edge)
        g = self.attn(z_local, batch=batch_vec).squeeze(0)
        gamma, beta = self.film(g).chunk(2, dim=-1)
        z_mod = z_local * (1 + gamma) + beta
        return z_mod, g


def _infer_in_dim(pt_dir: str) -> int:
    pt_dir = Path(pt_dir)
    first = sorted(pt_dir.glob("*.pt"))[0]
    d = torch.load(first)
    return int(d.x.shape[1])


# =========================================================
# Main
# =========================================================

def build_argparser():
    p = argparse.ArgumentParser("Extract z to .npz from trained GAE_FiLM")
    p.add_argument("--dataset", choices=["lee", "nagano"], required=True)
    p.add_argument("--pt_dir", required=True)
    p.add_argument("--label_path", required=True)
    p.add_argument("--weight_path", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--z_dim", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--save_g", action="store_true", help="also store graph embedding g in each npz")
    p.add_argument("--strict", action="store_true", help="strict state_dict loading (default: False)")
    return p


def main():
    args = build_argparser().parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # dataset / loader
    if args.dataset == "lee":
        ds = LeePTDataset(args.pt_dir, args.label_path)
    else:
        ds = NaganoPTDataset(args.pt_dir, args.label_path)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # model
    in_dim = _infer_in_dim(args.pt_dir)
    encoder = GCNEncoder(in_dim, h=args.hidden_dim, z_dim=args.z_dim, dropout=args.dropout)
    decoder = MLPDecoder(args.z_dim)
    model = GAE_FiLM(encoder, decoder, z_dim=args.z_dim, target_dim=50)

    # config for extraction
    cfg = ZExtractionConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        remove_self_loops=True,
        keep_upper_triangle=True,
        save_g=bool(args.save_g),
        save_metadata=True,
    )

    meta = extract_and_save_z(
        model=model,
        loader=loader,
        out_dir=out_dir,
        device=device,
        cfg=cfg,
        weight_path=args.weight_path,
        strict=bool(args.strict),
        verbose=True,
    )

    print("\n✅ Done.")
    print(f"- graphs processed: {len(meta)}")
    print(f"- saved under: {out_dir}")


if __name__ == "__main__":
    main()

