#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================
# scripts/predict_loops.py
#   Loop inference → per-sample bedpe/tsv outputs
# ==============================================================

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import pandas as pd
from torch_geometric.loader import DataLoader

from src.inference.loops import LoopPredictConfig, predict_loops

# your dataset / model modules
from src.dataset.lee import LeeDataset
from src.dataset.nagano import NaganoDataset
from src.model.encoder_gcn import GCNEncoder
from src.model.gae_film import GAEFiLM


def infer_dataset_type(label_path: str) -> str:
    df = pd.read_csv(label_path, sep="\t")
    if "cell_nm" in df.columns:
        return "nagano"
    if "file_name" in df.columns:
        return "lee"
    raise ValueError("label_path must contain either 'cell_nm' (Nagano) or 'file_name' (Lee).")


def build_dataset(dataset: str, data_dir: str, label_path: str):
    if dataset == "nagano":
        return NaganoDataset(data_dir=data_dir, label_path=label_path)
    if dataset == "lee":
        return LeeDataset(data_dir=data_dir, label_path=label_path)
    raise ValueError("dataset must be 'nagano' or 'lee'")


def load_loop_finetuned_model(
    weight_path: str,
    sample_pt_dir: str,
    *,
    z_dim: int,
    target_dim: int,
    loop_dim: int,
    device: torch.device,
):
    # infer in_dim
    pt_files = sorted([f for f in os.listdir(sample_pt_dir) if f.endswith(".pt")])
    if len(pt_files) == 0:
        raise RuntimeError(f"No .pt files found in: {sample_pt_dir}")
    sample = torch.load(os.path.join(sample_pt_dir, pt_files[0]))
    in_dim = int(sample.x.shape[1])

    # build model (must match training 100%)
    enc = GCNEncoder(in_dim=in_dim, hidden_dim=128, z_dim=z_dim, dropout=0.2)
    model = GAEFiLM(
        encoder=enc,
        z_dim=z_dim,
        target_dim=target_dim,
        loop_dim=loop_dim,
    ).to(device)

    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--pt_dir", required=True, help="val/test pt directory")
    p.add_argument("--label_path", required=True)
    p.add_argument("--dataset", default="auto", choices=["auto", "lee", "nagano"])

    p.add_argument("--weights", required=True, help="fine-tuned model .pt (state_dict)")
    p.add_argument("--out_dir", required=True)

    # inference options
    p.add_argument("--mode", default="full", choices=["full", "existing"])
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--resolution", type=int, default=100000)
    p.add_argument("--min_bin_dist", type=int, default=2)
    p.add_argument("--max_bin_dist", type=int, default=20)

    p.add_argument("--save_per_sample", action="store_true")
    p.add_argument("--merge_all", action="store_true")
    p.add_argument("--out_format", default="tsv", choices=["tsv", "bedpe"])

    # model dims
    p.add_argument("--z_dim", type=int, default=32)
    p.add_argument("--target_dim", type=int, default=50)
    p.add_argument("--loop_dim", type=int, default=32)

    # loader
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--device", default="cuda:0")

    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = args.dataset
    if dataset == "auto":
        dataset = infer_dataset_type(args.label_path)

    ds = build_dataset(dataset, args.pt_dir, args.label_path)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = load_loop_finetuned_model(
        weight_path=args.weights,
        sample_pt_dir=args.pt_dir,
        z_dim=args.z_dim,
        target_dim=args.target_dim,
        loop_dim=args.loop_dim,
        device=device,
    )

    cfg = LoopPredictConfig(
        threshold=args.threshold,
        resolution=args.resolution,
        min_bin_dist=args.min_bin_dist,
        max_bin_dist=args.max_bin_dist,
        out_format=args.out_format,
        save_per_sample=args.save_per_sample,
        merge_all=args.merge_all,
    )

    predict_loops(
        model=model,
        loader=loader,
        out_dir=Path(args.out_dir),
        cfg=cfg,
        mode=args.mode,
        device=device,
    )


if __name__ == "__main__":
    main()

