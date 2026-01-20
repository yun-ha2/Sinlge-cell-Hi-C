#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scripts/eval_clustering_one.py

One-shot clustering evaluation:
- (model checkpoint) + (pt_dir split) -> cell embedding -> KMeans -> metrics

Requires:
- src/inference/cell_embedding.py  (already in your repo)
- src/eval/clustering.py           (this file)
- src/dataset/{lee,nagano}.py
- src/model/{encoder_gcn,gae_film}.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.inference.cell_embedding import CellEmbeddingConfig, run_cell_embedding_inference
from src.evaluation.clustering import (
    load_label_maps,
    align_embeddings_to_labels,
    evaluate_kmeans,
    build_embeddings_table,
    metrics_to_frame,
)

from src.dataset.lee import LeeDataset
from src.dataset.nagano import NaganoDataset
from src.model.encoder_gcn import GCNEncoder
from src.model.gae_film import GAEFiLM


def seed_all(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_in_dim(pt_dir: str) -> int:
    files = sorted([f for f in os.listdir(pt_dir) if f.endswith(".pt")])
    if not files:
        raise RuntimeError(f"No .pt files in {pt_dir}")
    sample = torch.load(os.path.join(pt_dir, files[0]), map_location="cpu")
    return int(sample.x.shape[1])


def build_dataset(dataset: str, pt_dir: str, label_path: str):
    dataset = dataset.lower()
    if dataset == "lee":
        return LeeDataset(data_dir=pt_dir, label_path=label_path)
    if dataset == "nagano":
        return NaganoDataset(data_dir=pt_dir, label_path=label_path)
    raise ValueError("dataset must be lee or nagano")


def main():
    p = argparse.ArgumentParser()

    # fixed pair
    p.add_argument("--dataset", required=True, choices=["lee", "nagano"])
    p.add_argument("--pt_dir", required=True)
    p.add_argument("--label_path", required=True)
    p.add_argument("--model_path", required=True)

    # model (must match training)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--z_dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--target_dim", type=int, default=50)

    # embedding extraction
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--agg", default="mean", choices=["mean", "sum"])
    p.add_argument("--min_graphs_per_cell", type=int, default=1)
    p.add_argument("--remove_self_loops", action="store_true")
    p.add_argument("--keep_upper_triangle", action="store_true")

    # clustering
    p.add_argument("--kmeans_seed", type=int, default=42)

    # runtime
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--seed", type=int, default=123)

    # out
    p.add_argument("--out_dir", required=True)

    args = p.parse_args()
    seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) dataset/loader
    ds = build_dataset(args.dataset, args.pt_dir, args.label_path)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 2) labels
    cell_to_type, type_to_idx, idx_to_type = load_label_maps(args.dataset, args.label_path)
    n_classes = len(idx_to_type)

    # 3) model arch + weights -> embedding extraction
    in_dim = infer_in_dim(args.pt_dir)
    encoder = GCNEncoder(in_dim=in_dim, hidden_dim=args.hidden_dim, z_dim=args.z_dim, dropout=args.dropout)
    model = GAEFiLM(encoder=encoder, z_dim=args.z_dim, target_dim=args.target_dim)

    emb_cfg = CellEmbeddingConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        agg=args.agg,
        remove_self_loops=args.remove_self_loops,
        keep_upper_triangle=args.keep_upper_triangle,
        min_graphs_per_cell=args.min_graphs_per_cell,
        save_metadata=True,
    )

    emb_out = out_dir / "embeddings"
    X, cell_ids, _meta = run_cell_embedding_inference(
        model=model,
        weight_path=args.model_path,
        loader=loader,
        out_dir=str(emb_out),
        device=device,
        cfg=emb_cfg,
        strict=args.strict,
    )

    # 4) align to labels
    X_use, y_true, used_ids, used_types = align_embeddings_to_labels(
        X=X, cell_ids=cell_ids, cell_to_type=cell_to_type, type_to_idx=type_to_idx
    )

    # 5) kmeans + metrics
    y_pred, metrics, _X_scaled = evaluate_kmeans(
        X=X_use, y_true=y_true, n_classes=n_classes, random_state=args.kmeans_seed
    )

    # 6) save tables
    df_emb = build_embeddings_table(
        X=X_use, cell_ids=used_ids, cell_types=used_types, y_true=y_true, y_pred=y_pred
    )
    df_emb.to_csv(out_dir / "embeddings_with_labels.csv", index=False)

    df_m = metrics_to_frame(
        metrics,
        extra={
            "dataset": args.dataset,
            "pt_dir": args.pt_dir,
            "label_path": args.label_path,
            "model_path": args.model_path,
        },
    )
    df_m.to_csv(out_dir / "metrics.csv", index=False)

    print(
        f"[DONE] n_cells={metrics.n_cells} n_classes={metrics.n_classes} "
        f"ARI={metrics.ari:.3f} NMI={metrics.nmi:.3f} AMI={metrics.ami:.3f} HUNG={metrics.hungarian:.3f}"
    )
    print(f"- metrics: {out_dir / 'metrics.csv'}")
    print(f"- embeddings: {out_dir / 'embeddings_with_labels.csv'}")
    print(f"- raw embeddings (npy/txt/meta): {emb_out}")


if __name__ == "__main__":
    main()

