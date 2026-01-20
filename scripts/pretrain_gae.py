#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pretrain GAE+FiLM for scHi-C graphs
- Reconstruction (BCE) + Cosine alignment (optional)
- Supports Lee / Nagano / generic PT dataset
- Saves: best.pt, last.pt, train_log.csv
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

# --- project imports ---
from src.model.gae_film import GAEFiLM, GAEFiLMConfig
from src.model.samplers import MixedNegSamplerConfig, sample_negatives_for_recon, sample_negatives_mixed
from src.model.losses import LossWeights, total_loss


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic option (can be slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_str(x) -> str:
    if isinstance(x, (list, tuple)):
        x = x[0]
    if isinstance(x, torch.Tensor):
        x = x.item()
    return str(x)


def load_target_embeddings(cell_emb_dir: Optional[str]) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[int]]:
    """
    Loads:
      - cell_names.txt (1 column, optionally with header 'cell_name')
      - cell_embeddings.npy
    Returns:
      dict: cell_id -> embedding tensor
      target_dim
    """
    if cell_emb_dir is None:
        return None, None

    cell_emb_dir = str(cell_emb_dir)
    names_path = os.path.join(cell_emb_dir, "cell_names.txt")
    embs_path = os.path.join(cell_emb_dir, "cell_embeddings.npy")

    if (not os.path.exists(names_path)) or (not os.path.exists(embs_path)):
        raise FileNotFoundError(f"Missing {names_path} or {embs_path}")

    names = []
    with open(names_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.lower().startswith("cell_name"):
                continue
            names.append(s)

    embs = np.load(embs_path)
    if len(names) != embs.shape[0]:
        raise ValueError(f"cell_names ({len(names)}) != embeddings rows ({embs.shape[0]})")

    target_dim = int(embs.shape[1])
    d = {nm: torch.tensor(vec, dtype=torch.float32) for nm, vec in zip(names, embs)}
    return d, target_dim


def make_dataset(dataset_name: str, data_dir: str, label_path: Optional[str]):
    """
    Tries to construct dataset with your existing modules:
      dataset/lee.py, dataset/nagano.py, dataset/pt_dataset.py
    """
    dataset_name = dataset_name.lower()
    if dataset_name == "lee":
        # try common class names
        try:
            from src.dataset.lee import LeeDataset  # type: ignore
            return LeeDataset(data_dir=data_dir, label_path=label_path)
        except Exception:
            # fallback: maybe class name differs
            from src.dataset.pt_dataset import PTDataset  # type: ignore
            return PTDataset(data_dir=data_dir, label_path=label_path)
    elif dataset_name == "nagano":
        try:
            from src.dataset.nagano import NaganoDataset  # type: ignore
            return NaganoDataset(data_dir=data_dir, label_path=label_path)
        except Exception:
            from src.dataset.pt_dataset import PTDataset  # type: ignore
            return PTDataset(data_dir=data_dir, label_path=label_path)
    else:
        from src.dataset.pt_dataset import PTDataset  # type: ignore
        return PTDataset(data_dir=data_dir, label_path=label_path)


def get_in_dim_from_dir(train_dir: str) -> int:
    files = sorted([f for f in os.listdir(train_dir) if f.endswith(".pt")])
    if not files:
        raise FileNotFoundError(f"No .pt files found in {train_dir}")
    sample = torch.load(os.path.join(train_dir, files[0]), map_location="cpu")
    if not hasattr(sample, "x"):
        raise AttributeError("Sample Data has no 'x'.")
    return int(sample.x.shape[1])


def save_ckpt(path: str, model: torch.nn.Module, opt: torch.optim.Optimizer, epoch: int,
              best_val: float, cfg: dict) -> None:
    ckpt = {
        "epoch": epoch,
        "best_val": best_val,
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "config": cfg,
    }
    torch.save(ckpt, path)


# -----------------------------
# Training / Validation
# -----------------------------
def run_epoch_pretrain(
    model: GAEFiLM,
    loader: DataLoader,
    device: torch.device,
    neg_cfg: MixedNegSamplerConfig,
    weights: LossWeights,
    cell_emb_dict: Optional[Dict[str, torch.Tensor]],
    cosine_start_epoch: int,
    epoch: int,
    train: bool,
) -> Dict[str, float]:

    model.train(train)

    sums: Dict[str, float] = {
        "recon_pos": 0.0,
        "recon_neg": 0.0,
        "cosine": 0.0,
        "total": 0.0,
    }
    denom = 0

    iterator = tqdm(loader, desc=f"[{'Train' if train else 'Val'} {epoch}]", leave=False)
    for data in iterator:
        data = data.to(device)

        # batch vector (batch_size=1 default)
        batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

        # encode
        z_mod, g = model.encode(data.x, data.edge_index, batch)

        # neg edges (val에서 미리 제공되면 사용)
        if (not train) and hasattr(data, "recon_neg_edge_index"):
            neg_edge = data.recon_neg_edge_index.to(device)
        else:
            neg_edge = sample_negatives_for_recon(data.edge_index, data.num_nodes, neg_cfg, device)

        # cosine target
        target_global = None
        cosine_enabled = False
        if cell_emb_dict is not None and epoch >= cosine_start_epoch and hasattr(data, "cell_id"):
            cid = safe_str(data.cell_id)
            if cid in cell_emb_dict:
                target_global = cell_emb_dict[cid].to(device)
                # g is (B,z_dim). project_global returns (B,target_dim)
                # target_global should match (target_dim,) for B=1; losses handles either, but keep (1,D) safe:
                target_global = target_global.unsqueeze(0)  # (1, D)
                cosine_enabled = True

        out = total_loss(
            model=model,
            z_mod=z_mod,
            g=g,
            pos_edge_index=data.edge_index,
            neg_edge_index=neg_edge,
            weights=weights,
            target_global=target_global,
            cosine_enabled=cosine_enabled,
        )

        loss = out["total"]

        if train:
            # standard step
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
            loss.backward()
            # optional: clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            # optimizer step done outside for speed? keep simple:
            # (we will step via optimizer passed by closure in main)
        # accumulate
        sums["recon_pos"] += float(out["recon_pos"].detach().cpu())
        sums["recon_neg"] += float(out["recon_neg"].detach().cpu())
        sums["cosine"] += float(out["cosine"].detach().cpu())
        sums["total"] += float(out["total"].detach().cpu())
        denom += 1

    if denom == 0:
        return {k: float("nan") for k in sums}
    return {k: v / denom for k, v in sums.items()}


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["lee", "nagano", "pt"], default="lee")

    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--label_path", default=None)

    ap.add_argument("--cell_emb_dir", default=None, help="Directory containing cell_names.txt + cell_embeddings.npy")
    ap.add_argument("--out_dir", required=True)

    # model
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--z_dim", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--recon_dec_hidden", type=int, default=64)
    ap.add_argument("--proj_hidden", type=int, default=64)
    ap.add_argument("--film_hidden", type=int, default=64)

    # training
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cosine_start_epoch", type=int, default=1)

    # loss weights
    ap.add_argument("--recon_weight", type=float, default=1.0)
    ap.add_argument("--cosine_weight", type=float, default=1.0)

    # negative sampling
    ap.add_argument("--neg_pos_ratio", type=float, default=1.0)
    ap.add_argument("--near_lower", type=int, default=5)
    ap.add_argument("--near_upper", type=int, default=100)
    ap.add_argument("--near_ratio", type=float, default=0.7)
    ap.add_argument("--oversample_factor", type=int, default=3)
    ap.add_argument("--max_rounds", type=int, default=4)

    # checkpointing
    ap.add_argument("--save_every", type=int, default=0, help="If >0, save periodic epoch_XXX.pt")
    ap.add_argument("--resume", default=None, help="Path to last.pt to resume")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_csv = out_dir / "train_log.csv"

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # targets
    cell_emb_dict, target_dim = load_target_embeddings(args.cell_emb_dir)

    # infer input dim
    in_dim = get_in_dim_from_dir(args.train_dir)
    if target_dim is None:
        # cosine disabled but keep config consistent
        target_dim = 0

    cfg = GAEFiLMConfig(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        z_dim=args.z_dim,
        dropout=args.dropout,
        recon_dec_hidden=args.recon_dec_hidden,
        target_dim=int(target_dim),
        proj_hidden=args.proj_hidden,
        film_hidden=args.film_hidden,
        enable_loop_head=False,
    )
    model = GAEFiLM(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # dataset / loader
    train_ds = make_dataset(args.dataset, args.train_dir, args.label_path)
    val_ds = make_dataset(args.dataset, args.val_dir, args.label_path)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # sampler config
    neg_cfg = MixedNegSamplerConfig(
        neg_pos_ratio=args.neg_pos_ratio,
        near_lower=args.near_lower,
        near_upper=args.near_upper,
        near_ratio=args.near_ratio,
        oversample_factor=args.oversample_factor,
        max_rounds=args.max_rounds,
    )

    weights = LossWeights(recon=args.recon_weight, cosine=args.cosine_weight, loop=0.0)

    best_val = float("inf")
    start_epoch = 1
    rows = []

    # resume
    if args.resume is not None and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        best_val = float(ckpt.get("best_val", best_val))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[Resume] from {args.resume} (start_epoch={start_epoch}, best_val={best_val:.4f})")

    # save config
    with open(out_dir / "config.txt", "w") as f:
        f.write(str(vars(args)) + "\n")
        f.write(str(asdict(cfg)) + "\n")

    for epoch in range(start_epoch, args.epochs + 1):
        # --- train ---
        model.train(True)
        train_sums = {"recon_pos": 0.0, "recon_neg": 0.0, "cosine": 0.0, "total": 0.0}
        denom_tr = 0

        for data in tqdm(train_loader, desc=f"[Train {epoch}]"):
            data = data.to(device)
            opt.zero_grad(set_to_none=True)

            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
            z_mod, g = model.encode(data.x, data.edge_index, batch)

            neg_edge = sample_negatives_for_recon(data.edge_index, data.num_nodes, neg_cfg, device)

            target_global = None
            cosine_enabled = False
            if cell_emb_dict is not None and epoch >= args.cosine_start_epoch and hasattr(data, "cell_id"):
                cid = safe_str(data.cell_id)
                if cid in cell_emb_dict:
                    target_global = cell_emb_dict[cid].to(device).unsqueeze(0)
                    cosine_enabled = True

            out = total_loss(
                model=model,
                z_mod=z_mod,
                g=g,
                pos_edge_index=data.edge_index,
                neg_edge_index=neg_edge,
                weights=weights,
                target_global=target_global,
                cosine_enabled=cosine_enabled,
            )

            loss = out["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            for k in train_sums:
                train_sums[k] += float(out[k].detach().cpu())
            denom_tr += 1

        train_avg = {k: train_sums[k] / max(1, denom_tr) for k in train_sums}

        # --- val ---
        model.eval()
        val_sums = {"recon_pos": 0.0, "recon_neg": 0.0, "cosine": 0.0, "total": 0.0}
        denom_va = 0

        with torch.no_grad():
            for data in tqdm(val_loader, desc=f"[Val {epoch}]"):
                data = data.to(device)
                if hasattr(data, "recon_neg_edge_index"):
                    neg_edge = data.recon_neg_edge_index.to(device)
                else:
                    neg_edge = sample_negatives_for_recon(data.edge_index, data.num_nodes, neg_cfg, device)

                batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
                z_mod, g = model.encode(data.x, data.edge_index, batch)

                target_global = None
                cosine_enabled = False
                if cell_emb_dict is not None and epoch >= args.cosine_start_epoch and hasattr(data, "cell_id"):
                    cid = safe_str(data.cell_id)
                    if cid in cell_emb_dict:
                        target_global = cell_emb_dict[cid].to(device).unsqueeze(0)
                        cosine_enabled = True

                out = total_loss(
                    model=model,
                    z_mod=z_mod,
                    g=g,
                    pos_edge_index=data.edge_index,
                    neg_edge_index=neg_edge,
                    weights=weights,
                    target_global=target_global,
                    cosine_enabled=cosine_enabled,
                )

                for k in val_sums:
                    val_sums[k] += float(out[k].detach().cpu())
                denom_va += 1

        val_avg = {k: val_sums[k] / max(1, denom_va) for k in val_sums}

        # log
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_avg.items()},
            **{f"val_{k}": v for k, v in val_avg.items()},
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(log_csv, index=False)

        # ckpt
        save_ckpt(str(out_dir / "last.pt"), model, opt, epoch, best_val, vars(args))

        if args.save_every and (epoch % args.save_every == 0):
            save_ckpt(str(out_dir / f"epoch_{epoch:03d}.pt"), model, opt, epoch, best_val, vars(args))

        if val_avg["total"] < best_val:
            best_val = val_avg["total"]
            save_ckpt(str(out_dir / "best.pt"), model, opt, epoch, best_val, vars(args))
            print(f"🌟 Best updated: epoch={epoch} val_total={best_val:.6f}")

        print(
            f"[Epoch {epoch}] "
            f"train_total={train_avg['total']:.4f} val_total={val_avg['total']:.4f} "
            f"(recon={val_avg['recon_pos']+val_avg['recon_neg']:.4f}, cos={val_avg['cosine']:.4f})"
        )

    print(f"Done. Best val_total={best_val:.6f}  (saved: {out_dir/'best.pt'})")


if __name__ == "__main__":
    main()

