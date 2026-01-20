#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from src.dataset.lee import LeeDataset
from src.dataset.nagano import NaganoDataset
from src.dataset.pt_dataset import PTGraphDataset

from src.model.encoder_gcn import GCNEncoder
from src.model.gae_film import GAEFiLM
from src.model.losses import recon_bce_loss, cosine_align_loss, loop_bce_loss
from src.model.samplers import sample_negatives_mixed
from src.model.ref_loops import build_ref_loop_provider, SamplerArgs


# ==============================================================
# small utils
# ==============================================================

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def read_cell_emb(cell_emb_dir: str) -> Dict[str, torch.Tensor]:
    """
    Reads:
      - cell_names*.txt / .csv / .tsv
      - cell_embeddings*.npy
    Returns dict: {cell_name: embedding_tensor}
    """
    cell_emb_dir = str(cell_emb_dir)

    name_candidates = [
        "cell_names.txt",
        "cell_names_all.txt",
        "cell_names.tsv",
        "cell_names.csv",
    ]
    emb_candidates = [
        "cell_embeddings.npy",
        "cell_embeddings_all.npy",
    ]

    name_path = None
    for fn in name_candidates:
        p = os.path.join(cell_emb_dir, fn)
        if os.path.exists(p):
            name_path = p
            break
    if name_path is None:
        raise FileNotFoundError(f"[cell_emb] missing names file under: {cell_emb_dir}")

    emb_path = None
    for fn in emb_candidates:
        p = os.path.join(cell_emb_dir, fn)
        if os.path.exists(p):
            emb_path = p
            break
    if emb_path is None:
        raise FileNotFoundError(f"[cell_emb] missing embeddings file under: {cell_emb_dir}")

    # names
    if name_path.endswith(".csv"):
        df = pd.read_csv(name_path)
        col = "cell_name" if "cell_name" in df.columns else df.columns[0]
        names = df[col].astype(str).tolist()
    elif name_path.endswith(".tsv"):
        df = pd.read_csv(name_path, sep="\t")
        col = "cell_name" if "cell_name" in df.columns else df.columns[0]
        names = df[col].astype(str).tolist()
    else:
        names = []
        with open(name_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.lower().startswith("cell_name"):
                    continue
                names.append(s)

    embs = np.load(emb_path)
    if len(names) != embs.shape[0]:
        raise ValueError(f"[cell_emb] names({len(names)}) != embs({embs.shape[0]})")

    return {n: torch.tensor(v, dtype=torch.float32) for n, v in zip(names, embs)}


def load_state(model: nn.Module, path: str, device: torch.device, strict: bool = True):
    obj = torch.load(path, map_location=device)

    if isinstance(obj, dict):
        # checkpoint style
        if "model_state" in obj:
            model.load_state_dict(obj["model_state"], strict=strict)
            return
        if "state_dict" in obj:
            model.load_state_dict(obj["state_dict"], strict=strict)
            return
        # pure state_dict
        if all(torch.is_tensor(v) for v in obj.values()):
            model.load_state_dict(obj, strict=strict)
            return

    raise ValueError(f"[load_state] unsupported format: {path}")


def get_dataset(name: str, data_dir: str, label_path: Optional[str]):
    name = name.lower()
    if name == "lee":
        return LeeDataset(data_dir=data_dir, label_path=label_path)
    if name == "nagano":
        return NaganoDataset(data_dir=data_dir, label_path=label_path)
    return PTGraphDataset(split_dir=data_dir, label_path=label_path)


# ==============================================================
# main
# ==============================================================

def main():
    p = argparse.ArgumentParser()

    # dataset
    p.add_argument("--dataset", required=True, choices=["lee", "nagano"])
    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir", required=True)
    p.add_argument("--label_path", required=True)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)

    # pretrained & save
    p.add_argument("--pretrain_model", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)

    # ref loop
    p.add_argument("--ref_loop_path", required=True)
    p.add_argument("--val_neg_field", default="ref_neg_edge_index")
    p.add_argument("--neuron_types", nargs="*", default=["Sst","L23","L4","L5","L6","Vip","Pvalb","Ndnf"])

    # cell embedding alignment
    p.add_argument("--cell_emb_dir", required=True)
    p.add_argument("--cosine_start_epoch", type=int, default=1)

    # model dims
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--z_dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--proj_hidden", type=int, default=64)

    # loop head dims
    p.add_argument("--loop_dim", type=int, default=32)
    p.add_argument("--loop_proj_hidden", type=int, default=64)
    p.add_argument("--loop_dec_hidden", type=int, default=64)

    # loss weights
    p.add_argument("--w_recon", type=float, default=0.2)
    p.add_argument("--w_cos", type=float, default=0.6)
    p.add_argument("--w_loop", type=float, default=1.5)

    # negative sampling
    p.add_argument("--neg_pos_ratio", type=float, default=1.0)
    p.add_argument("--near_lower", type=int, default=5)
    p.add_argument("--near_upper", type=int, default=100)
    p.add_argument("--near_ratio", type=float, default=0.7)
    p.add_argument("--oversample_factor", type=int, default=3)
    p.add_argument("--max_rounds", type=int, default=4)

    # freeze options
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--freeze_film", action="store_true")
    p.add_argument("--strict_pretrain", action="store_true")

    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    mkdir(args.out_dir)
    log_csv = os.path.join(args.out_dir, "train_log_loop.csv")

    # ----------------------------------------------------------
    # data
    # ----------------------------------------------------------
    train_ds = get_dataset(args.dataset, args.train_dir, args.label_path)
    val_ds = get_dataset(args.dataset, args.val_dir, args.label_path)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # infer input dim
    sample_pt = torch.load(os.path.join(args.train_dir, train_ds.file_list[0]))
    in_dim = int(sample_pt.x.shape[1])

    # cell embedding targets
    cell_emb = read_cell_emb(args.cell_emb_dir)
    target_dim = int(next(iter(cell_emb.values())).numel())

    # ----------------------------------------------------------
    # model (pretrain) → load → loop head attach
    # ----------------------------------------------------------
    enc = GCNEncoder(in_dim=in_dim, hidden_dim=args.hidden_dim, z_dim=args.z_dim, dropout=args.dropout)
    model = GAEFiLM(encoder=enc, z_dim=args.z_dim, proj_hidden=args.proj_hidden, target_dim=target_dim).to(device)

    load_state(model, args.pretrain_model, device=device, strict=args.strict_pretrain)
    print(f"pretrained loaded: {args.pretrain_model}")

    model.add_loop_head(loop_dim=args.loop_dim, proj_hidden=args.loop_proj_hidden, dec_hidden=args.loop_dec_hidden)
    model = model.to(device)

    if args.freeze_encoder:
        for p_ in model.encoder.parameters():
            p_.requires_grad = False
        print("encoder frozen")

    if args.freeze_film:
        for p_ in model.attn.parameters():
            p_.requires_grad = False
        for p_ in model.film.parameters():
            p_.requires_grad = False
        print("FiLM/pooling frozen")

    opt = torch.optim.Adam([p_ for p_ in model.parameters() if p_.requires_grad], lr=args.lr)

    # ----------------------------------------------------------
    # ref loop provider
    # ----------------------------------------------------------
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

    bce = nn.BCEWithLogitsLoss()

    # ----------------------------------------------------------
    # logger
    # ----------------------------------------------------------
    cols = [
        "epoch",
        "train_pos","train_neg","train_cos","train_loop_pos","train_loop_neg",
        "val_pos","val_neg","val_cos","val_loop_pos","val_loop_neg",
        "val_total",
        "loop_cov_train","loop_cov_val"
    ]
    log_df = pd.DataFrame(columns=cols)

    best_val = float("inf")

    # ==========================================================
    # training
    # ==========================================================
    for epoch in range(1, args.epochs + 1):

        # ------------------ TRAIN ------------------
        model.train()
        t_pos = t_neg = t_cos = 0.0
        t_lpos = t_lneg = 0.0
        used = total = 0

        for data in tqdm(train_loader, desc=f"[Train {epoch}]"):
            data = data.to(device)
            opt.zero_grad()

            z_mod, g = model.encode(data.x, data.edge_index, batch=getattr(data, "batch", None))

            # recon neg sampling
            num_pos = int(data.edge_index.size(1))
            num_neg = int(num_pos * args.neg_pos_ratio)

            neg_edge = sample_negatives_mixed(
                pos_edge_index=data.edge_index,
                num_nodes=int(data.num_nodes),
                num_neg=num_neg,
                device=device,
                near_lower=args.near_lower,
                near_upper=args.near_upper,
                near_ratio=args.near_ratio,
                oversample_factor=args.oversample_factor,
                max_rounds=args.max_rounds,
            )

            recon_l, pos_l, neg_l = recon_bce_loss(z_mod, data.edge_index, neg_edge, model.recon_dec, bce=bce)

            # cosine loss
            cid = getattr(data, "cell_id", "")
            if isinstance(cid, (list, tuple)):
                cid = cid[0]
            cid = str(cid)

            if epoch >= args.cosine_start_epoch and cid in cell_emb:
                cos_l = cosine_align_loss(model.project(g), cell_emb[cid].to(device))
            else:
                cos_l = torch.tensor(0.0, device=device)

            # loop loss
            pos_loop = provider.get_pos(data)
            total += 1
            if pos_loop.size(1) > 0:
                used += 1
            neg_loop = provider.get_neg(data, pos_loop, sample_negatives_mixed, sargs, split="train")

            loop_l, lp, ln = loop_bce_loss(model, z_mod, pos_loop, neg_loop, bce=bce)

            loss = args.w_recon * recon_l + args.w_cos * cos_l + args.w_loop * loop_l
            loss.backward()
            opt.step()

            t_pos += float(pos_l)
            t_neg += float(neg_l)
            t_cos += float(cos_l)
            t_lpos += float(lp)
            t_lneg += float(ln)

        ntr = max(1, len(train_loader))
        cov_tr = used / max(1, total)

        # ------------------ VAL ------------------
        model.eval()
        v_pos = v_neg = v_cos = 0.0
        v_lpos = v_lneg = 0.0
        v_used = v_total = 0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)

                if not hasattr(data, "recon_neg_edge_index"):
                    continue

                z_mod, g = model.encode(data.x, data.edge_index, batch=getattr(data, "batch", None))

                neg_edge = data.recon_neg_edge_index.to(device)
                recon_l, pos_l, neg_l = recon_bce_loss(z_mod, data.edge_index, neg_edge, model.recon_dec, bce=bce)

                cid = getattr(data, "cell_id", "")
                if isinstance(cid, (list, tuple)):
                    cid = cid[0]
                cid = str(cid)

                if epoch >= args.cosine_start_epoch and cid in cell_emb:
                    cos_l = cosine_align_loss(model.project(g), cell_emb[cid].to(device))
                else:
                    cos_l = torch.tensor(0.0, device=device)

                pos_loop = provider.get_pos(data)
                v_total += 1
                if pos_loop.size(1) > 0:
                    v_used += 1
                neg_loop = provider.get_neg(data, pos_loop, sample_negatives_mixed, sargs, split="val")

                loop_l, lp, ln = loop_bce_loss(model, z_mod, pos_loop, neg_loop, bce=bce)

                v_pos += float(pos_l)
                v_neg += float(neg_l)
                v_cos += float(cos_l)
                v_lpos += float(lp)
                v_lneg += float(ln)

        nval = max(1, len(val_loader))
        cov_val = v_used / max(1, v_total)

        val_total = (
            args.w_recon * (v_pos + v_neg) / nval
            + args.w_cos * (v_cos / nval)
            + args.w_loop * (v_lpos + v_lneg) / nval
        )

        print(
            f"[Epoch {epoch}] "
            f"TR pos={t_pos/ntr:.4f} neg={t_neg/ntr:.4f} cos={t_cos/ntr:.4f} "
            f"lp={t_lpos/ntr:.4f} ln={t_lneg/ntr:.4f} cov={cov_tr:.2%} | "
            f"VA total={val_total:.4f} pos={v_pos/nval:.4f} neg={v_neg/nval:.4f} cos={v_cos/nval:.4f} "
            f"lp={v_lpos/nval:.4f} ln={v_lneg/nval:.4f} cov={cov_val:.2%}"
        )

        # log
        row = dict(
            epoch=epoch,
            train_pos=t_pos/ntr,
            train_neg=t_neg/ntr,
            train_cos=t_cos/ntr,
            train_loop_pos=t_lpos/ntr,
            train_loop_neg=t_lneg/ntr,
            val_pos=v_pos/nval,
            val_neg=v_neg/nval,
            val_cos=v_cos/nval,
            val_loop_pos=v_lpos/nval,
            val_loop_neg=v_lneg/nval,
            val_total=val_total,
            loop_cov_train=cov_tr,
            loop_cov_val=cov_val,
        )
        log_df.loc[len(log_df)] = row
        log_df.to_csv(log_csv, index=False)

        # save
        torch.save(model.state_dict(), os.path.join(args.out_dir, f"epoch_{epoch:03d}.pt"))
        torch.save(model.state_dict(), os.path.join(args.out_dir, "last_model.pt"))

        if val_total < best_val:
            best_val = val_total
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))
            print(" Best updated (epoch {epoch}) val={best_val:.4f}")

    print(f"\nFinished. best val = {best_val:.4f}")


if __name__ == "__main__":
    main()

