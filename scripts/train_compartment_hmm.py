#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib

import glob

from src.downstream.compartment_gmmhmm import (
    CompartmentHMMConfig,
    collect_train_blocks,
    train_gmmhmm,
)


def build_parser():
    p = argparse.ArgumentParser("Train 2-state GMMHMM for compartment calling from z embeddings")
    p.add_argument("--z_dir", required=True, help="directory containing z npz files")
    p.add_argument("--out_dir", required=True, help="where to save hmm/scaler/pca")

    p.add_argument("--hmm_iter", type=int, default=100)
    p.add_argument("--train_max_seqs", type=int, default=12000)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--standardize", action="store_true")
    p.add_argument("--pca_dim", type=int, default=8)  # 0 -> disable

    return p


def main():
    args = build_parser().parse_args()
    z_dir = Path(args.z_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    z_files = sorted(glob.glob(str(z_dir / "*.npz")))
    if len(z_files) == 0:
        raise RuntimeError(f"No npz under: {z_dir}")

    pca_dim = args.pca_dim if args.pca_dim and args.pca_dim > 0 else None

    X_train, lengths, scaler, pca = collect_train_blocks(
        z_files=z_files,
        max_seqs=args.train_max_seqs,
        standardize=bool(args.standardize),
        pca_dim=pca_dim,
        random_state=args.seed,
    )

    hmm = train_gmmhmm(
        X_train=X_train,
        lengths=lengths,
        n_states=2,
        n_iter=args.hmm_iter,
        random_state=args.seed,
    )

    # save
    joblib.dump(hmm, out_dir / "gmmhmm.pkl")
    if scaler is not None:
        joblib.dump(scaler, out_dir / "scaler.pkl")
    if pca is not None:
        joblib.dump(pca, out_dir / "pca.pkl")

    meta = {
        "z_dir": str(z_dir),
        "n_files": len(z_files),
        "train_max_seqs": args.train_max_seqs,
        "hmm_iter": args.hmm_iter,
        "seed": args.seed,
        "standardize": bool(args.standardize),
        "pca_dim": pca_dim,
    }
    with open(out_dir / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("✅ HMM trained & saved to:", out_dir)
    print("- gmmhmm.pkl")
    if scaler is not None:
        print("- scaler.pkl")
    if pca is not None:
        print("- pca.pkl")


if __name__ == "__main__":
    main()

