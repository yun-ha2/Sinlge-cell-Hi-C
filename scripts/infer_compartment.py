#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import glob
import joblib

from src.downstream.compartment_gmmhmm import (
    CompartmentHMMConfig,
    write_per_cell_tables_and_consensus,
)


def build_parser():
    p = argparse.ArgumentParser("Infer compartment score tables from z embeddings with trained GMMHMM")

    p.add_argument("--z_dir", required=True)
    p.add_argument("--model_dir", required=True, help="directory containing gmmhmm.pkl (+ scaler/pca)")
    p.add_argument("--out_dir", required=True)

    p.add_argument("--chrom_sizes", required=True, help="chrom.sizes path")
    p.add_argument("--gc_table", required=True, help="merged GC table (parquet or tsv/csv)")

    p.add_argument("--resolution", type=int, default=10_000)
    p.add_argument("--coarsen_factor", type=int, default=10)
    p.add_argument("--float_format", type=str, default="%.5f")

    return p


def main():
    args = build_parser().parse_args()

    z_files = sorted(glob.glob(str(Path(args.z_dir) / "*.npz")))
    if len(z_files) == 0:
        raise RuntimeError(f"No npz under: {args.z_dir}")

    model_dir = Path(args.model_dir)
    hmm = joblib.load(model_dir / "gmmhmm.pkl")
    scaler = joblib.load(model_dir / "scaler.pkl") if (model_dir / "scaler.pkl").exists() else None
    pca = joblib.load(model_dir / "pca.pkl") if (model_dir / "pca.pkl").exists() else None

    cfg = CompartmentHMMConfig(
        resolution=args.resolution,
        coarsen_factor=args.coarsen_factor,
        hmm_n_states=2,
        chrom_sizes_path=args.chrom_sizes,
        gc_table_path=args.gc_table,
        float_format=args.float_format,
        # standardize/pca_dim are informational here (already baked in scaler/pca)
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells, consensus_path = write_per_cell_tables_and_consensus(
        z_files=z_files,
        hmm=hmm,
        scaler=scaler,
        pca=pca,
        cfg=cfg,
        out_dir=out_dir,
    )

    print("\n✅ Compartment inference done.")
    print(f"- cells written: {len(cells)}")
    print(f"- per-cell dir: {out_dir / 'per_cell'}")
    print(f"- consensus: {consensus_path}")


if __name__ == "__main__":
    main()

