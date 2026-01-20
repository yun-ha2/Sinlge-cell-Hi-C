#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TAD/TLD boundary extraction from pre-extracted z embeddings (.z.npz).

You should have already run:
  scripts/inference/extract_z.py

which produced:
  <Z_DIR>/*.z.npz

Now run:
  python scripts/downstream/tad_from_z.py \
    --z_dir <Z_DIR> \
    --out_dir <OUT_DIR> \
    --binsize 10000 \
    --avg_tld_size_bins 20 \
    --consensus_vote 0.3 \
    --consensus_min_sep_bins 3 \
    --ctcf_bed /path/to/CTCF_motifs.bed \
    --ctcf_window_bp 20000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.downstream.tads.tad import run_tad_from_z_npz, TADFromZConfig


def build_parser():
    p = argparse.ArgumentParser("TAD/TLD boundary extraction from z (.npz)")

    p.add_argument("--z_dir", required=True, help="directory containing *.z.npz")
    p.add_argument("--out_dir", required=True)

    # core params
    p.add_argument("--binsize", type=int, default=10_000)
    p.add_argument("--avg_tld_size_bins", type=int, default=20)
    p.add_argument("--linkage", type=str, default="ward")

    # PCA option
    p.add_argument("--use_pca", action="store_true")
    p.add_argument("--pca_ncomp", type=int, default=16)

    # CTCF refinement (optional)
    p.add_argument("--ctcf_bed", type=str, default=None)
    p.add_argument("--ctcf_window_bp", type=int, default=20_000)

    # consensus
    p.add_argument("--consensus_vote", type=float, default=0.3)
    p.add_argument("--consensus_min_sep_bins", type=int, default=3)

    # outputs
    p.add_argument("--no_vote_bedgraph", action="store_true")
    p.add_argument("--no_per_cell_boundaries", action="store_true")

    return p


def main():
    args = build_parser().parse_args()

    cfg = TADFromZConfig(
        binsize=args.binsize,
        avg_tld_size_bins=args.avg_tld_size_bins,
        linkage=args.linkage,
        use_pca=bool(args.use_pca),
        pca_ncomp=args.pca_ncomp,
        ctcf_bed=args.ctcf_bed,
        ctcf_window_bp=args.ctcf_window_bp,
        consensus_vote=args.consensus_vote,
        consensus_min_sep_bins=args.consensus_min_sep_bins,
        save_vote_bedgraph=not bool(args.no_vote_bedgraph),
        save_per_cell_boundaries=not bool(args.no_per_cell_boundaries),
    )

    z_dir = Path(args.z_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = run_tad_from_z_npz(z_dir=z_dir, out_dir=out_dir, cfg=cfg)

    print("\n TAD/TLD extraction done.")
    print(f"- z files: {len(meta)}")
    print(f"- outputs: {out_dir}")
    print(f"  - per-cell: {out_dir / 'cells'}")
    print(f"  - consensus: {out_dir / 'consensus'}")
    print(f"  - metadata: {out_dir / 'metadata.tsv'}")


if __name__ == "__main__":
    main()

