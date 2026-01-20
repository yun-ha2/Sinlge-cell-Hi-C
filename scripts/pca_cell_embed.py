#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================
#   - input: contact_dir/<cell_id>.npz (chr matrices)
#   - output: cell_embeddings.npy, cell_names.txt, metadata.tsv, config.json
# ==============================================================

from __future__ import annotations

import argparse
from pathlib import Path

from src.embedding.pca_cell_embed import (
    PCACellEmbeddingConfig,
    build_pca_cell_embeddings,
)


def parse_chroms(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--contact_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--chromosomes", required=True, help="comma-separated, e.g. chr1,chr2,...")

    # dims
    p.add_argument("--dim_per_chr", type=int, default=50)
    p.add_argument("--dim_global", type=int, default=50)
    p.add_argument("--random_state", type=int, default=0)

    # normalization flags
    p.add_argument("--apply_normalization", action="store_true")
    p.add_argument("--no_coverage_balance", action="store_true")
    p.add_argument("--no_remove_diagonal", action="store_true")
    p.add_argument("--no_sqrtvc_norm", action="store_true")

    # filtering
    p.add_argument("--require_all_chroms", action="store_true")
    p.add_argument("--skip_empty_chrom", action="store_true")

    args = p.parse_args()

    chroms = parse_chroms(args.chromosomes)

    cfg = PCACellEmbeddingConfig(
        chromosomes=chroms,
        dim_per_chr=args.dim_per_chr,
        dim_global=args.dim_global,
        random_state=args.random_state,
        apply_normalization=args.apply_normalization,
        coverage_balance=(not args.no_coverage_balance),
        remove_diagonal=(not args.no_remove_diagonal),
        sqrtvc_norm=(not args.no_sqrtvc_norm),
        require_all_chroms=args.require_all_chroms,
        skip_empty_chrom=args.skip_empty_chrom,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    build_pca_cell_embeddings(
        contact_dir=args.contact_dir,
        out_dir=args.out_dir,
        config=cfg,
        cell_ids=None,
    )

    print("✅ Saved embeddings to:", args.out_dir)


if __name__ == "__main__":
    main()

