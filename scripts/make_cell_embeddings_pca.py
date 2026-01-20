from __future__ import annotations

import argparse
from pathlib import Path

from src.tasks.embedding.pca_cells import PCACellEmbeddingConfig, build_pca_cell_embeddings


def main():
    p = argparse.ArgumentParser("PCA cell embedding builder (scHiCluster-like)")
    p.add_argument("--contact_dir", required=True, help="Directory with per-cell contact .npz")
    p.add_argument("--out_dir", required=True, help="Output directory")

    p.add_argument("--chromosomes", nargs="+", required=True, help="Chrom list, e.g. chr1 chr2 ...")
    p.add_argument("--dim_per_chr", type=int, default=50)
    p.add_argument("--dim_global", type=int, default=50)
    p.add_argument("--random_state", type=int, default=0)

    # normalization options
    p.add_argument("--normalize", action="store_true", help="Apply scHiCluster-like normalization")
    p.add_argument("--no_coverage_balance", action="store_true")
    p.add_argument("--keep_diagonal", action="store_true")
    p.add_argument("--no_sqrtvc", action="store_true")

    # filtering
    p.add_argument("--allow_missing_chroms", action="store_true")
    p.add_argument("--keep_empty_chrom", action="store_true")

    args = p.parse_args()

    cfg = PCACellEmbeddingConfig(
        chromosomes=args.chromosomes,
        dim_per_chr=args.dim_per_chr,
        dim_global=args.dim_global,
        random_state=args.random_state,
        apply_normalization=args.normalize,
        coverage_balance=not args.no_coverage_balance,
        remove_diagonal=not args.keep_diagonal,
        sqrtvc_norm=not args.no_sqrtvc,
        require_all_chroms=not args.allow_missing_chroms,
        skip_empty_chrom=not args.keep_empty_chrom,
    )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    Z, names = build_pca_cell_embeddings(
        contact_dir=args.contact_dir,
        out_dir=args.out_dir,
        config=cfg,
    )

    print(f"[DONE] saved embeddings: {Z.shape} to {args.out_dir}")


if __name__ == "__main__":
    main()

