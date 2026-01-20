#!/usr/bin/env python3
"""
CLI for merging LDP + DNA embeddings into final node features.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from src.preprocessing.merge_ldp_dna import MergeLDPSpec, run_merge_ldp_dna


def preset_chromosomes(style: str, chroms: Optional[str]) -> List[str]:
    if style == "hg19":
        return [f"chr{i}" for i in range(1, 23)]
    if style == "mm9":
        return [f"chr{i}" for i in range(1, 20)]
    if not chroms:
        raise ValueError("--chroms is required when --chrom_style=custom")
    return [c.strip() for c in chroms.split(",") if c.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge per-cell LDP with shared DNA embeddings (merge-only).")
    p.add_argument("--ldp_dir", required=True, help="Directory with per-cell LDP files (*.npz).")
    p.add_argument("--dna_npz", required=True, help="Merged DNA embedding .npz (keys: chr*).")
    p.add_argument("--out_dir", required=True, help="Output directory for merged node features (*.npz).")

    p.add_argument("--chrom_style", choices=["hg19", "mm9", "custom"], default="custom")
    p.add_argument("--chroms", default=None, help="Comma-separated chroms when chrom_style=custom.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    chromosomes = preset_chromosomes(args.chrom_style, args.chroms)

    spec = MergeLDPSpec(
        ldp_dir=Path(args.ldp_dir),
        dna_npz=Path(args.dna_npz),
        out_dir=Path(args.out_dir),
        chromosomes=chromosomes,
    )

    run_merge_ldp_dna(spec)


if __name__ == "__main__":
    main()

