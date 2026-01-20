#!/usr/bin/env python3
"""
CLI for HyenaDNA embedding extraction from DNA bin JSON (merge-only).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from src.preprocessing.dna_embed_hyenadna import HyenaDNASpec, run_hyenadna_embedding


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute merged DNA embeddings using HyenaDNA from bin-sequence JSON.")
    p.add_argument("--input_json", required=True, help="Input JSON from dna_bins.py.")
    p.add_argument("--output_npz", required=True, help="Output merged .npz path (single file).")

    p.add_argument("--checkpoint", default="LongSafari/hyenadna-large-1m-seqlen-hf", help="Hugging Face checkpoint.")
    p.add_argument("--device", default=None, help='Torch device (e.g., "cuda:0", "cuda:1", "cpu").')
    p.add_argument("--pool", choices=["first", "last"], default="last", help="Token position to use for embedding.")

    p.add_argument("--include_chroms", default=None, help="Comma-separated chromosomes to include (e.g., chr1,chr2).")
    p.add_argument("--exclude_chroms", default="chrX,chrY,chrM", help="Comma-separated chromosomes to exclude.")
    p.add_argument("--clear_cache_every", type=int, default=64, help="Call empty_cache() every N sequences on CUDA.")

    return p


def main() -> None:
    args = build_parser().parse_args()

    device = args.device
    if device is None:
        device = HyenaDNASpec(input_json_path="x", output_npz_path="x").device

    include_chroms: Optional[List[str]] = None
    if args.include_chroms:
        include_chroms = [c.strip() for c in args.include_chroms.split(",") if c.strip()]

    exclude_chroms = tuple(c.strip() for c in args.exclude_chroms.split(",") if c.strip())

    spec = HyenaDNASpec(
        input_json_path=Path(args.input_json),
        output_npz_path=Path(args.output_npz),
        checkpoint=args.checkpoint,
        device=device,
        pool=args.pool,
        include_chroms=include_chroms,
        exclude_chroms=exclude_chroms,
        clear_cache_every=args.clear_cache_every,
    )

    run_hyenadna_embedding(spec)


if __name__ == "__main__":
    main()

