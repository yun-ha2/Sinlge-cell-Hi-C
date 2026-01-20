#!/usr/bin/env python3
"""
CLI for Local Degree Profile (LDP) feature extraction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.preprocessing.ldp import LDPSpec, run_ldp_directory


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute chromosome-wise LDP features from per-cell sparse matrices (.npz).")
    p.add_argument("--input_dir", required=True, help="Directory containing per-cell .npz sparse matrices.")
    p.add_argument("--output_dir", required=True, help="Output directory for per-cell LDP features (.npz).")
    p.add_argument("--max_workers", type=int, default=8, help="Number of processes.")
    p.add_argument(
        "--exclude_chroms",
        default="chrX,chrY,chrM",
        help="Comma-separated chromosomes to skip (default: chrX,chrY,chrM).",
    )
    p.add_argument(
        "--no_pickle",
        action="store_true",
        help="Disable allow_pickle when loading npz (use only if your npz stores raw arrays, not objects).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    exclude = tuple(c.strip() for c in args.exclude_chroms.split(",") if c.strip())

    spec = LDPSpec(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        max_workers=args.max_workers,
        exclude_chroms=exclude,
        allow_pickle=(not args.no_pickle),
    )

    run_ldp_directory(spec)


if __name__ == "__main__":
    main()

