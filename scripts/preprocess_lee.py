#!/usr/bin/env python3
import os
import sys
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.preprocessing.run_lee import run  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Preprocess Lee scHi-C (mcool -> npz)")
    p.add_argument("--input_dir", required=True, help="Directory containing .mcool files")
    p.add_argument("--output_dir", required=True, help="Output directory for .npz files")

    # unified name: resolution (bp)
    p.add_argument("--resolution", type=int, required=True, help="Resolution in bp (e.g., 100000)")

    # unified QC option
    p.add_argument("--min_total", type=int, default=2000,
                   help="Minimum total contact count per cell (2000 disables filtering)")

    p.add_argument("--exclude_chroms", nargs="+", default=["chrX", "chrY", "chrM"],
                   help="Chromosomes to exclude (default: chrX chrY chrM)")
    args = p.parse_args()

    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        resolution=args.resolution,
        exclude_chroms=args.exclude_chroms,
        min_total=args.min_total,
    )


if __name__ == "__main__":
    main()

