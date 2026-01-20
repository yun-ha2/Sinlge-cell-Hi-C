#!/usr/bin/env python3
import os
import sys
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.preprocessing.run_nagano import run  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Preprocess Nagano scHi-C (schic2 adj -> npz)")
    p.add_argument("--input_dir", required=True, help="Base directory containing per-cell schic2 outputs")
    p.add_argument("--output_dir", required=True, help="Output directory for .npz files")

    # unified name: resolution (bp)
    p.add_argument("--resolution", type=int, default=100_000,
                   help="Resolution (bin size) in bp (default: 100000)")

    # unified QC option
    p.add_argument("--min_total", type=int, default=2000,
                   help="Minimum total contact count per cell (default: 2000)")

    p.add_argument("--fends", required=True, help="Path to GATC.fends file")
    p.add_argument("--chrom_sizes", required=True, help="Path to chrom.sizes file")
    args = p.parse_args()

    run(
        input_base=args.input_dir,          
        output_dir=args.output_dir,
        fends_path=args.fends,
        chrom_size_path=args.chrom_sizes,
        bin_size=args.resolution,           
        min_total=args.min_total,
    )


if __name__ == "__main__":
    main()

