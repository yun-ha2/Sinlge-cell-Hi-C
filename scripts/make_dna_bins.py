#!/usr/bin/env python3
"""
CLI wrapper for DNA bin sequence extraction.

Example
-------
hg19 (500kb):
  python scripts/make_dna_bins.py \
    --fasta /path/to/hg19.fa \
    --out /path/to/hg19_bins_500kb.json \
    --bin_size 500000 \
    --chrom_style hg19

mm9 (10kb):
  python scripts/make_dna_bins.py \
    --fasta /path/to/mm9.fa \
    --out /path/to/mm9_bins_10kb.json \
    --bin_size 10000 \
    --chrom_style mm9
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from src.preprocessing.dna_bins import BinSpec, run_make_bins


# ============================================================
# Chromosome presets
# ============================================================

def preset_chromosomes(style: str, chroms: Optional[str]) -> List[str]:
    """
    Get chromosome list from a preset style or user-provided list.

    Parameters
    ----------
    style : {"hg19", "mm9", "custom"}
        Preset name or custom.
    chroms : str or None
        Comma-separated chromosome list when style="custom".

    Returns
    -------
    list[str]
        Chromosome names.
    """
    if style == "hg19":
        # autosomes only; add chrX/chrY if needed downstream
        return [str(i) for i in range(1, 23)]

    if style == "mm9":
        return [f"chr{i}" for i in range(1, 20)]

    if not chroms:
        raise ValueError("--chroms is required when --chrom_style=custom")

    return [c.strip() for c in chroms.split(",") if c.strip()]


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract fixed-size DNA bins from FASTA and save as JSON.")
    p.add_argument("--fasta", required=True, help="Path to reference FASTA.")
    p.add_argument("--out", required=True, help="Output JSON path.")
    p.add_argument("--bin_size", required=True, type=int, help="Bin size in bp.")
    p.add_argument(
        "--chrom_style",
        choices=["hg19", "mm9", "custom"],
        default="custom",
        help="Chromosome preset (hg19/mm9) or custom list via --chroms.",
    )
    p.add_argument(
        "--chroms",
        default=None,
        help="Comma-separated chromosome list for custom mode (e.g., 'chr1,chr2,chrX' or '1,2,...,22').",
    )

    # module defaults: allow_alias=True, strict=False, include_intervals=False
    p.add_argument(
        "--no_alias",
        action="store_true",
        help="Disable alias mapping between 'chr1' and '1'.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Raise error if any chromosome is missing in FASTA (default: skip).",
    )
    p.add_argument(
        "--include_intervals",
        action="store_true",
        help="Include start/end coordinates in JSON records.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    chromosomes = preset_chromosomes(args.chrom_style, args.chroms)

    spec = BinSpec(
        fasta_path=Path(args.fasta),
        output_json_path=Path(args.out),
        bin_size=args.bin_size,
        chromosomes=chromosomes,
    )

    run_make_bins(
        spec,
        allow_alias=(not args.no_alias),
        strict=args.strict,
        include_intervals=args.include_intervals,
    )


if __name__ == "__main__":
    main()

