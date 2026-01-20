from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedName:
    cell_id: str
    chrom: str  # "chr1", ...


def parse_pt_name_default(filename: str) -> ParsedName:
    """
    Parse "<cell_id>_chr<k>.pt" → (cell_id, chrom).

    Example
    -------
    "C2_AD008_L5_chr1.pt" -> cell_id="C2_AD008_L5", chrom="chr1"
    """
    base = os.path.basename(filename)
    if not base.endswith(".pt"):
        raise ValueError(f"Expected .pt filename, got: {filename}")

    stem = base[:-3]
    if "_chr" not in stem:
        raise ValueError(f"Invalid .pt name (missing '_chr'): {filename}")

    cell_id, chrom_num = stem.rsplit("_chr", 1)
    if not chrom_num:
        raise ValueError(f"Invalid chrom suffix in: {filename}")

    return ParsedName(cell_id=cell_id, chrom=f"chr{chrom_num}")


def parse_pt_name_nagano(filename: str) -> ParsedName:
    """
    Nagano filenames may include '.' in cell_id; unified labels typically use '_'.

    Example
    -------
    "1CDX1.128_chr7.pt" -> raw cell_id="1CDX1.128" -> normalized "1CDX1_128"
    """
    parsed = parse_pt_name_default(filename)
    return ParsedName(cell_id=parsed.cell_id.replace(".", "_"), chrom=parsed.chrom)

