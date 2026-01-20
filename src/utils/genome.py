"""
Genome-related utilities.

Currently:
- load chromosome sizes from a UCSC-style chrom.sizes file.
"""

from __future__ import annotations
from typing import Dict, Iterable, Optional


def load_chrom_sizes(
    path: str,
    *,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    require_chr_prefix: bool = True,
) -> Dict[str, int]:
    """
    Load chromosome sizes from a chrom.sizes file.

    File format (whitespace-separated):
        chr1    197195432
        chr2    181748087
        ...

    Parameters
    ----------
    path : str
        Path to chrom.sizes
    include : iterable[str] or None
        If given, only keep these chromosomes (after normalization).
    exclude : iterable[str] or None
        If given, drop these chromosomes.
    require_chr_prefix : bool
        If True, ensure chromosome names start with 'chr' by adding it if missing.

    Returns
    -------
    dict[str, int]
        Chromosome -> size (bp)
    """
    include_set = set(include) if include is not None else None
    exclude_set = set(exclude) if exclude is not None else set()

    chrom_sizes: Dict[str, int] = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            chrom, size = parts[0], parts[1]

            if require_chr_prefix and not chrom.startswith("chr"):
                chrom = "chr" + chrom

            try:
                size_int = int(size)
            except ValueError:
                continue

            if include_set is not None and chrom not in include_set:
                continue
            if chrom in exclude_set:
                continue

            chrom_sizes[chrom] = size_int

    if len(chrom_sizes) == 0:
        raise ValueError(f"No chromosome sizes loaded from: {path}")

    return chrom_sizes

