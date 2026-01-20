"""
DNA bin sequence extraction.

This module slices a reference genome FASTA into fixed-size bins and stores
per-bin sequences as a JSON list. The resulting JSON can be used for downstream
feature construction (e.g., DNA embedding extraction).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

from pyfaidx import Fasta

JsonBin = Dict[str, Union[str, int]]


# ============================================================
# Spec
# ============================================================

@dataclass(frozen=True)
class BinSpec:
    """
    Specification for genome bin extraction.
    """
    fasta_path: Union[str, Path]
    output_json_path: Union[str, Path]
    bin_size: int
    chromosomes: Sequence[str]


# ============================================================
# Utilities
# ============================================================

def open_fasta(fasta_path: Union[str, Path]) -> Fasta:
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")
    return Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)


def resolve_chrom_name(
    fasta: Fasta,
    chrom: str,
    *,
    allow_alias: bool = True,
) -> Optional[str]:
    """
    Resolve chromosome name against FASTA headers.
    """
    if chrom in fasta:
        return chrom

    if not allow_alias:
        return None

    if chrom.startswith("chr"):
        alt = chrom[3:]
    else:
        alt = f"chr{chrom}"

    if alt in fasta:
        return alt

    return None


def canonical_chrom_name(chrom: str) -> str:
    """
    Convert chromosome name to UCSC-style (chr-prefixed).
    """
    chrom = chrom.strip()
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def iter_bins_for_chrom(
    fasta: Fasta,
    chrom: str,
    bin_size: int,
    *,
    include_intervals: bool = False,
) -> Iterable[JsonBin]:
    """
    Iterate fixed-size bins for a single chromosome.
    """
    chrom_len = len(fasta[chrom])
    out_chrom = canonical_chrom_name(chrom)

    for start in range(0, chrom_len, bin_size):
        end = min(start + bin_size, chrom_len)
        bin_id = start // bin_size + 1
        seq = fasta[chrom][start:end].seq

        rec: JsonBin = {
            "chrom": out_chrom,
            "bin": int(bin_id),
            "sequence": seq,
        }

        if include_intervals:
            rec["start"] = int(start)
            rec["end"] = int(end)

        yield rec


# ============================================================
# Core
# ============================================================

def build_bin_sequence_list(
    fasta_path: Union[str, Path],
    chromosomes: Sequence[str],
    bin_size: int,
    *,
    allow_alias: bool = True,
    strict: bool = False,
    include_intervals: bool = False,
) -> List[JsonBin]:
    """
    Build genome bin sequences as a list of JSON records.
    """
    if bin_size <= 0:
        raise ValueError(f"bin_size must be positive, got {bin_size}")

    fasta = open_fasta(fasta_path)
    results: List[JsonBin] = []
    skipped = 0

    for chrom in chromosomes:
        resolved = resolve_chrom_name(fasta, chrom, allow_alias=allow_alias)
        if resolved is None:
            msg = f"Chromosome '{chrom}' not found in FASTA"
            if strict:
                raise KeyError(msg)
            print(msg)
            skipped += 1
            continue

        results.extend(
            iter_bins_for_chrom(
                fasta,
                resolved,
                bin_size,
                include_intervals=include_intervals,
            )
        )

    print(f"Built {len(results)} bins (bin_size={bin_size}, skipped_chroms={skipped})")
    return results


def save_bins_to_json(
    bins: List[JsonBin],
    output_json_path: Union[str, Path],
    *,
    indent: int = 2,
) -> Path:
    out = Path(output_json_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        json.dump(bins, f, indent=indent)

    print(f"Saved {len(bins)} bins to {out}")
    return out


def run_make_bins(
    spec: BinSpec,
    *,
    allow_alias: bool = True,
    strict: bool = False,
    include_intervals: bool = False,
) -> Path:
    bins = build_bin_sequence_list(
        fasta_path=spec.fasta_path,
        chromosomes=spec.chromosomes,
        bin_size=spec.bin_size,
        allow_alias=allow_alias,
        strict=strict,
        include_intervals=include_intervals,
    )
    return save_bins_to_json(bins, spec.output_json_path)

