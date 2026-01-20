"""
Merge per-cell LDP features with shared DNA embeddings into final node features.

Inputs
------
1) Per-cell LDP feature files:
   ldp_dir/<cell_id>.npz
     chr1 -> (n_bins, 5)
     chr2 -> (n_bins, 5)
     ...

2) Shared DNA embedding file (merged):
   dna_npz
     chr1 -> (n_bins, d_dna)
     chr2 -> (n_bins, d_dna)
     ...

Output
------
out_dir/<cell_id>.npz
  chr1 -> (n_bins, 5 + d_dna)
  chr2 -> (n_bins, 5 + d_dna)
  ...

Notes
-----
- Chromosome keys are expected to be UCSC-style (chr*).
- For each chromosome, n_bins must match between LDP and DNA.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np


# ============================================================
# Spec
# ============================================================

@dataclass(frozen=True)
class MergeLDPSpec:
    """
    Parameters
    ----------
    ldp_dir : str or Path
        Directory containing per-cell LDP files (*.npz). Used to enumerate cells.
    dna_npz : str or Path
        Path to merged DNA embedding .npz (keys: chr1, chr2, ...).
    out_dir : str or Path
        Output directory for merged per-cell node features.
    chromosomes : sequence[str]
        Chromosomes to process (e.g., chr1..chr22).
    """
    ldp_dir: Union[str, Path]
    dna_npz: Union[str, Path]
    out_dir: Union[str, Path]
    chromosomes: Sequence[str]


# ============================================================
# Utilities
# ============================================================

def ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Convert (n,) -> (n,1). Keep (n,d) as-is."""
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected 1D or 2D array, got shape={arr.shape}")


def list_cell_ids_from_ldp(ldp_dir: Union[str, Path]) -> List[str]:
    """Enumerate cell IDs from <ldp_dir>/*.npz."""
    return sorted(p.stem for p in Path(ldp_dir).glob("*.npz"))


# ============================================================
# Core
# ============================================================

def merge_one_cell(
    cell_id: str,
    *,
    ldp_dir: Union[str, Path],
    dna_npz: np.lib.npyio.NpzFile,
    out_dir: Union[str, Path],
    chromosomes: Sequence[str],
) -> Path:
    """Merge LDP + DNA for one cell and save as <cell_id>.npz."""
    ldp_path = Path(ldp_dir) / f"{cell_id}.npz"
    ldp = np.load(str(ldp_path), allow_pickle=True)

    merged: Dict[str, np.ndarray] = {}
    for chrom in chromosomes:
        if chrom not in ldp.files:
            raise KeyError(f"Missing {chrom} in LDP: {ldp_path}")
        if chrom not in dna_npz.files:
            raise KeyError(f"Missing {chrom} in DNA npz: {chrom}")

        ldp_chr = ensure_2d(ldp[chrom])
        dna_chr = ensure_2d(dna_npz[chrom])

        if ldp_chr.shape[0] != dna_chr.shape[0]:
            raise ValueError(
                f"Shape mismatch for {cell_id} {chrom}: LDP={ldp_chr.shape}, DNA={dna_chr.shape}"
            )

        merged[chrom] = np.concatenate([ldp_chr, dna_chr], axis=1)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{cell_id}.npz"
    np.savez_compressed(str(out_path), **merged)
    return out_path


def run_merge_ldp_dna(spec: MergeLDPSpec) -> None:
    """Run merging for all cells in spec.ldp_dir."""
    ldp_dir = Path(spec.ldp_dir)
    out_dir = Path(spec.out_dir)

    cell_ids = list_cell_ids_from_ldp(ldp_dir)
    if len(cell_ids) == 0:
        raise RuntimeError(f"No LDP .npz files found in {ldp_dir}")

    dna_npz = np.load(str(spec.dna_npz), allow_pickle=True)

    print(f"Found {len(cell_ids)} cells in {ldp_dir}")
    print(f"Using DNA embedding: {spec.dna_npz}")
    print(f"Saving merged node features to {out_dir}")

    saved = 0
    for cell_id in cell_ids:
        out = merge_one_cell(
            cell_id,
            ldp_dir=ldp_dir,
            dna_npz=dna_npz,
            out_dir=out_dir,
            chromosomes=spec.chromosomes,
        )
        saved += 1
        if saved % 50 == 0:
            print(f"Saved {saved}/{len(cell_ids)} cells...")

    print(f"Saved merged node features for {saved} cells")

