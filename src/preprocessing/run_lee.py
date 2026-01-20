"""
Lee scHi-C preprocessing pipeline.

This script converts Lee et al. scHi-C cooler files (.mcool)
into chromosome-wise sparse contact matrices (.npz).

Pipeline:
1. Load multi-resolution cooler (.mcool)
2. Select a fixed resolution
3. Extract per-chromosome sparse contact matrices
4. Remove diagonal (self-interactions)
5. Save chromosome-wise sparse matrices in a unified .npz format

The output format is identical to that of the Nagano preprocessing pipeline,
enabling unified downstream feature construction and graph learning.
"""

from __future__ import annotations

import os
import argparse
from typing import Dict, Iterable

import numpy as np
import scipy.sparse as sp
import cooler
from tqdm import tqdm

from src.data_io.sparse_npz import save_sparse_npz


# ============================================================
# Utilities
# ============================================================

def list_mcool_files(input_dir: str) -> Iterable[str]:
    """List .mcool files in a directory."""
    return sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".mcool")
    )


def extract_cell_id(mcool_path: str) -> str:
    """
    Extract a cell/sample identifier from a Lee mcool filename.

    Example:
    181218_21yr_3_C2_AD008_L5_10kb_contacts.mcool
        -> C2_AD008_L5
    """
    fname = os.path.basename(mcool_path).replace(".mcool", "")
    parts = fname.split("_")

    # Lee naming convention: use tokens [3:6]
    if len(parts) < 6:
        raise ValueError(f"Unexpected mcool filename format: {fname}")

    return "_".join(parts[3:6])


# ============================================================
# Core preprocessing
# ============================================================

def process_mcool(
    mcool_path: str,
    resolution: int,
    exclude_chroms: Iterable[str],
) -> Dict[str, sp.csr_matrix]:
    """
    Process a single mcool file into chromosome-wise sparse matrices.

    Parameters
    ----------
    mcool_path : str
        Path to .mcool file
    resolution : int
        Target resolution (bp)
    exclude_chroms : iterable
        Chromosomes to exclude (e.g., chrX, chrY, chrM)

    Returns
    -------
    dict[str, scipy.sparse.csr_matrix]
    """
    c = cooler.Cooler(f"{mcool_path}::/resolutions/{resolution}")

    matrices: Dict[str, sp.csr_matrix] = {}

    for chrom in c.chromnames:
        if chrom in exclude_chroms:
            continue

        # fetch sparse contact matrix
        mat = c.matrix(balance=False, sparse=True).fetch(chrom)

        if mat.nnz == 0:
            continue

        # remove diagonal (self-interactions)
        mat = mat.tocsr()
        mat.setdiag(0)
        mat.eliminate_zeros()

        if mat.nnz == 0:
            continue

        matrices[chrom] = mat

    return matrices


# ============================================================
# Main pipeline
# ============================================================

def run(
    input_dir: str,
    output_dir: str,
    resolution: int,
    exclude_chroms: Iterable[str] = ("chrX", "chrY", "chrM"),
    min_total: int = 0,
):
    """
    Run Lee scHi-C preprocessing pipeline.

    min_total:
      If > 0, skip a cell if total contact count (sum of matrix entries across chromosomes)
      is below min_total.
    """
    os.makedirs(output_dir, exist_ok=True)

    mcool_files = list_mcool_files(input_dir)
    if len(mcool_files) == 0:
        raise RuntimeError(f"No .mcool files found in {input_dir}")

    print(f"🚀 Processing {len(mcool_files)} mcool files")

    for mcool_path in tqdm(mcool_files, desc="Lee preprocessing"):
        cell_id = extract_cell_id(mcool_path)

        matrices = process_mcool(
            mcool_path=mcool_path,
            resolution=resolution,
            exclude_chroms=exclude_chroms,
        )

        if len(matrices) == 0:
            continue

        # optional QC: total contact count
        if min_total and min_total > 0:
            total_contacts = int(sum(m.data.sum() for m in matrices.values()))
            if total_contacts < min_total:
                continue

        out_path = os.path.join(output_dir, f"{cell_id}.npz")
        save_sparse_npz(out_path, matrices)


# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lee scHi-C preprocessing pipeline"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing .mcool files",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for .npz files",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        required=True,
        help="Target resolution in bp (e.g., 100000)",
    )
    parser.add_argument(
        "--exclude_chroms",
        nargs="+",
        default=["chrX", "chrY", "chrM"],
        help="Chromosomes to exclude",
    )

    args = parser.parse_args()

    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        resolution=args.resolution,
        exclude_chroms=args.exclude_chroms,
    )

