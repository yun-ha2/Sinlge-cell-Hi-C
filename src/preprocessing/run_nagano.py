"""
Nagano scHi-C preprocessing pipeline.

This script converts raw schic2 outputs (adj files with restriction fragment IDs)
into chromosome-wise binned sparse contact matrices (.npz).

Pipeline:
1. Map restriction fragment IDs (fends) to genomic coordinates
2. Filter interactions (cis-only, remove chrX/Y, remove self-loops)
3. Merge duplicate contacts and enforce i < j ordering
4. Cell-level QC based on minimum total contact counts
5. Bin contacts at a fixed resolution
6. Save chromosome-wise sparse matrices in a unified .npz format

This script is intended to be a complete, reproducible preprocessing pipeline
for the Nagano scHi-C dataset.
"""

from __future__ import annotations

import os
import argparse
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from src.utils.genome import load_chrom_sizes
from src.data_io.sparse_npz import save_sparse_npz


# ============================================================
# Fend utilities
# ============================================================

def load_fend_maps(fends_path: str) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Load schic2 fend table and build mapping dictionaries.

    Parameters
    ----------
    fends_path : str
        Path to schic2 fend file (e.g., GATC.fends)

    Returns
    -------
    chr_map : dict
        Mapping from fend ID -> chromosome (e.g., chr1)
    coord_map : dict
        Mapping from fend ID -> genomic coordinate (bp)
    """
    fends = pd.read_csv(
        fends_path,
        sep="\t",
        header=None,
        names=["fend", "chr", "coord"],
        dtype=str,
        na_filter=False,
    )

    # prepend "chr" to chromosome column
    fends["chr"] = "chr" + fends["chr"]
    fends["coord"] = fends["coord"].astype(int)

    chr_map = dict(zip(fends["fend"], fends["chr"]))
    coord_map = dict(zip(fends["fend"], fends["coord"]))

    return chr_map, coord_map


# ============================================================
# Contact loading & filtering
# ============================================================

def load_and_filter_adj(
    adj_path: str,
    chr_map: Dict[str, str],
    coord_map: Dict[str, int],
    remove_sex_chrom: bool = True,
) -> pd.DataFrame:
    """
    Load a schic2 adj file and convert it into genomic contacts.

    Filtering steps:
    - keep numeric counts only
    - map fends to (chr, coord)
    - remove chrX / chrY (optional)
    - keep cis interactions only
    - remove self-interactions
    - enforce coord1 < coord2
    - merge duplicate interactions

    Returns
    -------
    pd.DataFrame with columns:
    ['chr1', 'coord1', 'chr2', 'coord2', 'count']
    """
    adj = pd.read_csv(
        adj_path,
        sep="\t",
        dtype=str,
        usecols=["fend1", "fend2", "count"],
        na_filter=False,
    )

    # numeric counts only
    adj = adj[adj["count"].str.isnumeric()]
    adj["count"] = adj["count"].astype(int)

    # map fends to genomic coordinates
    adj["chr1"] = adj["fend1"].map(chr_map)
    adj["coord1"] = adj["fend1"].map(coord_map)
    adj["chr2"] = adj["fend2"].map(chr_map)
    adj["coord2"] = adj["fend2"].map(coord_map)

    adj = adj.dropna(subset=["chr1", "coord1", "chr2", "coord2"])

    if remove_sex_chrom:
        adj = adj[~adj["chr1"].isin(["chrX", "chrY"])]
        adj = adj[~adj["chr2"].isin(["chrX", "chrY"])]

    # cis only
    adj = adj[adj["chr1"] == adj["chr2"]]

    # remove self-interactions
    adj = adj[adj["coord1"] != adj["coord2"]]

    # enforce coord1 < coord2
    swap_mask = adj["coord1"] > adj["coord2"]
    adj.loc[swap_mask, ["coord1", "coord2"]] = (
        adj.loc[swap_mask, ["coord2", "coord1"]].values
    )

    # merge duplicates
    adj = adj.groupby(
        ["chr1", "coord1", "chr2", "coord2"], as_index=False
    )["count"].sum()

    return adj


# ============================================================
# Binning
# ============================================================

def bin_contacts(
    df: pd.DataFrame,
    chrom_sizes: Dict[str, int],
    bin_size: int,
) -> Dict[str, sp.csr_matrix]:
    """
    Bin genomic contacts into chromosome-wise sparse matrices.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered genomic contacts
    chrom_sizes : dict
        Chromosome sizes (bp)
    bin_size : int
        Bin resolution in bp

    Returns
    -------
    dict[str, scipy.sparse.csr_matrix]
    """
    matrices: Dict[str, sp.dok_matrix] = {}

    for chrom, size in chrom_sizes.items():
        n_bins = size // bin_size + 1
        matrices[chrom] = sp.dok_matrix((n_bins, n_bins), dtype=np.int32)

    for _, row in df.iterrows():
        chrom = row["chr1"]
        if chrom not in matrices:
            continue
        i = row["coord1"] // bin_size
        j = row["coord2"] // bin_size
        c = int(row["count"])

        matrices[chrom][i, j] += c
        matrices[chrom][j, i] += c  # symmetric

    return {
        chrom: mat.tocsr()
        for chrom, mat in matrices.items()
        if mat.nnz > 0
    }


# ============================================================
# Main pipeline
# ============================================================

def run(
    input_base: str,
    output_dir: str,
    fends_path: str,
    chrom_size_path: str,
    bin_size: int = 100_000,
    min_total: int = 2000,
):
    """
    Run Nagano scHi-C preprocessing pipeline.

    Parameters
    ----------
    input_base : str
        Directory containing per-cell schic2 outputs
    output_dir : str
        Output directory for .npz files
    fends_path : str
        Path to GATC.fends file
    chrom_size_path : str
        Path to chrom.sizes file
    bin_size : int
        Binning resolution (bp)
    min_total : int
        Minimum total contact count per cell
    """
    os.makedirs(output_dir, exist_ok=True)

    chrom_sizes = load_chrom_sizes(chrom_size_path)
    chr_map, coord_map = load_fend_maps(fends_path)

    cells = [
        c for c in os.listdir(input_base)
        if os.path.isdir(os.path.join(input_base, c))
    ]

    print(f"🚀 Processing {len(cells)} cells")

    for cell in tqdm(cells, desc="Nagano preprocessing"):
        adj_path = os.path.join(input_base, cell, "adj")
        if not os.path.exists(adj_path):
            continue

        df = load_and_filter_adj(adj_path, chr_map, coord_map)

        total_contacts = df["count"].sum()
        if total_contacts < min_total:
            continue

        matrices = bin_contacts(df, chrom_sizes, bin_size)
        if len(matrices) == 0:
            continue

        out_path = os.path.join(output_dir, f"{cell}.npz")
        save_sparse_npz(out_path, matrices)


# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nagano scHi-C preprocessing pipeline"
    )
    parser.add_argument("--input_base", required=True,
                        help="Base directory containing schic2 outputs")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for .npz files")
    parser.add_argument("--fends", required=True,
                        help="Path to GATC.fends file")
    parser.add_argument("--chrom_sizes", required=True,
                        help="Path to chrom.sizes file")
    parser.add_argument("--bin_size", type=int, default=100_000,
                        help="Binning resolution (bp)")
    parser.add_argument("--min_total", type=int, default=2000,
                        help="Minimum total contact count per cell")

    args = parser.parse_args()

    run(
        input_base=args.input_base,
        output_dir=args.output_dir,
        fends_path=args.fends,
        chrom_size_path=args.chrom_sizes,
        bin_size=args.bin_size,
        min_total=args.min_total,
    )

