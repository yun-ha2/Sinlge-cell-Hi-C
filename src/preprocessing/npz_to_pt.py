"""
NPZ → PT builder (minimal).

This script builds per-(cell, chrom) PyTorch Geometric graphs (.pt)
using ONLY two existing .npz outputs:

Inputs
------
1) contact_dir/<cell_id>.npz
   - keys: chr1, chr2, ...
   - values: chromosome-wise sparse contact matrix
     supported encodings:
       (a) object array containing scipy sparse matrix (npz[chrom].item())
       (b) CSR components per chrom:
           <chrom>__data, <chrom>__indices, <chrom>__indptr, <chrom>__shape

2) feature_dir/<cell_id>.npz
   - keys: chr1, chr2, ...
   - values: node feature array (n_bins, d)

Outputs
-------
out_dir/<cell_id>_<chrom>.pt
  Data(x, edge_index, edge_attr)

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Data


# ----------------------------
# Utilities: file matching
# ----------------------------
def list_npz_by_stem(dir_path: str) -> Dict[str, str]:
    """Return stem -> filename mapping for *.npz in a directory."""
    out: Dict[str, str] = {}
    for f in os.listdir(dir_path):
        if f.endswith(".npz"):
            out[f[:-4]] = f
    return out


# ----------------------------
# Utilities: sparse loading
# ----------------------------
def _is_object_array(a: np.ndarray) -> bool:
    return isinstance(a, np.ndarray) and a.dtype == object and a.size == 1


def load_sparse_matrix(npz: np.lib.npyio.NpzFile, chrom: str):
    """
    Load scipy sparse matrix for chrom key from a contact npz.

    Supports:
    - npz[chrom] as object array storing scipy sparse (npz[chrom].item())
    - CSR component keys:
        chrom__data, chrom__indices, chrom__indptr, chrom__shape
    """
    import scipy.sparse as sp

    # (a) Lee-style object sparse (or sometimes dense)
    if chrom in npz.files:
        obj = npz[chrom]
        if _is_object_array(obj):
            mat = obj.item()
            if hasattr(mat, "tocoo") and hasattr(mat, "nnz"):
                return mat
        if isinstance(obj, np.ndarray) and obj.ndim == 2:
            # dense adjacency
            return sp.coo_matrix(obj)

    # (b) CSR components
    data_k = f"{chrom}__data"
    indices_k = f"{chrom}__indices"
    indptr_k = f"{chrom}__indptr"
    shape_k = f"{chrom}__shape"

    if all(k in npz.files for k in (data_k, indices_k, indptr_k, shape_k)):
        data = npz[data_k]
        indices = npz[indices_k]
        indptr = npz[indptr_k]
        shape = tuple(npz[shape_k].tolist())
        return sp.csr_matrix((data, indices, indptr), shape=shape)

    raise KeyError(f"Chrom '{chrom}' not found in contact npz (stem unknown encoding).")


def scipy_sparse_to_edge_list(mat) -> Tuple[torch.Tensor, torch.Tensor]:
    """scipy sparse -> (edge_index, edge_attr)"""
    mat = mat.tocoo()
    row = torch.tensor(mat.row, dtype=torch.long)
    col = torch.tensor(mat.col, dtype=torch.long)
    edge_index = torch.stack([row, col], dim=0)  # (2, E)
    edge_attr = torch.tensor(mat.data, dtype=torch.float32)  # (E,)
    return edge_index, edge_attr


# ----------------------------
# Core
# ----------------------------
def build_pt_minimal(
    contact_dir: str,
    feature_dir: str,
    out_dir: str,
    *,
    chromosomes: Optional[Iterable[str]] = None,
    strict_shape_check: bool = True,
    skip_empty: bool = True,
) -> None:
    """
    Minimal builder:
      - match cells by stem (<cell_id>.npz)
      - for each chrom in intersection, save Data(x, edge_index, edge_attr)
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    contact_map = list_npz_by_stem(contact_dir)
    feature_map = list_npz_by_stem(feature_dir)

    cell_ids = sorted(set(contact_map) & set(feature_map))
    if not cell_ids:
        raise RuntimeError(
            "No matched <cell_id>.npz between contact_dir and feature_dir.\n"
            f"contact_dir={contact_dir}\nfeature_dir={feature_dir}"
        )

    saved = 0
    failed = 0

    for cell_id in tqdm(cell_ids, desc="NPZ → PT (minimal)"):
        c_path = os.path.join(contact_dir, contact_map[cell_id])
        f_path = os.path.join(feature_dir, feature_map[cell_id])

        try:
            c_npz = np.load(c_path, allow_pickle=True)
            f_npz = np.load(f_path, allow_pickle=True)

            # Determine chrom list
            if chromosomes is None:
                feat_chroms = set(f_npz.files)
                contact_chroms = set(c_npz.files) | {
                    k.split("__data")[0] for k in c_npz.files if k.endswith("__data")
                }
                chroms = sorted(feat_chroms & contact_chroms)
            else:
                chroms = list(chromosomes)

            for chrom in chroms:
                if chrom not in f_npz.files:
                    continue

                x_np = f_npz[chrom]
                if x_np.ndim == 1:
                    x_np = x_np.reshape(-1, 1)
                x = torch.tensor(x_np, dtype=torch.float32)

                mat = load_sparse_matrix(c_npz, chrom)
                if skip_empty and getattr(mat, "nnz", 0) == 0:
                    continue

                if strict_shape_check:
                    if mat.shape[0] != x.size(0) or mat.shape[1] != x.size(0):
                        raise ValueError(
                            f"[{cell_id} {chrom}] shape mismatch: adj={mat.shape}, x={tuple(x.shape)}"
                        )

                edge_index, edge_attr = scipy_sparse_to_edge_list(mat)

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

                out_name = f"{cell_id}_{chrom}.pt"
                torch.save(data, os.path.join(out_dir, out_name))
                saved += 1

        except Exception as e:
            failed += 1
            print(f"[ERROR] {cell_id}: {e}")

    print(f"[DONE] saved_pt={saved} | failed_cells={failed} | out_dir={out_dir}")


# ----------------------------
# CLI
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal NPZ→PT builder")
    parser.add_argument("--contact_dir", required=True, help="Directory of contact <cell_id>.npz")
    parser.add_argument("--feature_dir", required=True, help="Directory of feature <cell_id>.npz")
    parser.add_argument("--out_dir", required=True, help="Output directory for .pt files")
    parser.add_argument(
        "--chromosomes",
        nargs="*",
        default=None,
        help="Optional list of chromosomes to process (e.g., chr1 chr2 ...). Default: intersection.",
    )
    parser.add_argument(
        "--no_strict_shape_check",
        action="store_true",
        help="Disable strict adjacency vs feature length shape check.",
    )
    parser.add_argument(
        "--keep_empty",
        action="store_true",
        help="Do not skip empty matrices (nnz==0).",
    )

    args = parser.parse_args()

    build_pt_minimal(
        contact_dir=args.contact_dir,
        feature_dir=args.feature_dir,
        out_dir=args.out_dir,
        chromosomes=args.chromosomes,
        strict_shape_check=not args.no_strict_shape_check,
        skip_empty=not args.keep_empty,
    )


if __name__ == "__main__":
    main()

