"""
I/O utilities for chromosome-wise sparse matrices stored in .npz.

We store chromosome-wise contact matrices as:
  - key   : chromosome name (e.g., 'chr1')
  - value : scipy.sparse CSR matrix

Saved via np.savez_compressed with pickled scipy objects.
"""

from __future__ import annotations

from typing import Dict, Union, Optional
import os

import numpy as np
import scipy.sparse as sp


SparseMatrix = Union[sp.csr_matrix, sp.coo_matrix, sp.csc_matrix, sp.dok_matrix, sp.lil_matrix]


def save_sparse_npz(
    path: str,
    matrices: Dict[str, SparseMatrix],
    *,
    overwrite: bool = True,
) -> None:
    """
    Save chromosome-wise sparse matrices into a compressed .npz file.

    Parameters
    ----------
    path : str
        Output path ending with .npz
    matrices : dict[str, sparse matrix]
        Chromosome -> sparse matrix
    overwrite : bool
        If False, raise error when file exists
    """
    if not path.endswith(".npz"):
        raise ValueError(f"Output must end with .npz: {path}")

    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"File exists: {path}")

    payload = {}
    for chrom, mat in matrices.items():
        if mat is None:
            continue
        if not sp.issparse(mat):
            raise TypeError(f"Matrix for {chrom} is not a scipy sparse matrix: {type(mat)}")
        payload[chrom] = mat.tocsr()

    if len(payload) == 0:
        # Save an empty file is usually not desired; fail early
        raise ValueError("No non-empty sparse matrices to save.")

    np.savez_compressed(path, **payload)


def load_sparse_npz(
    path: str,
    *,
    as_format: str = "csr",
) -> Dict[str, sp.spmatrix]:
    """
    Load chromosome-wise sparse matrices from a .npz file.

    Notes
    -----
    - Uses allow_pickle=True because scipy sparse matrices are stored as objects.

    Parameters
    ----------
    path : str
        Path to .npz
    as_format : str
        One of {'csr', 'coo', 'csc'}

    Returns
    -------
    dict[str, scipy.sparse matrix]
    """
    if as_format not in {"csr", "coo", "csc"}:
        raise ValueError(f"Unsupported as_format={as_format}. Use 'csr'/'coo'/'csc'.")

    data = np.load(path, allow_pickle=True)
    out: Dict[str, sp.spmatrix] = {}

    for chrom in data.files:
        obj = data[chrom]

        # When saved with np.savez_compressed, scipy sparse matrices often come back as
        # 0-d object arrays -> use .item()
        if isinstance(obj, np.ndarray) and obj.shape == () and obj.dtype == object:
            mat = obj.item()
        elif obj.dtype == object and obj.size == 1:
            mat = obj.item()
        else:
            # In some cases it may already be a sparse matrix object
            mat = obj

        if not sp.issparse(mat):
            raise TypeError(f"Loaded object for {chrom} is not sparse: {type(mat)}")

        if as_format == "csr":
            out[chrom] = mat.tocsr()
        elif as_format == "coo":
            out[chrom] = mat.tocoo()
        else:
            out[chrom] = mat.tocsc()

    return out


def inspect_sparse_npz(path: str) -> Dict[str, Dict[str, int]]:
    """
    Quick inspection utility: returns per-chromosome stats (shape, nnz).

    Returns
    -------
    dict[chrom] = {'n_rows':..., 'n_cols':..., 'nnz':...}
    """
    mats = load_sparse_npz(path, as_format="csr")
    stats = {}
    for chrom, mat in mats.items():
        stats[chrom] = {"n_rows": mat.shape[0], "n_cols": mat.shape[1], "nnz": int(mat.nnz)}
    return stats

