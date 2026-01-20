"""
Local Degree Profile (LDP) feature extraction for scHi-C graphs.

This module computes node-level Local Degree Profile features from a sparse
adjacency matrix and applies log1p + per-chromosome min-max normalization.

For each node i, LDP features are:
- d_i   : degree of node i
- AND_i : mean degree of neighbors of i
- MND_i : min degree of neighbors of i
- MXD_i : max degree of neighbors of i
- SD_i  : std of degrees of neighbors of i

Output is a dense array of shape (n_nodes, 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
from concurrent.futures import ProcessPoolExecutor
from functools import partial


# ============================================================
# Spec
# ============================================================

@dataclass(frozen=True)
class LDPSpec:
    """
    Specification for LDP extraction on a directory of per-cell .npz files.

    Parameters
    ----------
    input_dir : str or pathlib.Path
        Directory containing per-cell sparse matrices (.npz).
    output_dir : str or pathlib.Path
        Directory to save per-cell LDP features (.npz).
    max_workers : int
        Number of processes for parallel execution.
    exclude_chroms : sequence[str]
        Chromosomes to skip.
    allow_pickle : bool
        Whether to allow pickle when loading npz. Needed if matrices were saved as objects.
    """
    input_dir: Union[str, Path]
    output_dir: Union[str, Path]
    max_workers: int = 8
    exclude_chroms: Sequence[str] = ("chrX", "chrY", "chrM")
    allow_pickle: bool = True


# ============================================================
# Core math
# ============================================================

def compute_ldp(
    sparse_matrix: sp.spmatrix,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute Local Degree Profile (LDP) features for one sparse adjacency matrix.

    Parameters
    ----------
    sparse_matrix : scipy.sparse.spmatrix
        Adjacency matrix (n_nodes, n_nodes).
    normalize : bool, default=True
        If True, apply log1p and per-feature min-max normalization.

    Returns
    -------
    numpy.ndarray
        Dense array of shape (n_nodes, 5).
    """
    A = sparse_matrix.tocsr()
    n = A.shape[0]

    # degree (weighted degree if matrix has weights)
    degree = np.asarray(A.sum(axis=1)).ravel()

    feats = np.zeros((n, 5), dtype=np.float32)

    for i in range(n):
        neighbors = A[i].indices
        if neighbors.size == 0:
            continue

        nd = degree[neighbors]
        feats[i, 0] = degree[i]
        feats[i, 1] = float(nd.mean())
        feats[i, 2] = float(nd.min())
        feats[i, 3] = float(nd.max())
        feats[i, 4] = float(nd.std())

    if not normalize:
        return feats

    log_feats = np.log1p(feats)

    min_vals = log_feats.min(axis=0)
    max_vals = log_feats.max(axis=0)
    denom = max_vals - min_vals
    denom[denom == 0] = 1.0

    return (log_feats - min_vals) / denom


# ============================================================
# IO helpers
# ============================================================

def load_cell_npz(path: Union[str, Path], *, allow_pickle: bool = True) -> Mapping[str, object]:
    """
    Load per-cell .npz file.

    Notes
    -----
    Many scHi-C pipelines save scipy sparse matrices as Python objects inside npz,
    so allow_pickle=True is typically required.
    """
    return np.load(str(path), allow_pickle=allow_pickle)


def coerce_to_csr(obj: object) -> sp.csr_matrix:
    """
    Convert loaded npz item to scipy.sparse.csr_matrix.

    Handles:
    - already a scipy sparse matrix
    - object arrays storing a sparse matrix (common when using np.savez with .item())
    """
    if sp.issparse(obj):
        return obj.tocsr()

    # common pattern: stored as 0-d object array
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        obj = obj.item()

    if sp.issparse(obj):
        return obj.tocsr()

    raise TypeError(f"Cannot convert object of type {type(obj)} to csr_matrix.")


def save_cell_features_npz(path: Union[str, Path], arrays: Mapping[str, np.ndarray]) -> None:
    """Save per-cell features (.npz). Keys are chromosome names."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **dict(arrays))


# ============================================================
# Per-cell processing
# ============================================================

def process_one_cell_file(
    file_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    exclude_chroms: Sequence[str] = ("chrX", "chrY", "chrM"),
    allow_pickle: bool = True,
) -> Optional[Path]:
    """
    Compute chromosome-wise LDP features for one per-cell .npz file.

    Returns
    -------
    pathlib.Path or None
        Saved output path, or None if skipped.
    """
    file_path = Path(file_path)
    if file_path.suffix != ".npz":
        return None

    cell_id = file_path.stem
    out_path = Path(output_dir) / f"{cell_id}.npz"

    data = load_cell_npz(file_path, allow_pickle=allow_pickle)

    ldp_by_chrom: Dict[str, np.ndarray] = {}
    for chrom in data.files:
        if chrom in exclude_chroms:
            continue

        mat = coerce_to_csr(data[chrom])
        ldp_by_chrom[chrom] = compute_ldp(mat, normalize=True)

    if len(ldp_by_chrom) == 0:
        return None

    save_cell_features_npz(out_path, ldp_by_chrom)
    return out_path


# ============================================================
# Directory runner (parallel)
# ============================================================

def run_ldp_directory(spec: LDPSpec) -> None:
    """
    Run LDP extraction for all .npz files in a directory.
    """
    input_dir = Path(spec.input_dir)
    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.npz"))
    if len(files) == 0:
        raise RuntimeError(f"No .npz files found in {input_dir}")

    print(f"Processing {len(files)} cells from {input_dir}")
    print(f"Saving LDP features to {output_dir} (max_workers={spec.max_workers})")

    func = partial(
        process_one_cell_file,
        output_dir=output_dir,
        exclude_chroms=spec.exclude_chroms,
        allow_pickle=spec.allow_pickle,
    )

    saved = 0
    with ProcessPoolExecutor(max_workers=spec.max_workers) as ex:
        for out in ex.map(func, files):
            if out is not None:
                saved += 1

    print(f"Saved LDP features for {saved} / {len(files)} cells")

