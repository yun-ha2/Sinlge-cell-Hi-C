"""
PCA-based cell embedding from per-cell chromosome-wise scHi-C contact matrices.

This module produces scHiCluster-like PCA cell embeddings using ONLY:
- contact_dir/<cell_id>.npz (keys: chr1, chr2, ...)
  values: sparse matrices (object-stored scipy sparse OR dense ndarray)

Pipeline
--------
1) Load per-cell per-chrom sparse matrices
2) (Optional) scHiCluster-like normalization:
   - coverage scaling to mean total contacts (cell-level)
   - remove diagonal
   - sqrtVC normalization
3) Flatten upper triangle (k=1) per chrom using a fixed mask
4) Per-chrom TruncatedSVD -> distance-normalized (divide by singular values)
5) Concatenate per-chrom embeddings -> global TruncatedSVD
6) Save:
   - cell_embeddings.npy
   - cell_names.txt
   - metadata.tsv
   - config.json

Notes
-----
- Designed to be dataset-agnostic (Lee/Nagano/hg19/mm9) via chromosome list.
"""

from __future__ import annotations

import gc
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm


# ----------------------------
# Config
# ----------------------------
@dataclass(frozen=True)
class PCACellEmbeddingConfig:
    chromosomes: List[str]
    dim_per_chr: int = 50
    dim_global: int = 50
    random_state: int = 0

    # normalization (scHiCluster-like)
    apply_normalization: bool = False
    coverage_balance: bool = True
    remove_diagonal: bool = True
    sqrtvc_norm: bool = True

    # filtering
    require_all_chroms: bool = True
    skip_empty_chrom: bool = True


# ----------------------------
# Sparse loading helpers
# ----------------------------
def _coo_to_csr_safe(x):
    from scipy.sparse import csr_matrix, coo_matrix
    if isinstance(x, csr_matrix):
        return x
    if isinstance(x, coo_matrix):
        return x.tocsr()
    if isinstance(x, np.ndarray) and x.dtype == object:
        # object array storing sparse
        return _coo_to_csr_safe(x.item()) if x.shape == () else _coo_to_csr_safe(x.ravel()[0])
    if isinstance(x, np.ndarray) and x.ndim == 2:
        # dense adjacency
        from scipy.sparse import coo_matrix as _coo
        return _coo(x).tocsr()
    raise TypeError(f"Unsupported matrix container: {type(x)}")


def _remove_diag(A):
    from scipy.sparse import diags
    return A - diags(A.diagonal())


def _sqrtvc_norm(E):
    """
    Symmetric normalization by sqrt of row/col sums (sqrtVC).
    """
    from scipy.sparse import diags
    E = (E + E.T).tocsr()
    d = E.sum(axis=0).A.ravel().astype(np.float32)
    d[d == 0] = 1.0
    invsqrt = 1.0 / np.sqrt(d)
    B = diags(invsqrt)
    return B.dot(E).dot(B)


def _total_contacts_npz(npz_path: Path, chroms: Iterable[str]) -> float:
    d = np.load(str(npz_path), allow_pickle=True)
    s = 0.0
    for c in chroms:
        if c in d.files:
            A = _coo_to_csr_safe(d[c])
            s += float(A.sum())
    return float(s)


def _scale_to_target_sum(A, target_sum: float):
    total = float(A.sum())
    if total <= 0:
        return A
    return (A / total) * float(target_sum)


def _normalize_matrix(A, target_total_sum: float, *, remove_diagonal: bool, sqrtvc: bool):
    if target_total_sum > 0:
        A = _scale_to_target_sum(A, target_total_sum)
    if remove_diagonal:
        A = _remove_diag(A)
        A.eliminate_zeros()
    if sqrtvc:
        A = _sqrtvc_norm(A)
        A.eliminate_zeros()
    return A


# ----------------------------
# Flatten utilities
# ----------------------------
def _upper_triangle_mask(n: int) -> Tuple[np.ndarray, np.ndarray]:
    ii, jj = np.triu_indices(n, k=1)
    return ii, jj


def _flatten_with_mask(A_csr, mask_ij) -> np.ndarray:
    """
    Flatten upper triangle using a precomputed mask.
    Using toarray() is memory-heavy, but simple and robust for moderate n.
    """
    i, j = mask_ij
    dense = A_csr.toarray().astype(np.float32)
    return dense[i, j].astype(np.float32)


# ----------------------------
# Main
# ----------------------------
def build_pca_cell_embeddings(
    *,
    contact_dir: str | Path,
    out_dir: str | Path,
    config: PCACellEmbeddingConfig,
    cell_ids: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build PCA cell embeddings from per-cell contact npz files.

    Parameters
    ----------
    contact_dir : directory containing <cell_id>.npz
    out_dir : output directory
    config : PCACellEmbeddingConfig
    cell_ids : optional explicit list of cell ids (stems). If None, enumerate from dir.

    Returns
    -------
    Z : (n_cells, dim_global) embedding
    names : list of cell_id in the same order as Z
    """
    contact_dir = Path(contact_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # enumerate cells
    if cell_ids is None:
        npz_files = sorted(p for p in contact_dir.glob("*.npz"))
        cell_ids = [p.stem for p in npz_files]
    else:
        npz_files = [contact_dir / f"{cid}.npz" for cid in cell_ids]

    if len(npz_files) == 0:
        raise RuntimeError(f"No .npz files found in {contact_dir}")

    # --------- optional coverage balance target ---------
    target_sum = 0.0
    if config.apply_normalization and config.coverage_balance:
        totals = []
        for p in tqdm(npz_files, desc="Compute total contacts (coverage balance)"):
            totals.append(_total_contacts_npz(p, config.chromosomes))
        target_sum = float(np.mean(totals)) if len(totals) else 0.0

    # --------- build reference masks per chrom (from first valid cell) ---------
    chrom_masks: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    chrom_nbins: Dict[str, int] = {}

    # find a reference file that has chromosomes
    for p in npz_files:
        d = np.load(str(p), allow_pickle=True)
        ok = all(c in d.files for c in config.chromosomes) if config.require_all_chroms else any(
            c in d.files for c in config.chromosomes
        )
        if not ok:
            continue
        for c in config.chromosomes:
            if c not in d.files:
                continue
            A = _coo_to_csr_safe(d[c])
            chrom_nbins[c] = int(A.shape[0])
            chrom_masks[c] = _upper_triangle_mask(chrom_nbins[c])
        if len(chrom_masks) > 0:
            break

    if len(chrom_masks) == 0:
        raise RuntimeError("Could not build reference masks (no valid cell with required chromosomes).")

    # --------- collect flattened vectors ---------
    per_chr_vectors: Dict[str, List[np.ndarray]] = {c: [] for c in config.chromosomes}
    names: List[str] = []
    meta_rows = []

    for p in tqdm(npz_files, desc="Flatten per chromosome"):
        cell_id = p.stem
        d = np.load(str(p), allow_pickle=True)

        # filtering rules
        if config.require_all_chroms and not all(c in d.files for c in config.chromosomes):
            meta_rows.append({"cell_id": cell_id, "status": "skip_missing_chrom"})
            continue

        # per cell collection
        usable = True
        cell_total = 0.0
        used_chroms = []

        # load & normalize each chrom
        mats = {}
        for c in config.chromosomes:
            if c not in d.files:
                continue
            A = _coo_to_csr_safe(d[c])
            if config.skip_empty_chrom and getattr(A, "nnz", 0) == 0:
                usable = False
                break

            if config.apply_normalization:
                A = _normalize_matrix(
                    A,
                    target_total_sum=target_sum if config.coverage_balance else 0.0,
                    remove_diagonal=config.remove_diagonal,
                    sqrtvc=config.sqrtvc_norm,
                )

            if config.skip_empty_chrom and getattr(A, "nnz", 0) == 0:
                usable = False
                break

            mats[c] = A
            used_chroms.append(c)
            cell_total += float(A.sum())

        if not usable:
            meta_rows.append({"cell_id": cell_id, "status": "skip_empty_or_broken"})
            continue

        # flatten using fixed masks
        for c in used_chroms:
            if c not in chrom_masks:
                # if chromosome not in reference masks, skip
                usable = False
                break
            vec = _flatten_with_mask(mats[c], chrom_masks[c])
            per_chr_vectors[c].append(vec)

        if not usable:
            meta_rows.append({"cell_id": cell_id, "status": "skip_mask_missing"})
            continue

        names.append(cell_id)
        meta_rows.append({"cell_id": cell_id, "status": "ok", "total_contacts": cell_total, "used_chroms": ",".join(used_chroms)})

        del d, mats
        gc.collect()

    if len(names) == 0:
        raise RuntimeError("No usable cells after filtering.")

    # --------- per-chrom SVD ---------
    per_chr_embeds: Dict[str, np.ndarray] = {}
    for c in config.chromosomes:
        if len(per_chr_vectors[c]) == 0:
            continue
        X = np.vstack(per_chr_vectors[c])  # (n_cells, n_features_chr)

        # choose n_components safely
        n_comp = min(config.dim_per_chr, X.shape[0] - 1, X.shape[1] - 1)
        if n_comp <= 0:
            continue

        svd = TruncatedSVD(n_components=n_comp, random_state=config.random_state, algorithm="arpack")
        Zc = svd.fit_transform(X)

        s = svd.singular_values_.astype(np.float32)
        pos = s > 0
        Zc = Zc[:, pos] / s[pos][None, :]
        per_chr_embeds[c] = Zc

        del X
        gc.collect()

    if len(per_chr_embeds) == 0:
        raise RuntimeError("No per-chrom embeddings were created.")

    # --------- global SVD ---------
    Z_cat = np.concatenate([per_chr_embeds[c] for c in per_chr_embeds], axis=1)
    n_comp_g = min(config.dim_global, Z_cat.shape[0] - 1, Z_cat.shape[1] - 1)
    if n_comp_g <= 0:
        raise RuntimeError("Global SVD n_components <= 0.")

    svd_g = TruncatedSVD(n_components=n_comp_g, random_state=config.random_state, algorithm="arpack")
    Z = svd_g.fit_transform(Z_cat)

    # --------- save outputs ---------
    np.save(str(out_dir / "cell_embeddings.npy"), Z.astype(np.float32))
    pd.Series(names, name="cell_id").to_csv(out_dir / "cell_names.txt", index=False)

    meta = pd.DataFrame(meta_rows)
    meta.to_csv(out_dir / "metadata.tsv", sep="\t", index=False)

    with open(out_dir / "config.json", "w") as f:
        json.dump(
            {"contact_dir": str(contact_dir), "out_dir": str(out_dir), **asdict(config)},
            f,
            indent=2,
        )

    return Z, names

