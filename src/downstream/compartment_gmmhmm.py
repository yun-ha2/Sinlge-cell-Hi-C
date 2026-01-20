# src/downstream/compartment_gmmhmm.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from hmmlearn.hmm import GMMHMM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================================
# Config
# =========================================================

@dataclass
class CompartmentHMMConfig:
    """
    Compartment calling from node embeddings z using 2-state GMMHMM.

    Input z files:
      - pattern: <cell>_chr<chrom>.npz  (example: 1CDS1.206_chr15.npz)
      - or: <cell>_chr<chrom>.z.npz    (if you used z extractor naming)
      - the npz must contain one 2D array (T, D), usually key 'z'

    Output:
      - per-cell table: <cell>.tsv with columns [chrom, start, end, score]
      - consensus table: consensus.tsv (mean score across cells)
    """
    resolution: int = 10_000           # input bin size (10kb)
    coarsen_factor: int = 10           # 10kb -> 100kb
    hmm_n_states: int = 2
    hmm_iter: int = 100
    random_seed: int = 0

    # preprocessing for HMM
    standardize: bool = True
    pca_dim: Optional[int] = 8         # None or 0 => no PCA

    # training sampling
    train_max_seqs: int = 12_000

    # GC polarity alignment
    gc_table_path: str = ""            # parquet or tsv/csv with columns: chrom,start,end,gc
    gc_column: str = "gc"

    # chrom sizes
    chrom_sizes_path: str = ""         # chrom.sizes (chrom \t size)
    chrom_prefix: str = "chr"

    # IO
    float_format: str = "%.5f"


# =========================================================
# I/O helpers
# =========================================================

_FNAME_PAT = re.compile(r"^(?P<cell>.+?)_chr(?P<chrom>[^.]+)\.(?P<ext>npz|z\.npz)$")

def parse_z_filename(path: str | Path) -> Tuple[str, str]:
    """
    Accept:
      - <cell>_chr<chrom>.npz
      - <cell>_chr<chrom>.z.npz
    Return: (cell, chrom) where chrom startswith 'chr'
    """
    name = Path(path).name
    m = _FNAME_PAT.match(name)
    if not m:
        raise ValueError(f"Unexpected z filename: {name}")

    cell = m.group("cell")
    chrom_raw = m.group("chrom")
    chrom = chrom_raw if chrom_raw.startswith("chr") else f"chr{chrom_raw}"
    return cell, chrom


def load_z_from_npz(path: str | Path) -> np.ndarray:
    """
    Load (T, D) z embedding array from npz.
    If key unknown, use the first array.
    """
    with np.load(path, allow_pickle=False) as npz:
        if len(npz.files) == 0:
            raise ValueError(f"{path}: empty npz")
        key = "z" if "z" in npz.files else npz.files[0]
        z = npz[key]
    if z.ndim != 2:
        raise ValueError(f"{path}: expected 2D (T,D), got {z.shape}")
    return z


# =========================================================
# Genome bins / GC
# =========================================================

def get_chrom_sizes(chrom_sizes_path: str) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    with open(chrom_sizes_path) as f:
        for line in f:
            if not line.strip():
                continue
            chrom, size = line.split()[:2]
            sizes[chrom] = int(size)
    return sizes


def get_bin_count(chrom_sizes: Dict[str, int], resolution: int, chrom: str) -> int:
    return (chrom_sizes[chrom] + resolution - 1) // resolution


def create_bin_df(chrom_sizes: Dict[str, int], resolution: int, chrom: str) -> pd.DataFrame:
    size = chrom_sizes[chrom]
    nbin = get_bin_count(chrom_sizes, resolution, chrom)
    rows = []
    for i in range(nbin):
        s = i * resolution
        e = min((i + 1) * resolution, size)
        rows.append((chrom, s, e))
    return pd.DataFrame(rows, columns=["chrom", "start", "end"])


def load_gc_table(gc_table_path: str) -> Dict[str, pd.DataFrame]:
    """
    Expect merged GC table across chroms with columns:
      chrom, start, end, gc
    Supports parquet or tsv/csv.
    Return dict: chrom -> df sorted by start
    """
    p = Path(gc_table_path)
    if not p.exists():
        raise FileNotFoundError(f"GC table not found: {gc_table_path}")

    if p.suffix == ".parquet":
        all_gc = pd.read_parquet(p)
    else:
        # auto sep
        all_gc = pd.read_csv(p, sep=r"\s+|\t|,", engine="python")

    need = {"chrom", "start", "end"}
    if not need.issubset(all_gc.columns):
        raise ValueError(f"GC table missing columns: {need - set(all_gc.columns)}")

    # find gc column
    gc_col = "gc" if "gc" in all_gc.columns else None
    if gc_col is None:
        # fallback: last col
        gc_col = all_gc.columns[-1]
    all_gc = all_gc.rename(columns={gc_col: "gc"}).copy()

    out = {}
    for chrom, df in all_gc.groupby("chrom"):
        df = df.sort_values(["start", "end"]).reset_index(drop=True)
        out[str(chrom)] = df
    return out


# =========================================================
# HMM training / transform
# =========================================================

def collect_train_blocks(
    z_files: List[str],
    max_seqs: int,
    standardize: bool,
    pca_dim: Optional[int],
    random_state: int,
) -> Tuple[np.ndarray, List[int], Optional[StandardScaler], Optional[PCA]]:
    """
    Sample up to max_seqs sequences, concatenate as X, and return lengths for hmmlearn.
    """
    rng = np.random.RandomState(random_state)
    if len(z_files) <= max_seqs:
        pick = z_files
    else:
        pick = rng.choice(z_files, size=max_seqs, replace=False).tolist()

    X_blocks: List[np.ndarray] = []
    lengths: List[int] = []

    for path in tqdm(pick, desc=f"[HMM Train] load up to {max_seqs} seqs"):
        z = load_z_from_npz(path)
        X_blocks.append(z)
        lengths.append(int(z.shape[0]))

    X = np.concatenate(X_blocks, axis=0)

    scaler = None
    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X = scaler.fit_transform(X)

    pca = None
    if pca_dim is not None and int(pca_dim) > 0:
        pca = PCA(n_components=int(pca_dim), random_state=random_state)
        X = pca.fit_transform(X)

    return X, lengths, scaler, pca


def transform_seq(z: np.ndarray, scaler: Optional[StandardScaler], pca: Optional[PCA]) -> np.ndarray:
    X = z
    if scaler is not None:
        X = scaler.transform(X)
    if pca is not None:
        X = pca.transform(X)
    return X


def train_gmmhmm(
    X_train: np.ndarray,
    lengths: List[int],
    n_states: int,
    n_iter: int,
    random_state: int,
) -> GMMHMM:
    hmm = GMMHMM(
        n_components=int(n_states),
        covariance_type="diag",
        n_iter=int(n_iter),
        random_state=int(random_state),
    )
    hmm.fit(X_train, lengths=lengths)
    return hmm


# =========================================================
# Compartment scoring
# =========================================================

def choose_A_state_by_gc(cluster_labels: np.ndarray, gc: np.ndarray, n_states: int = 2) -> int:
    """
    Choose A-state as the one with higher mean GC.
    """
    means = []
    for k in range(n_states):
        idx = (cluster_labels == k)
        means.append(gc[idx].mean() if np.any(idx) else -1e9)
    return int(np.argmax(means))


def score_from_proba(proba: np.ndarray, A_state_idx: int) -> np.ndarray:
    """
    score = p(A) - p(B) for 2-state
    """
    pA = proba[:, A_state_idx]
    pB = proba[:, 1 - A_state_idx]
    return pA - pB


def coarsen_to_factor(
    bin_df: pd.DataFrame,
    proba: np.ndarray,
    gc: np.ndarray,
    factor: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Coarsen 10kb bins to 100kb bins by grouping factor bins.
    - bin coords: min(start), max(end)
    - proba: mean over group
    - gc: mean over group
    """
    n = len(bin_df)
    group = np.arange(n) // int(factor)

    bin_out = (
        bin_df.assign(_g=group)
        .groupby("_g", as_index=False)
        .agg({"chrom": "first", "start": "min", "end": "max"})
        .drop(columns=["_g"])
    )

    proba_out = (
        pd.DataFrame(proba).assign(_g=group)
        .groupby("_g", as_index=False)
        .mean()
        .drop(columns=["_g"])
        .values
    )

    gc_out = (
        pd.Series(gc).to_frame("gc").assign(_g=group)
        .groupby("_g", as_index=False)["gc"]
        .mean()["gc"]
        .values
    )

    return bin_out, proba_out, gc_out


# =========================================================
# High-level: inference
# =========================================================

def infer_compartment_for_one_seq(
    *,
    z_path: str,
    hmm: GMMHMM,
    scaler: Optional[StandardScaler],
    pca: Optional[PCA],
    chrom_sizes: Dict[str, int],
    gc_tables: Dict[str, pd.DataFrame],
    cfg: CompartmentHMMConfig,
) -> pd.DataFrame:
    """
    Return 100kb compartment bed-like table for a single (cell, chrom) sequence.
    Columns: chrom, start, end, score
    """
    cell, chrom = parse_z_filename(z_path)
    z = load_z_from_npz(z_path)  # (T, D)

    # bins
    bin10 = create_bin_df(chrom_sizes, cfg.resolution, chrom)
    n = min(len(bin10), z.shape[0])
    bin10 = bin10.iloc[:n].reset_index(drop=True)
    z = z[:n]

    # GC series aligned to 10kb bins
    if chrom not in gc_tables:
        raise KeyError(f"GC table has no chrom={chrom}")
    gc10 = gc_tables[chrom]["gc"].values[:n]

    # posterior
    X = transform_seq(z, scaler=scaler, pca=pca)
    proba10 = hmm.predict_proba(X)  # (n, 2)

    # coarsen to 100kb
    bin100, proba100, gc100 = coarsen_to_factor(bin10, proba10, gc10, cfg.coarsen_factor)

    cluster100 = np.argmax(proba100, axis=1)
    A_idx = choose_A_state_by_gc(cluster100, gc100, n_states=cfg.hmm_n_states)
    score100 = score_from_proba(proba100, A_idx)

    out = bin100[["chrom", "start", "end"]].copy()
    out["score"] = score100.astype(float)
    return out


def write_per_cell_tables_and_consensus(
    *,
    z_files: List[str],
    hmm: GMMHMM,
    scaler: Optional[StandardScaler],
    pca: Optional[PCA],
    cfg: CompartmentHMMConfig,
    out_dir: str | Path,
) -> Tuple[List[str], Path]:
    """
    Writes:
      - per-cell: out_dir/per_cell/<cell>.tsv (append chrom blocks)
      - consensus: out_dir/consensus.tsv (mean across cells)
    """
    out_dir = Path(out_dir)
    per_cell_dir = out_dir / "per_cell"
    per_cell_dir.mkdir(parents=True, exist_ok=True)

    chrom_sizes = get_chrom_sizes(cfg.chrom_sizes_path)
    gc_tables = load_gc_table(cfg.gc_table_path)

    written_cells = set()

    # cell -> path
    # normalize cell_id in filename 그대로 사용
    for z_path in tqdm(z_files, desc="[Infer] compartment per seq"):
        cell, chrom = parse_z_filename(z_path)
        out_path = per_cell_dir / f"{cell}.tsv"

        df = infer_compartment_for_one_seq(
            z_path=z_path,
            hmm=hmm,
            scaler=scaler,
            pca=pca,
            chrom_sizes=chrom_sizes,
            gc_tables=gc_tables,
            cfg=cfg,
        )

        header = not out_path.exists()
        df.to_csv(out_path, sep="\t", index=False, header=header, mode="a", float_format=cfg.float_format)
        written_cells.add(cell)

    # consensus
    tables = []
    for cell in sorted(written_cells):
        fp = per_cell_dir / f"{cell}.tsv"
        if fp.exists():
            tables.append(pd.read_csv(fp, sep="\t"))

    if len(tables) == 0:
        raise RuntimeError("No per-cell tables were written; cannot build consensus.")

    big = pd.concat(tables, axis=0, ignore_index=True)
    consensus = (
        big.groupby(["chrom", "start", "end"], as_index=False)["score"]
        .mean()
        .sort_values(["chrom", "start", "end"])
    )

    consensus_path = out_dir / "consensus.tsv"
    consensus.to_csv(consensus_path, sep="\t", index=False, float_format=cfg.float_format)

    return sorted(list(written_cells)), consensus_path

