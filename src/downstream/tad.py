# src/downstream/tads/tad.py
from __future__ import annotations

import json
import os
import bisect
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.sparse import diags


# =========================================================
# Config
# =========================================================

@dataclass
class TADFromZConfig:
    """
    TAD/TLD boundary extraction from node embeddings z (per cell x chrom).

    Input format:
      - each file: <cell_id>.<chrom>.z.npz
      - contains key "z": (n_bins, z_dim)

    Outputs:
      out_dir/
        cells/
          <cell_id>.<chrom>.tld.bed
          <cell_id>.<chrom>.boundaries.bed
        consensus/
          <chrom>.tld.bed
          <chrom>.tld.boundaries.bed
          allChrom.tld.bed
          boundary_vote_frac.hard.bedgraph
        metadata.tsv
        config.json
    """
    binsize: int = 10_000

    # segmentation (HAC)
    avg_tld_size_bins: int = 20
    linkage: str = "ward"
    use_pca: bool = True
    pca_ncomp: int = 16

    # ctcf refinement (optional)
    ctcf_bed: Optional[str] = None
    ctcf_window_bp: int = 20_000

    # consensus
    consensus_vote: float = 0.3
    consensus_min_sep_bins: int = 3

    # vote track
    save_vote_bedgraph: bool = True

    # I/O
    save_per_cell_boundaries: bool = True


# =========================================================
# Filename parsing
# =========================================================

def parse_cell_chrom_from_znpz(path: str | Path) -> Tuple[str, str]:
    """
    Expect: <cell_id>.<chrom>.z.npz
      e.g. 1CDX1_101.chr1.z.npz
           A10_AD008_Pvalb.chr10.z.npz
    """
    p = Path(path)
    stem = p.name
    # remove suffix ".z.npz"
    if not stem.endswith(".z.npz"):
        base = p.stem  # might be ".z"
    else:
        base = stem[:-len(".z.npz")]

    # split by last ".chr"
    # safest: find ".chr" occurrence near end
    if ".chr" in base:
        cell_id, chrom = base.rsplit(".chr", 1)
        chrom = "chr" + chrom
    else:
        # fallback: split last "."
        toks = base.split(".")
        if len(toks) >= 2:
            chrom = toks[-1]
            cell_id = ".".join(toks[:-1])
            if not chrom.startswith("chr"):
                chrom = f"chr{chrom}"
        else:
            cell_id, chrom = base, "unknown"

    # normalize
    cell_id = str(cell_id).replace(" ", "_").replace(".", "_")
    chrom = str(chrom)
    return cell_id, chrom


# =========================================================
# Core utils
# =========================================================

def boundaries_from_labels(labels: np.ndarray) -> np.ndarray:
    """boundaries = indices b where label changes between b and b+1"""
    labels = np.asarray(labels)
    if len(labels) <= 1:
        return np.array([], dtype=int)
    return np.where(np.diff(labels) != 0)[0].astype(int)


def contiguous_labels_to_intervals(labels: np.ndarray) -> List[Tuple[int, int]]:
    labels = np.asarray(labels)
    if len(labels) == 0:
        return []
    intervals = []
    s = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            intervals.append((s, i - 1))
            s = i
    intervals.append((s, len(labels) - 1))
    return intervals


def tld_bed_rows(chrom: str, intervals: List[Tuple[int, int]], binsize: int) -> List[Tuple[str, int, int]]:
    rows = []
    for s, e in intervals:
        rows.append((chrom, int(s * binsize), int((e + 1) * binsize)))
    return rows


def hac_segments_from_Z(
    Z: np.ndarray,
    avg_size_bins: int,
    linkage: str,
    use_pca: bool,
    pca_n: int,
) -> np.ndarray:
    """
    HAC segmentation with adjacency connectivity to enforce contiguous domains.
    """
    Z = np.asarray(Z, dtype=float)
    N = Z.shape[0]
    if N <= 1:
        return np.zeros(N, dtype=int)

    X = Z
    if use_pca and Z.shape[1] > pca_n:
        X = PCA(n_components=pca_n, random_state=0).fit_transform(Z)

    # connectivity: i connected to i-1 and i+1
    conn = diags([np.ones(N - 1), np.ones(N - 1)], offsets=[-1, 1], shape=(N, N))

    k = max(1, int(round(N / max(1, avg_size_bins))))
    k = min(k, N)

    Xs = StandardScaler().fit_transform(X)

    hac = AgglomerativeClustering(
        n_clusters=k,
        linkage=linkage,
        connectivity=conn
    )
    return hac.fit_predict(Xs)


# =========================================================
# CTCF refinement (optional)
# =========================================================

def load_ctcf_midpoints(ctcf_bed: str) -> Dict[str, np.ndarray]:
    df = pd.read_csv(ctcf_bed, sep="\t", header=None)
    df = df.iloc[:, :3]
    df.columns = ["chrom", "start", "end"]
    mids = {}
    for chrom, g in df.groupby("chrom"):
        arr = ((g["start"].to_numpy() + g["end"].to_numpy()) // 2).astype(int)
        arr.sort()
        mids[chrom] = arr
    return mids


def _count_in_window(mid_sorted: np.ndarray, center_bp: int, halfwin_bp: int) -> int:
    L = center_bp - halfwin_bp
    R = center_bp + halfwin_bp
    li = bisect.bisect_left(mid_sorted, L)
    ri = bisect.bisect_right(mid_sorted, R)
    return max(0, ri - li)


def refine_boundaries_by_ctcf(
    chrom: str,
    boundaries: np.ndarray,
    binsize: int,
    ctcf_mid: Dict[str, np.ndarray],
    halfwin_bp: int,
) -> np.ndarray:
    """
    Keep boundaries that are enriched for CTCF motif midpoints within +/- window.
    We cluster boundary motif counts into 2 groups and keep the higher-mean group.
    """
    boundaries = np.asarray(boundaries, dtype=int)
    if len(boundaries) == 0:
        return boundaries
    if chrom not in ctcf_mid:
        return boundaries

    mids = ctcf_mid[chrom]
    centers = boundaries * int(binsize)
    counts = np.array([_count_in_window(mids, int(c), int(halfwin_bp)) for c in centers], dtype=float).reshape(-1, 1)

    if counts.max() == counts.min():
        return boundaries

    km = KMeans(n_clusters=2, n_init=10, random_state=0)
    lab = km.fit_predict(counts)

    means = [counts[lab == i].mean() for i in (0, 1)]
    keep_label = int(np.argmax(means))
    keep_idx = np.where(lab == keep_label)[0]
    return boundaries[keep_idx]


# =========================================================
# Consensus + vote track
# =========================================================

def consensus_boundaries(
    boundaries_list: List[np.ndarray],
    n_bins: int,
    vote: float,
    min_sep_bins: int,
) -> np.ndarray:
    if len(boundaries_list) == 0:
        return np.array([], dtype=int)

    K = len(boundaries_list)
    acc = np.zeros(n_bins, dtype=float)

    for arr in boundaries_list:
        arr = np.asarray(arr, dtype=int)
        arr = arr[(arr >= 0) & (arr < n_bins - 1)]
        acc[arr] += 1.0

    frac = acc / float(K)
    idx = np.where(frac >= vote)[0].astype(int)

    if len(idx) <= 1:
        return idx

    # greedily choose peaks by vote strength, enforcing separation
    order = idx[np.argsort(frac[idx])[::-1]]
    chosen: List[int] = []
    for b in order:
        if all(abs(int(b) - c) >= int(min_sep_bins) for c in chosen):
            chosen.append(int(b))
    chosen.sort()
    return np.asarray(chosen, dtype=int)


def boundary_vote_frac_hard(boundaries_list: List[np.ndarray], n_bins: int) -> np.ndarray:
    if len(boundaries_list) == 0:
        return np.zeros(n_bins, dtype=float)
    K = len(boundaries_list)
    acc = np.zeros(n_bins, dtype=float)
    for arr in boundaries_list:
        arr = np.asarray(arr, dtype=int)
        arr = arr[(arr >= 0) & (arr < n_bins - 1)]
        acc[arr] += 1.0
    return acc / float(K)


def save_bedgraph(frac_by_chrom: Dict[str, np.ndarray], out_path: str | Path, binsize: int) -> None:
    rows = []
    for chrom, frac in frac_by_chrom.items():
        for i, v in enumerate(frac):
            rows.append((chrom, int(i * binsize), int((i + 1) * binsize), float(v)))
    pd.DataFrame(rows).to_csv(out_path, sep="\t", header=False, index=False)


# =========================================================
# Main pipeline
# =========================================================

def run_tad_from_z_npz(
    *,
    z_dir: str | Path,
    out_dir: str | Path,
    cfg: TADFromZConfig,
) -> pd.DataFrame:
    """
    Run TAD/TLD extraction from z npz files.
    """
    z_dir = Path(z_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cell_dir = out_dir / "cells"
    cons_dir = out_dir / "consensus"
    cell_dir.mkdir(parents=True, exist_ok=True)
    cons_dir.mkdir(parents=True, exist_ok=True)

    z_files = sorted(z_dir.glob("*.z.npz"))
    if len(z_files) == 0:
        raise RuntimeError(f"No *.z.npz found under: {z_dir}")

    # optional CTCF
    ctcf_mid = None
    if cfg.ctcf_bed:
        if not Path(cfg.ctcf_bed).exists():
            raise FileNotFoundError(f"CTCF bed not found: {cfg.ctcf_bed}")
        ctcf_mid = load_ctcf_midpoints(cfg.ctcf_bed)

    # store boundaries for consensus
    boundaries_all: Dict[str, List[np.ndarray]] = defaultdict(list)
    n_bins_by_chrom: Dict[str, int] = {}

    meta_rows = []

    for fp in z_files:
        cell_id, chrom = parse_cell_chrom_from_znpz(fp)

        d = np.load(fp, allow_pickle=False)
        if "z" not in d:
            raise ValueError(f"Missing key 'z' in {fp}")
        Z = d["z"]  # (n_bins, z_dim)

        if Z.ndim != 2:
            raise ValueError(f"z must be 2D array, got {Z.shape} in {fp}")

        n_bins = int(Z.shape[0])
        n_bins_by_chrom[chrom] = n_bins

        # HAC segmentation
        labels = hac_segments_from_Z(
            Z,
            avg_size_bins=cfg.avg_tld_size_bins,
            linkage=cfg.linkage,
            use_pca=cfg.use_pca,
            pca_n=cfg.pca_ncomp,
        )

        # raw boundaries
        b_raw = boundaries_from_labels(labels)

        # CTCF refinement (optional)
        b_ref = b_raw
        if ctcf_mid is not None:
            b_ref = refine_boundaries_by_ctcf(
                chrom=chrom,
                boundaries=b_raw,
                binsize=cfg.binsize,
                ctcf_mid=ctcf_mid,
                halfwin_bp=cfg.ctcf_window_bp,
            )

        # intervals -> TLD bed
        ivals = contiguous_labels_to_intervals(labels)
        tld_rows = tld_bed_rows(chrom, ivals, cfg.binsize)

        # save per-cell
        out_tld = cell_dir / f"{cell_id}.{chrom}.tld.bed"
        pd.DataFrame(tld_rows).to_csv(out_tld, sep="\t", header=False, index=False)

        if cfg.save_per_cell_boundaries:
            out_bd = cell_dir / f"{cell_id}.{chrom}.boundaries.bed"
            bd_rows = [(chrom, int(b * cfg.binsize), int(b * cfg.binsize + 1)) for b in b_ref]
            pd.DataFrame(bd_rows).to_csv(out_bd, sep="\t", header=False, index=False)

        boundaries_all[chrom].append(b_ref)

        meta_rows.append({
            "file": fp.name,
            "cell_id": cell_id,
            "chrom": chrom,
            "n_bins": n_bins,
            "z_dim": int(Z.shape[1]),
            "n_domains": int(len(ivals)),
            "n_boundaries_raw": int(len(b_raw)),
            "n_boundaries_refined": int(len(b_ref)),
        })

    meta = pd.DataFrame(meta_rows)
    meta.to_csv(out_dir / "metadata.tsv", sep="\t", index=False)

    # save config
    with open(out_dir / "config.json", "w") as f:
        json.dump({"z_dir": str(z_dir), "out_dir": str(out_dir), **asdict(cfg)}, f, indent=2)

    # ---------------------------------------------------------
    # Consensus per chromosome
    # ---------------------------------------------------------
    all_cons_rows = []
    vote_tracks: Dict[str, np.ndarray] = {}

    for chrom, blist in boundaries_all.items():
        n_bins = n_bins_by_chrom[chrom]

        b_cons = consensus_boundaries(
            boundaries_list=[np.asarray(x, dtype=int) for x in blist],
            n_bins=n_bins,
            vote=cfg.consensus_vote,
            min_sep_bins=cfg.consensus_min_sep_bins,
        )

        # consensus boundary bed
        bd_rows = [(chrom, int(b * cfg.binsize), int(b * cfg.binsize + 1)) for b in b_cons]
        pd.DataFrame(bd_rows).to_csv(cons_dir / f"{chrom}.tld.boundaries.bed", sep="\t", header=False, index=False)

        # consensus intervals from b_cons
        ivals = []
        s = 0
        for b in b_cons:
            ivals.append((s, int(b)))
            s = int(b) + 1
        if s <= n_bins - 1:
            ivals.append((s, n_bins - 1))

        rows = tld_bed_rows(chrom, ivals, cfg.binsize)
        pd.DataFrame(rows).to_csv(cons_dir / f"{chrom}.tld.bed", sep="\t", header=False, index=False)

        all_cons_rows.extend(rows)

        # vote track
        if cfg.save_vote_bedgraph:
            vote_tracks[chrom] = boundary_vote_frac_hard(blist, n_bins=n_bins)

    pd.DataFrame(all_cons_rows).to_csv(cons_dir / "allChrom.tld.bed", sep="\t", header=False, index=False)

    if cfg.save_vote_bedgraph:
        save_bedgraph(vote_tracks, cons_dir / "boundary_vote_frac.hard.bedgraph", binsize=cfg.binsize)

    return meta

