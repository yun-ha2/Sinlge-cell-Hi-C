# src/eval/loop_f1.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class LoopEvalConfig:
    resolution: int = 10_000
    min_bin_dist: int = 1
    max_bin_dist: int = 100
    allowed_slack_bp: int = 20_000  # +/- bp

    # parsing
    pred_sep_regex: str = r"\s+|,|\t"
    ref_sep_regex: str = r"\s+|\t|,"

    # dedup
    dedup_cols: Tuple[str, str, str] = ("chrom1", "x1", "y1")


def slack_bin(cfg: LoopEvalConfig) -> int:
    return max(0, int(round(cfg.allowed_slack_bp / cfg.resolution)))


# -------------------------
# IO
# -------------------------
def load_pred_csv(path: str | Path, cfg: LoopEvalConfig) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, sep=cfg.pred_sep_regex, engine="python")
    need = ["chrom1", "x1", "x2", "chrom2", "y1", "y2"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{path.name} missing columns: {miss}")
    return df[need].copy()


def load_ref_bedpe(path: str | Path, cfg: LoopEvalConfig) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, sep=cfg.ref_sep_regex, engine="python", header=None)
    df.columns = ["chrom1", "x1", "x2", "chrom2", "y1", "y2"]
    return df


def filter_intra_distance(df: pd.DataFrame, cfg: LoopEvalConfig) -> pd.DataFrame:
    df = df[df["chrom1"] == df["chrom2"]].copy()
    xbin = (df["x1"] // cfg.resolution).astype(int)
    ybin = (df["y1"] // cfg.resolution).astype(int)
    dist = (ybin - xbin).abs()
    mask = (dist >= cfg.min_bin_dist) & (dist <= cfg.max_bin_dist)
    return df[mask].copy()


def dedup(df: pd.DataFrame, cfg: LoopEvalConfig) -> pd.DataFrame:
    return df.drop_duplicates(subset=list(cfg.dedup_cols)).copy()


def minimal_xy(df: pd.DataFrame) -> pd.DataFrame:
    return df[["chrom1", "x1", "y1"]].copy()


# -------------------------
# slack matching
# -------------------------
def expand_slack(df_xy: pd.DataFrame, cfg: LoopEvalConfig, sb: int) -> pd.DataFrame:
    if df_xy.empty:
        return df_xy

    dfs = []
    for dx in range(-sb, sb + 1):
        for dy in range(-sb, sb + 1):
            tmp = df_xy.copy()
            tmp["x1"] = tmp["x1"] + dx * cfg.resolution
            tmp["y1"] = tmp["y1"] + dy * cfg.resolution
            tmp = tmp[(tmp["x1"] >= 0) & (tmp["y1"] >= 0)]
            dfs.append(tmp)

    out = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["chrom1", "x1", "y1"])
    return out


def slack_precision(label_xy: pd.DataFrame, pred_xy: pd.DataFrame, cfg: LoopEvalConfig, sb: int) -> float:
    if pred_xy.empty:
        return 0.0
    slack_label = expand_slack(label_xy, cfg, sb)
    tp = pred_xy.merge(slack_label, on=["chrom1", "x1", "y1"], how="inner")
    return float(len(tp) / max(1, len(pred_xy)))


def slack_recall(label_xy: pd.DataFrame, pred_xy: pd.DataFrame, cfg: LoopEvalConfig, sb: int) -> float:
    if label_xy.empty:
        return 0.0
    slack_pred = expand_slack(pred_xy, cfg, sb)
    tp = slack_pred.merge(label_xy, on=["chrom1", "x1", "y1"], how="inner")
    return float(len(tp) / max(1, len(label_xy)))


def slack_f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else float(2 * p * r / (p + r))


# -------------------------
# Ref providers
# -------------------------
def infer_cell_type_from_filename(path: str | Path) -> str:
    # A10_AD001_L23.csv -> L23
    stem = Path(path).stem
    toks = stem.split("_")
    return toks[-1] if toks else "Unknown"


def make_nagano_ref_provider(ref_path: str | Path, cfg: LoopEvalConfig) -> Callable[[str, str], pd.DataFrame]:
    """
    Returns a function: (cell_id, cell_type) -> ref_df
    Uses the same ref for all cells.
    """
    ref_path = Path(ref_path)
    ref = load_ref_bedpe(ref_path, cfg)
    ref = dedup(filter_intra_distance(ref, cfg), cfg)
    ref_xy = minimal_xy(ref)

    def _provider(cell_id: str, cell_type: str) -> pd.DataFrame:
        return ref_xy

    return _provider


def make_lee_ref_provider(ref_dir: str | Path, cfg: LoopEvalConfig) -> Callable[[str, str], pd.DataFrame]:
    """
    Lee: cell_type -> one of {MG_chr.bedpe, ODC_chr.bedpe, Neuron_chr.bedpe}
    """
    ref_dir = Path(ref_dir)

    cache: Dict[str, pd.DataFrame] = {}

    def load_one(name: str) -> pd.DataFrame:
        if name in cache:
            return cache[name]
        p = ref_dir / name
        if not p.exists():
            raise FileNotFoundError(p)
        df = load_ref_bedpe(p, cfg)
        df = dedup(filter_intra_distance(df, cfg), cfg)
        df = minimal_xy(df)
        cache[name] = df
        return df

    def _provider(cell_id: str, cell_type: str) -> pd.DataFrame:
        ct = str(cell_type).upper()
        if ct == "MG":
            return load_one("MG_chr.bedpe")
        if ct == "ODC":
            return load_one("ODC_chr.bedpe")
        return load_one("Neuron_chr.bedpe")

    return _provider


# -------------------------
# Main evaluation
# -------------------------
def evaluate_pred_dir(
    *,
    pred_dir: str | Path,
    cfg: LoopEvalConfig,
    ref_provider: Callable[[str, str], pd.DataFrame],
    cell_type_from_filename: Callable[[str | Path], str] = infer_cell_type_from_filename,
    ignore_name_contains: Tuple[str, ...] = ("f1", "score", "metric"),
) -> pd.DataFrame:
    pred_dir = Path(pred_dir)
    sb = slack_bin(cfg)

    pred_files = sorted(pred_dir.glob("*.csv"))
    pred_files = [p for p in pred_files if not any(t in p.name.lower() for t in ignore_name_contains)]

    rows: List[dict] = []

    for pf in pred_files:
        cell_id = pf.stem
        cell_type = cell_type_from_filename(pf)

        pred = load_pred_csv(pf, cfg)
        pred = dedup(filter_intra_distance(pred, cfg), cfg)
        pred_xy = minimal_xy(pred)

        ref_xy = ref_provider(cell_id, cell_type)

        prec = slack_precision(ref_xy, pred_xy, cfg, sb)
        rec = slack_recall(ref_xy, pred_xy, cfg, sb)
        f1 = slack_f1(prec, rec)

        rows.append(
            dict(
                cell_id=cell_id,
                cell_type=cell_type,
                n_pred=int(len(pred_xy)),
                n_ref=int(len(ref_xy)),
                Precision=prec,
                Recall=rec,
                F1=f1,
                slack_bin=int(sb),
            )
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["cell_type", "cell_id"]).reset_index(drop=True)
    return df

