from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitSpec:
    """Cell-level train/validation split specification."""
    val_ratio: float = 0.2
    seed: int = 42
    stratify: bool = True


@dataclass(frozen=True)
class SplitResult:
    train_files: List[str]
    val_files: List[str]
    train_cell_ids: List[str]
    val_cell_ids: List[str]
    meta: Dict


def read_manifest(path: str) -> List[str]:
    lines = Path(path).read_text().splitlines()
    out: List[str] = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        out.append(ln)
    return out


def write_manifest(path: str, files: Sequence[str]) -> None:
    Path(path).write_text("\n".join(files) + "\n")


def extract_cell_id_from_pt(filename: str) -> str:
    """Default: '<cell_id>_chr<k>.pt' -> '<cell_id>'."""
    stem = filename[:-3] if filename.endswith(".pt") else filename
    if "_chr" not in stem:
        raise ValueError(f"Invalid pt filename (missing _chr): {filename}")
    return stem.rsplit("_chr", 1)[0]


def load_unified_label_table(label_tsv: str, sep: str = "\t") -> pd.DataFrame:
    df = pd.read_csv(label_tsv, sep=sep)
    if not {"cell_id", "cell_type"}.issubset(df.columns):
        raise ValueError(f"label_tsv must contain cell_id, cell_type columns. Got {list(df.columns)}")
    df["cell_id"] = df["cell_id"].astype(str)
    df["cell_type"] = df["cell_type"].astype(str)
    return df


def split_pt_files_by_cell(
    pt_files: Sequence[str],
    *,
    cell_id_fn: Callable[[str], str] = extract_cell_id_from_pt,
    normalize_cell_id_fn: Optional[Callable[[str], str]] = None,
    labels: Optional[pd.DataFrame] = None,
    spec: SplitSpec = SplitSpec(),
) -> SplitResult:
    """
    Split pt files by cell_id to prevent leakage across train/val.

    If labels provided and spec.stratify=True:
      - split is stratified by cell_type at the cell level.
    """
    if not (0.0 < spec.val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0,1). Got {spec.val_ratio}")

    file_to_cell: Dict[str, str] = {}
    for f in pt_files:
        cid = cell_id_fn(f)
        if normalize_cell_id_fn is not None:
            cid = normalize_cell_id_fn(cid)
        file_to_cell[f] = cid

    cell_ids = sorted(set(file_to_cell.values()))
    rng = np.random.default_rng(spec.seed)

    # stratified cell split
    if labels is not None and spec.stratify:
        label_map = labels.set_index("cell_id")["cell_type"].to_dict()
        known = [c for c in cell_ids if c in label_map]
        unknown = [c for c in cell_ids if c not in label_map]

        if len(known) == 0:
            # fallback random
            rng.shuffle(cell_ids)
            n_val = int(round(len(cell_ids) * spec.val_ratio))
            val_cells = sorted(cell_ids[:n_val])
            train_cells = sorted(cell_ids[n_val:])
        else:
            # type-wise split
            type_to_cells: Dict[str, List[str]] = {}
            for c in known:
                type_to_cells.setdefault(label_map[c], []).append(c)

            train_cells: List[str] = []
            val_cells: List[str] = []
            for ct, cells in type_to_cells.items():
                cells = sorted(cells)
                rng.shuffle(cells)
                n_val = int(round(len(cells) * spec.val_ratio))
                val_cells.extend(cells[:n_val])
                train_cells.extend(cells[n_val:])

            train_cells.extend(unknown)
            train_cells = sorted(set(train_cells))
            val_cells = sorted(set(val_cells))

    else:
        # random split
        shuffled = cell_ids.copy()
        rng.shuffle(shuffled)
        n_val = int(round(len(shuffled) * spec.val_ratio))
        val_cells = sorted(shuffled[:n_val])
        train_cells = sorted(shuffled[n_val:])

    train_set = set(train_cells)
    val_set = set(val_cells)

    train_files = sorted([f for f, c in file_to_cell.items() if c in train_set])
    val_files = sorted([f for f, c in file_to_cell.items() if c in val_set])

    meta = {
        "seed": spec.seed,
        "val_ratio": spec.val_ratio,
        "stratify": bool(spec.stratify and labels is not None),
        "n_total_files": len(pt_files),
        "n_train_files": len(train_files),
        "n_val_files": len(val_files),
        "n_total_cells": len(cell_ids),
        "n_train_cells": len(train_cells),
        "n_val_cells": len(val_cells),
    }

    return SplitResult(
        train_files=train_files,
        val_files=val_files,
        train_cell_ids=train_cells,
        val_cell_ids=val_cells,
        meta=meta,
    )


def save_split(out_dir: str, result: SplitResult) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(str(out / "train.txt"), result.train_files)
    write_manifest(str(out / "val.txt"), result.val_files)
    (out / "meta.json").write_text(json.dumps(result.meta, indent=2) + "\n")

