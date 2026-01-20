#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from glob import glob
import pandas as pd
from tqdm.auto import tqdm


def build_parser():
    p = argparse.ArgumentParser("Build cluster-wise compartment consensus from per-cell compartment tables")
    p.add_argument("--cell_embed_csv", required=True, help="CSV containing cell_id and cluster columns")
    p.add_argument("--per_cell_dir", required=True, help="directory with per-cell compartment tables (*.tsv)")
    p.add_argument("--out_dir", required=True)

    p.add_argument("--cell_id_col", type=str, default="cell_id")
    p.add_argument("--cluster_col", type=str, default="cluster")

    return p


def main():
    args = build_parser().parse_args()

    meta = pd.read_csv(args.cell_embed_csv)
    if args.cell_id_col not in meta.columns:
        raise ValueError(f"Missing {args.cell_id_col} in {args.cell_embed_csv}")
    if args.cluster_col not in meta.columns:
        raise ValueError(f"Missing {args.cluster_col} in {args.cell_embed_csv}")

    meta[args.cell_id_col] = meta[args.cell_id_col].astype(str)
    clusters = sorted(meta[args.cluster_col].unique())

    per_cell_dir = Path(args.per_cell_dir)
    files = sorted(glob(str(per_cell_dir / "*.tsv")))
    file_map = {Path(f).stem: f for f in files}  # stem == cell_id

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cl in clusters:
        cells = meta.loc[meta[args.cluster_col] == cl, args.cell_id_col].tolist()
        valid = [c for c in cells if c in file_map]

        if len(valid) == 0:
            print(f"[Skip] cluster {cl}: no valid cells")
            continue

        tables = []
        for c in tqdm(valid, desc=f"[cluster {cl}]"):
            df = pd.read_csv(file_map[c], sep="\t")
            tables.append(df)

        big = pd.concat(tables, axis=0, ignore_index=True)
        cons = (
            big.groupby(["chrom", "start", "end"], as_index=False)["score"]
            .mean()
            .sort_values(["chrom", "start", "end"])
        )
        out_path = out_dir / f"consensus_cluster{cl}.tsv"
        cons.to_csv(out_path, sep="\t", index=False)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

