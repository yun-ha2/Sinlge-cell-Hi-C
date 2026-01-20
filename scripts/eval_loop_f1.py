#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Common loop F1 evaluator for Lee / Nagano.

Nagano:
  --dataset nagano --ref_path /path/to/nagano_ref_loop.bedpe

Lee:
  --dataset lee --ref_dir /path/to/lee_dir_e ref_MG_ODC_Neuron_bedpe
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.loop_f1 import (
    LoopEvalConfig,
    evaluate_pred_dir,
    make_lee_ref_provider,
    make_nagano_ref_provider,
)


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", required=True, choices=["lee", "nagano"])
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--out_csv", required=True)

    # common params
    p.add_argument("--resolution", type=int, default=10_000)
    p.add_argument("--min_bin_dist", type=int, default=1)
    p.add_argument("--max_bin_dist", type=int, default=100)
    p.add_argument("--allowed_slack_bp", type=int, default=20_000)

    # ref inputs (choose one by dataset)
    p.add_argument("--ref_dir", default=None, help="Lee: directory with MG_chr.bedpe / ODC_chr.bedpe / Neuron_chr.bedpe")
    p.add_argument("--ref_path", default=None, help="Nagano: single reference bedpe path")

    args = p.parse_args()

    cfg = LoopEvalConfig(
        resolution=args.resolution,
        min_bin_dist=args.min_bin_dist,
        max_bin_dist=args.max_bin_dist,
        allowed_slack_bp=args.allowed_slack_bp,
    )

    if args.dataset == "lee":
        if args.ref_dir is None:
            raise ValueError("Lee requires --ref_dir")
        ref_provider = make_lee_ref_provider(args.ref_dir, cfg)

    else:  # nagano
        if args.ref_path is None:
            raise ValueError("Nagano requires --ref_path")
        ref_provider = make_nagano_ref_provider(args.ref_path, cfg)

    df = evaluate_pred_dir(
        pred_dir=args.pred_dir,
        cfg=cfg,
        ref_provider=ref_provider,
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    if df.empty:
        print("[WARN] No outputs. Check pred_dir or reference paths.")
    else:
        print(f"[OK] Saved: {out_csv} (n={len(df)})")
        # quick summary
        print(df[["Precision", "Recall", "F1"]].mean().to_string())


if __name__ == "__main__":
    main()

