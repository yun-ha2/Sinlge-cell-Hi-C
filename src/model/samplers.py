# src/model/samplers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class MixedNegSamplerConfig:
    """
    Mixed negative sampling for sparse Hi-C graphs.

    - num_neg = num_pos * neg_pos_ratio
    - near negatives sampled from pairs (i, i+k), k in [near_lower, near_upper]
    - random negatives sampled uniformly from all pairs
    - near_ratio controls near vs random composition
    """
    neg_pos_ratio: float = 1.0
    near_lower: int = 5
    near_upper: int = 100
    near_ratio: float = 0.7
    oversample_factor: int = 3
    max_rounds: int = 4


def _normalize_pairs(p: Tensor) -> Tensor:
    """Ensure pairs are ordered (i<j). p: (2, E)"""
    return torch.stack([torch.minimum(p[0], p[1]), torch.maximum(p[0], p[1])], dim=0)


def _keys_from_pairs(p: Tensor, N: int) -> Tensor:
    """Map (i,j) -> unique key = i*N + j (assumes i<j)."""
    p = _normalize_pairs(p)
    return p[0] * N + p[1]


def _pairs_from_keys(keys: Tensor, N: int) -> Tensor:
    """Inverse mapping keys -> (i,j)."""
    return torch.stack([keys // N, keys % N], dim=0)


@torch.no_grad()
def sample_negatives_mixed(
    pos_edge_index: Tensor,
    num_nodes: int,
    num_neg: int,
    device: torch.device,
    near_lower: int = 5,
    near_upper: int = 100,
    near_ratio: float = 0.7,
    oversample_factor: int = 3,
    max_rounds: int = 4,
) -> Tensor:
    """
    Sample negative edges (i<j) that do NOT overlap with existing positive edges.

    Args:
      pos_edge_index: (2, Epos)
      num_nodes: N
      num_neg: number of negatives to sample
      device: return tensor device
      near_lower/upper/ratio: near negative sampling policy
      oversample_factor/max_rounds: rejection sampling controls

    Returns:
      neg_edge_index: (2, num_neg), i<j
    """
    N = int(num_nodes)
    if N <= 1 or num_neg <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    # existing edge keys
    pos_norm = _normalize_pairs(pos_edge_index.detach().cpu())
    exist = torch.unique(_keys_from_pairs(pos_norm, N))

    selected = torch.empty(0, dtype=torch.long)  # keys
    near_quota = int(num_neg * float(near_ratio))

    def add(keys: Tensor) -> None:
        nonlocal selected
        keys = torch.unique(keys)
        mask = (~torch.isin(keys, exist)) & (~torch.isin(keys, selected))
        if mask.any():
            selected = torch.unique(torch.cat([selected, keys[mask]]))

    # ---- near sampling ----
    need = near_quota
    tries = max(1000, near_quota * int(oversample_factor))
    for _ in range(int(max_rounds)):
        if need <= 0:
            break

        k = torch.randint(int(near_lower), min(int(near_upper), N - 1) + 1, (tries,))
        i = torch.randint(0, N, (tries,))
        j = i + k
        mask = (i < j) & (j < N)
        if mask.any():
            keys = _keys_from_pairs(torch.stack([i[mask], j[mask]]), N)
            add(keys)

        need = near_quota - selected.numel()
        tries *= 2

    # ---- random sampling ----
    need = num_neg - selected.numel()
    tries = max(1000, num_neg * int(oversample_factor))
    for _ in range(int(max_rounds)):
        if need <= 0:
            break

        i = torch.randint(0, N, (tries,))
        j = torch.randint(0, N, (tries,))
        mask = i < j
        if mask.any():
            keys = _keys_from_pairs(torch.stack([i[mask], j[mask]]), N)
            add(keys)

        need = num_neg - selected.numel()
        tries *= 2

    # trim if oversampled
    if selected.numel() > num_neg:
        selected = selected[torch.randperm(selected.numel())[:num_neg]]

    neg_edge_index = _pairs_from_keys(selected, N).to(device)
    return neg_edge_index


@torch.no_grad()
def sample_negatives_for_recon(
    pos_edge_index: Tensor,
    num_nodes: int,
    cfg: MixedNegSamplerConfig,
    device: torch.device,
) -> Tensor:
    """
    Convenience wrapper for reconstruction negatives:
      num_neg = num_pos * cfg.neg_pos_ratio
    """
    num_pos = int(pos_edge_index.size(1))
    num_neg = int(num_pos * float(cfg.neg_pos_ratio))
    return sample_negatives_mixed(
        pos_edge_index=pos_edge_index,
        num_nodes=num_nodes,
        num_neg=num_neg,
        device=device,
        near_lower=cfg.near_lower,
        near_upper=cfg.near_upper,
        near_ratio=cfg.near_ratio,
        oversample_factor=cfg.oversample_factor,
        max_rounds=cfg.max_rounds,
    )

