# src/model/losses.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# default loss fn (logits)
_BCE = nn.BCEWithLogitsLoss()


@dataclass
class LossWeights:
    """
    Weights for combined objectives.
    Use the same scheme as your training scripts.
    """
    recon: float = 1.0
    cosine: float = 1.0
    loop: float = 0.0  # pretrain: 0, finetune: >0


def bce_pos_neg_from_logits(
    pos_logits: Tensor,
    neg_logits: Tensor,
    bce: Optional[nn.Module] = None,
) -> Dict[str, Tensor]:
    """
    Compute BCE losses for positive/negative logits.
    Returns dict of tensors: pos, neg, total.
    """
    loss_fn = _BCE if bce is None else bce
    pos_loss = loss_fn(pos_logits, torch.ones_like(pos_logits))
    neg_loss = loss_fn(neg_logits, torch.zeros_like(neg_logits))
    return {"pos": pos_loss, "neg": neg_loss, "total": pos_loss + neg_loss}


def recon_losses(
    model,                 # expects GAEFiLM (gae_film.py)
    z_mod: Tensor,
    pos_edge_index: Tensor,
    neg_edge_index: Tensor,
    bce: Optional[nn.Module] = None,
) -> Dict[str, Tensor]:
    """
    Reconstruction loss using model.decode_recon().
    """
    pos_logits = model.decode_recon(z_mod, pos_edge_index)
    neg_logits = model.decode_recon(z_mod, neg_edge_index)
    out = bce_pos_neg_from_logits(pos_logits, neg_logits, bce=bce)
    return {"recon_pos": out["pos"], "recon_neg": out["neg"], "recon_total": out["total"]}


def cosine_loss(
    pred: Tensor,
    target: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """
    1 - cosine_similarity, averaged over batch.
    pred/target: (D,) or (B,D)
    """
    cos = F.cosine_similarity(pred, target, dim=-1, eps=eps)
    return (1.0 - cos).mean()


def cosine_alignment_loss(
    model,                 # expects GAEFiLM
    g: Tensor,             # (B, z_dim)
    target_global: Optional[Tensor],
    enabled: bool = True,
) -> Tensor:
    """
    Cosine alignment loss using model.project_global(g).
    If target_global is None or enabled=False -> 0.
    """
    if (not enabled) or (target_global is None):
        return torch.tensor(0.0, device=g.device)

    pred = model.project_global(g)  # (B, target_dim)
    # target_global should be (B, target_dim) or (target_dim,) when B=1
    return cosine_loss(pred, target_global.to(g.device))


def loop_losses(
    model,                 # expects GAEFiLM with loop head enabled
    z_mod: Tensor,
    pos_loop_edge_index: Tensor,
    neg_loop_edge_index: Tensor,
    bce: Optional[nn.Module] = None,
) -> Dict[str, Tensor]:
    """
    Loop loss using model.decode_loop().
    """
    pos_logits = model.decode_loop(z_mod, pos_loop_edge_index)
    neg_logits = model.decode_loop(z_mod, neg_loop_edge_index)
    out = bce_pos_neg_from_logits(pos_logits, neg_logits, bce=bce)
    return {"loop_pos": out["pos"], "loop_neg": out["neg"], "loop_total": out["total"]}


def total_loss(
    model,
    z_mod: Tensor,
    g: Tensor,
    pos_edge_index: Tensor,
    neg_edge_index: Tensor,
    weights: LossWeights,
    target_global: Optional[Tensor] = None,
    cosine_enabled: bool = True,
    # loop parts (optional)
    pos_loop_edge_index: Optional[Tensor] = None,
    neg_loop_edge_index: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """
    Unified loss calculator for:
      - Pretrain: recon + cosine
      - Fine-tune: recon + cosine + loop

    Returns dict with components and 'total'.
    """
    out: Dict[str, Tensor] = {}

    # recon
    rec = recon_losses(model, z_mod, pos_edge_index, neg_edge_index)
    out.update(rec)

    # cosine
    cos = cosine_alignment_loss(model, g, target_global, enabled=cosine_enabled)
    out["cosine"] = cos

    # loop (only if provided and weight>0)
    loop_total = torch.tensor(0.0, device=z_mod.device)
    if weights.loop > 0.0 and pos_loop_edge_index is not None and neg_loop_edge_index is not None:
        lp = loop_losses(model, z_mod, pos_loop_edge_index, neg_loop_edge_index)
        out.update(lp)
        loop_total = out["loop_total"]

    # combine
    out["total"] = (
        weights.recon * out["recon_total"]
        + weights.cosine * out["cosine"]
        + weights.loop * loop_total
    )
    return out

