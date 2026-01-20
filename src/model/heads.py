# src/model/heads.py
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MLPDecoder(nn.Module):
    """
    Edge-wise MLP decoder.
    logits = MLP([z_i, z_j])

    Args:
      z_dim: dimension of node embeddings
      hidden_dim: hidden size of MLP
    """
    def __init__(self, z_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(z_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, z: Tensor, edge_index: Tensor) -> Tensor:
        zi = z[edge_index[0]]
        zj = z[edge_index[1]]
        h = torch.cat([zi, zj], dim=-1)
        h = F.relu(self.fc1(h))
        return self.fc2(h).squeeze(-1)


@dataclass
class LoopHeadConfig:
    """
    Loop head config to be used if you want a separate loop head module.

    NOTE:
      In your current gae_film.py, loop head is implemented inside GAEFiLM
      via `add_loop_head()` and `decode_loop()`.
      This module is optional and useful if you prefer explicit head objects.
    """
    in_z_dim: int = 32
    loop_dim: int = 32
    proj_hidden: int = 64
    dec_hidden: int = 64


class LoopHead(nn.Module):
    """
    Standalone loop head:
      z_mod -> loop_proj -> z_loop -> loop_dec(edge logits)
    """
    def __init__(self, cfg: LoopHeadConfig):
        super().__init__()
        self.loop_proj = nn.Sequential(
            nn.Linear(cfg.in_z_dim, cfg.proj_hidden),
            nn.ReLU(),
            nn.Linear(cfg.proj_hidden, cfg.loop_dim),
        )
        self.loop_dec = MLPDecoder(cfg.loop_dim, hidden_dim=cfg.dec_hidden)

    def forward(self, z_mod: Tensor, edge_index: Tensor) -> Tensor:
        z_loop = self.loop_proj(z_mod)
        return self.loop_dec(z_loop, edge_index)

