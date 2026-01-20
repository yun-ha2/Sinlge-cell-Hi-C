# src/model/encoder_gcn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv


@dataclass
class EncoderConfig:
    in_dim: int
    hidden_dim: int = 128
    z_dim: int = 32
    dropout: float = 0.2
    use_layernorm: bool = True


class GCNEncoder(nn.Module):
    """
    Simple 3-layer GCN encoder for node embeddings z.
    - x: (N, in_dim)
    - edge_index: (2, E)
    returns:
    - z: (N, z_dim)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        z_dim: int = 32,
        dropout: float = 0.2,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.use_layernorm = use_layernorm
        self.norm = nn.LayerNorm(in_dim) if use_layernorm else nn.Identity()

        self.c1 = GCNConv(in_dim, hidden_dim)
        self.c2 = GCNConv(hidden_dim, hidden_dim)
        self.out = GCNConv(hidden_dim, z_dim)

        self.dropout = float(dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.norm(x)
        x = F.relu(self.c1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.c2(x, edge_index))
        z = self.out(x, edge_index)
        return z


def build_gcn_encoder(cfg: EncoderConfig) -> GCNEncoder:
    return GCNEncoder(
        in_dim=cfg.in_dim,
        hidden_dim=cfg.hidden_dim,
        z_dim=cfg.z_dim,
        dropout=cfg.dropout,
        use_layernorm=cfg.use_layernorm,
    )

