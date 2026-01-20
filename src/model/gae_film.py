# src/model/gae_film.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GlobalAttention

from .encoder_gcn import GCNEncoder


# ------------------------------
# Config
# ------------------------------
@dataclass
class GAEFiLMConfig:
    in_dim: int
    hidden_dim: int = 128
    z_dim: int = 32
    dropout: float = 0.2

    # recon decoder
    recon_dec_hidden: int = 64

    # cosine alignment (global projection)
    target_dim: int = 50
    proj_hidden: int = 64

    # FiLM MLP
    film_hidden: int = 64

    # loop head (optional)
    enable_loop_head: bool = False
    loop_dim: int = 32
    loop_proj_hidden: int = 64
    loop_dec_hidden: int = 64


# ------------------------------
# Decoders
# ------------------------------
class MLPDecoder(nn.Module):
    """Edge-wise MLP decoder: logits = MLP([z_i, z_j])."""

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


# ------------------------------
# Main model (same idea as your code)
# ------------------------------
class GAEFiLM(nn.Module):
    """
    GAE + FiLM global modulation + (optional) loop head.

    - encode(): produce z_mod (node embeddings) and g (graph embedding)
    - decode_recon(): edge logits for reconstruction (using recon decoder)
    - project_global(): g -> target_dim (for cosine alignment)
    - decode_loop(): edge logits for loop calling (only if enabled)
    """

    def __init__(self, cfg: GAEFiLMConfig):
        super().__init__()
        self.cfg = cfg

        # encoder (same as your 3-layer GCN)
        self.encoder = GCNEncoder(
            in_dim=cfg.in_dim,
            hidden_dim=cfg.hidden_dim,
            z_dim=cfg.z_dim,
            dropout=cfg.dropout,
        )

        # reconstruction head (same as your main_dec)
        self.recon_dec = MLPDecoder(cfg.z_dim, hidden_dim=cfg.recon_dec_hidden)

        # global pooling (same as GlobalAttention(nn.Linear(z_dim, 1)))
        self.attn = GlobalAttention(gate_nn=nn.Linear(cfg.z_dim, 1))

        # FiLM: g -> (gamma, beta)
        self.film = nn.Sequential(
            nn.Linear(cfg.z_dim, cfg.film_hidden),
            nn.ReLU(),
            nn.Linear(cfg.film_hidden, cfg.z_dim * 2),
        )

        # global projection: g -> target_dim (for cosine alignment)
        self.proj = nn.Sequential(
            nn.Linear(cfg.z_dim, cfg.proj_hidden),
            nn.ReLU(),
            nn.Linear(cfg.proj_hidden, cfg.target_dim),
        )

        # optional loop head
        self.enable_loop_head = bool(cfg.enable_loop_head)
        if self.enable_loop_head:
            self.loop_proj = nn.Sequential(
                nn.Linear(cfg.z_dim, cfg.loop_proj_hidden),
                nn.ReLU(),
                nn.Linear(cfg.loop_proj_hidden, cfg.loop_dim),
            )
            self.loop_dec = MLPDecoder(cfg.loop_dim, hidden_dim=cfg.loop_dec_hidden)
        else:
            self.loop_proj = None
            self.loop_dec = None

    # ---------- core ----------
    def encode(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
          x: (N, in_dim)
          edge_index: (2, E)
          batch: (N,) graph id per node (batch_size=1이면 전부 0)

        Returns:
          z_mod: (N, z_dim)
          g: (B, z_dim)
        """
        z_local = self.encoder(x, edge_index)         # (N, z_dim)
        g = self.attn(z_local, batch=batch)           # (B, z_dim)

        gamma, beta = self.film(g).chunk(2, dim=-1)   # each (B, z_dim)
        gamma_nodes = gamma[batch]                    # (N, z_dim)
        beta_nodes  = beta[batch]                     # (N, z_dim)

        z_mod = z_local * (1.0 + gamma_nodes) + beta_nodes
        return z_mod, g

    # ---------- heads ----------
    def decode_recon(self, z_mod: Tensor, edge_index: Tensor) -> Tensor:
        """Reconstruction edge logits."""
        return self.recon_dec(z_mod, edge_index)

    def project_global(self, g: Tensor) -> Tensor:
        """Global embedding projection for cosine alignment."""
        return self.proj(g)

    def decode_loop(self, z_mod: Tensor, edge_index: Tensor) -> Tensor:
        """
        Loop edge logits (requires loop head enabled).
        """
        if not self.enable_loop_head or self.loop_proj is None or self.loop_dec is None:
            raise RuntimeError("Loop head is disabled. Set enable_loop_head=True in config.")
        z_loop = self.loop_proj(z_mod)
        return self.loop_dec(z_loop, edge_index)

    # ---------- utilities ----------
    def add_loop_head(self, loop_dim: int = 32, proj_hidden: int = 64, dec_hidden: int = 64) -> None:
        """
        Attach loop head after creating a pretrained model instance.
        Useful when you load a pretrained recon model and then fine-tune with loop loss.
        """
        if self.enable_loop_head:
            return
        self.enable_loop_head = True
        self.cfg.enable_loop_head = True
        self.cfg.loop_dim = loop_dim
        self.cfg.loop_proj_hidden = proj_hidden
        self.cfg.loop_dec_hidden = dec_hidden

        self.loop_proj = nn.Sequential(
            nn.Linear(self.cfg.z_dim, proj_hidden),
            nn.ReLU(),
            nn.Linear(proj_hidden, loop_dim),
        )
        self.loop_dec = MLPDecoder(loop_dim, hidden_dim=dec_hidden)

    def load_pretrained(self, state_dict: Dict[str, Any], strict: bool = False) -> torch.nn.modules.module._IncompatibleKeys:
        """
        Load pretrained weights. For fine-tuning with newly attached loop head,
        use strict=False to ignore missing keys for loop layers.
        """
        return self.load_state_dict(state_dict, strict=strict)

