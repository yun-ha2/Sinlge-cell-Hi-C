# src/model/ref_loops.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Set, Optional

import torch


def _to_str(x) -> str:
    if isinstance(x, (list, tuple)):
        x = x[0]
    if torch.is_tensor(x):
        x = x.item()
    return str(x).strip()


def normalize_chr(k: str) -> str:
    k = _to_str(k)
    if not k.startswith("chr"):
        k = "chr" + k.replace("chr", "")
    return k


def filter_by_num_nodes(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index
    m = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
    return edge_index[:, m]


@dataclass
class SamplerArgs:
    neg_pos_ratio: float = 1.0
    near_lower: int = 5
    near_upper: int = 100
    near_ratio: float = 0.7
    oversample_factor: int = 3
    max_rounds: int = 4


class BaseRefLoopProvider:
    """Base interface."""

    def __init__(self, ref_loop_path: str, device: torch.device, val_neg_field: str = "ref_neg_edge_index"):
        self.ref_loop_path = ref_loop_path
        self.device = device
        self.val_neg_field = val_neg_field

        self._gpu_cache: Dict[str, torch.Tensor] = {}

    def get_pos(self, data) -> torch.Tensor:
        raise NotImplementedError

    def get_neg(self, data, pos: torch.Tensor, sampler_fn, sargs: SamplerArgs, split: str) -> torch.Tensor:
        split = split.lower()

        # val/test: use saved negatives if present
        if split != "train" and hasattr(data, self.val_neg_field):
            neg = getattr(data, self.val_neg_field)
            if neg is None:
                return torch.empty((2, 0), dtype=torch.long, device=self.device)
            neg = neg.to(self.device)
            return filter_by_num_nodes(neg, int(data.num_nodes))

        # fallback sampling
        if pos.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)

        num_neg = int(pos.size(1) * sargs.neg_pos_ratio)
        neg = sampler_fn(
            pos_edge_index=pos,
            num_nodes=int(data.num_nodes),
            num_neg=num_neg,
            device=self.device,
            near_lower=sargs.near_lower,
            near_upper=sargs.near_upper,
            near_ratio=sargs.near_ratio,
            oversample_factor=sargs.oversample_factor,
            max_rounds=sargs.max_rounds,
        )
        return filter_by_num_nodes(neg, int(data.num_nodes))


class NaganoRefLoopProvider(BaseRefLoopProvider):
    """
    ref_loop_path: torch file containing dict(chr -> edge_index)
    """

    def __init__(self, ref_loop_path: str, device: torch.device, val_neg_field: str = "ref_neg_edge_index"):
        super().__init__(ref_loop_path, device, val_neg_field)
        obj = torch.load(ref_loop_path, map_location="cpu")
        if not isinstance(obj, dict):
            raise ValueError("Nagano ref loop must be dict(chr -> edge_index)")
        self.ref_cpu = {normalize_chr(k): v for k, v in obj.items()}

    def get_pos(self, data) -> torch.Tensor:
        chrom = normalize_chr(getattr(data, "chrom", ""))
        num_nodes = int(data.num_nodes)

        if chrom not in self.ref_cpu:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)

        key = f"nagano::{chrom}"
        if key not in self._gpu_cache:
            self._gpu_cache[key] = self.ref_cpu[chrom].long().to(self.device)

        pos = self._gpu_cache[key]
        return filter_by_num_nodes(pos, num_nodes)


class LeeRefLoopProvider(BaseRefLoopProvider):
    """
    ref_loop_path: torch file containing bundle:
      {"MG": {chr->edge}, "ODC": {...}, "Neuron": {...}}
    """

    def __init__(
        self,
        ref_loop_path: str,
        device: torch.device,
        neuron_types: Optional[Set[str]] = None,
        val_neg_field: str = "ref_neg_edge_index",
    ):
        super().__init__(ref_loop_path, device, val_neg_field)

        if neuron_types is None:
            neuron_types = {"Sst", "L23", "L4", "L5", "L6", "Vip", "Pvalb", "Ndnf"}
        self.neuron_types = set(neuron_types)

        obj = torch.load(ref_loop_path, map_location="cpu")
        if not isinstance(obj, dict):
            raise ValueError("Lee bundle must be dict(group -> dict(chr->edge_index))")

        bundle: Dict[str, Dict[str, torch.Tensor]] = {}
        for group, d in obj.items():
            if not isinstance(d, dict):
                raise ValueError(f"Lee bundle group {group} must map to dict(chr->edge_index)")
            bundle[group] = {normalize_chr(k): v for k, v in d.items()}
        self.bundle_cpu = bundle

    def _group(self, cell_type: str) -> str:
        if cell_type in self.neuron_types:
            return "Neuron"
        if cell_type == "MG":
            return "MG"
        if cell_type == "ODC":
            return "ODC"
        return "UNKNOWN"

    def get_pos(self, data) -> torch.Tensor:
        cell_type = _to_str(getattr(data, "cell_type", "UNKNOWN"))
        group = self._group(cell_type)

        if group == "UNKNOWN":
            return torch.empty((2, 0), dtype=torch.long, device=self.device)

        chrom = normalize_chr(getattr(data, "chrom", ""))
        num_nodes = int(data.num_nodes)

        d = self.bundle_cpu.get(group, {})
        if chrom not in d:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)

        key = f"lee::{group}::{chrom}"
        if key not in self._gpu_cache:
            self._gpu_cache[key] = d[chrom].long().to(self.device)

        pos = self._gpu_cache[key]
        return filter_by_num_nodes(pos, num_nodes)


def build_ref_loop_provider(
    dataset: str,
    ref_loop_path: str,
    device: torch.device,
    neuron_types: Optional[Set[str]] = None,
    val_neg_field: str = "ref_neg_edge_index",
) -> BaseRefLoopProvider:
    dataset = dataset.lower().strip()
    if dataset == "nagano":
        return NaganoRefLoopProvider(ref_loop_path, device, val_neg_field=val_neg_field)
    if dataset == "lee":
        return LeeRefLoopProvider(ref_loop_path, device, neuron_types=neuron_types, val_neg_field=val_neg_field)
    raise ValueError(f"Unknown dataset: {dataset}")

