from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import torch
from torch_geometric.data import Dataset, Data

from .labels import LabelMap
from .parsers import ParsedName
from .transforms import canonicalize_undirected_edges


def list_pt_files(data_dir: str) -> List[str]:
    """List *.pt basenames in a directory."""
    return sorted([p.name for p in Path(data_dir).iterdir() if p.is_file() and p.name.endswith(".pt")])


def read_manifest(path: str) -> List[str]:
    """
    Read a manifest file listing one .pt filename per line.

    - empty lines are ignored
    - lines starting with '#' are ignored
    """
    lines = Path(path).read_text().splitlines()
    out: List[str] = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        out.append(ln)
    return out


@dataclass(frozen=True)
class GraphDatasetConfig:
    """
    Parameters
    ----------
    canonicalize_edges : bool
        Apply canonical undirected edge representation (self-loop removal + i<j).
    attach_labels : bool
        Attach (y, cell_type) from LabelMap.
    unknown_label : int
        Used when cell_id not found in label table.
    """
    canonicalize_edges: bool = True
    attach_labels: bool = True
    unknown_label: int = -1


class PTGraphDataset(Dataset):
    """
    Dataset for per-(cell, chrom) PyG graphs stored as *.pt.

    Features
    --------
    - Loads Data objects from disk
    - Parses filename into (cell_id, chrom)
    - Optional: canonicalizes edges (remove self-loops; keep i<j)
    - Optional: attaches labels (y, cell_type) via unified label table
    - Supports subset loading via file_list or manifest_path
    """

    def __init__(
        self,
        data_dir: str,
        *,
        name_parser: Callable[[str], ParsedName],
        labels: Optional[LabelMap] = None,
        config: GraphDatasetConfig = GraphDatasetConfig(),
        file_list: Optional[Sequence[str]] = None,
        manifest_path: Optional[str] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.name_parser = name_parser
        self.labels = labels
        self.config = config

        if config.attach_labels and labels is None:
            raise ValueError("attach_labels=True but labels=None. Provide label_path or set attach_labels=False.")

        if file_list is not None and manifest_path is not None:
            raise ValueError("Provide only one of file_list or manifest_path (not both).")

        if manifest_path is not None:
            self.files = sorted(read_manifest(manifest_path))
        elif file_list is not None:
            self.files = sorted(list(file_list))
        else:
            self.files = list_pt_files(data_dir)

        if len(self.files) == 0:
            raise RuntimeError(f"No .pt files found/selected in: {data_dir}")

        # validate existence early (debug-friendly)
        missing = [f for f in self.files if not (Path(data_dir) / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Some files are missing under {data_dir}. Example: {missing[:5]}"
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        fname = self.files[idx]
        parsed = self.name_parser(fname)

        path = os.path.join(self.data_dir, fname)
        data: Data = torch.load(path)

        # always attach metadata
        data.cell_id = parsed.cell_id
        data.chrom = parsed.chrom

        # optional edge canonicalization
        if self.config.canonicalize_edges:
            data = canonicalize_undirected_edges(data)

        # optional label attachment
        if self.config.attach_labels and self.labels is not None:
            cid = parsed.cell_id
            if cid in self.labels.cell_id_to_y:
                data.y = torch.tensor([self.labels.cell_id_to_y[cid]], dtype=torch.long)
                data.cell_type = self.labels.cell_id_to_type[cid]
            else:
                data.y = torch.tensor([self.config.unknown_label], dtype=torch.long)
                data.cell_type = "unknown"

        return data

