from __future__ import annotations

from typing import Optional, Sequence

from .labels import LabelMap, load_unified_labels
from .parsers import parse_pt_name_default
from .pt_dataset import GraphDatasetConfig, PTGraphDataset


class LeeDataset(PTGraphDataset):
    """
    Lee dataset wrapper.

    Assumes filenames: <cell_id>_chr<k>.pt
    Uses unified labels: cell_id, cell_type, total_contacts
    """

    def __init__(
        self,
        data_dir: str,
        *,
        label_path: Optional[str] = None,
        canonicalize_edges: bool = True,
        file_list: Optional[Sequence[str]] = None,
        manifest_path: Optional[str] = None,
    ):
        labels: Optional[LabelMap] = None
        attach = False
        if label_path is not None:
            labels = load_unified_labels(label_path)
            attach = True

        super().__init__(
            data_dir=data_dir,
            name_parser=parse_pt_name_default,
            labels=labels,
            config=GraphDatasetConfig(
                canonicalize_edges=canonicalize_edges,
                attach_labels=attach,
            ),
            file_list=file_list,
            manifest_path=manifest_path,
        )

