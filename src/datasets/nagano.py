from __future__ import annotations

from typing import Optional, Sequence

from .labels import LabelMap, load_unified_labels
from .parsers import parse_pt_name_nagano
from .pt_dataset import GraphDatasetConfig, PTGraphDataset


class NaganoDataset(PTGraphDataset):
    """
    Nagano dataset wrapper.

    Filenames may contain '.' in cell_id. This wrapper normalizes '.' -> '_'
    to match unified label table.
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
            name_parser=parse_pt_name_nagano,
            labels=labels,
            config=GraphDatasetConfig(
                canonicalize_edges=canonicalize_edges,
                attach_labels=attach,
            ),
            file_list=file_list,
            manifest_path=manifest_path,
        )

