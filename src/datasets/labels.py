from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass(frozen=True)
class LabelMap:
    """
    Label lookup tables derived from a unified label file.

    Attributes
    ----------
    cell_id_to_type : dict
        cell_id -> cell_type (string)
    cell_id_to_y : dict
        cell_id -> numeric label (int)
    type_to_y : dict
        cell_type -> numeric label (int)
    """
    cell_id_to_type: Dict[str, str]
    cell_id_to_y: Dict[str, int]
    type_to_y: Dict[str, int]


def load_unified_labels(label_path: str, *, sep: str = "\t") -> LabelMap:
    """
    Load a unified label table with columns:
      - cell_id
      - cell_type
      - total_contacts (optional; ignored by this loader)

    Returns
    -------
    LabelMap
    """
    df = pd.read_csv(label_path, sep=sep)

    required = {"cell_id", "cell_type"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Label file must contain columns {sorted(required)}. "
            f"Got {list(df.columns)} in {label_path}"
        )

    df["cell_id"] = df["cell_id"].astype(str)
    df["cell_type"] = df["cell_type"].astype(str)

    uniq_types = sorted(df["cell_type"].unique())
    type_to_y = {ct: i for i, ct in enumerate(uniq_types)}

    cell_id_to_type = dict(zip(df["cell_id"], df["cell_type"]))
    cell_id_to_y = {cid: type_to_y[ct] for cid, ct in cell_id_to_type.items()}

    return LabelMap(
        cell_id_to_type=cell_id_to_type,
        cell_id_to_y=cell_id_to_y,
        type_to_y=type_to_y,
    )

