from __future__ import annotations

from torch_geometric.data import Data


def canonicalize_undirected_edges(data: Data) -> Data:
    """
    Canonicalize the edge representation of a graph.

    Enforces a unique undirected edge list by:
    1) removing self-loops (i == j),
    2) retaining only one direction for each undirected edge (i < j),
    while keeping edge attributes aligned if present.
    """
    edge_index = getattr(data, "edge_index", None)
    if edge_index is None or edge_index.numel() == 0:
        return data

    # 1) remove self-loops
    mask_no_self = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask_no_self]

    # 2) keep canonical undirected form (i < j)
    i, j = edge_index
    mask_canonical = i < j
    edge_index = edge_index[:, mask_canonical]
    data.edge_index = edge_index

    # align edge_attr if present
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        edge_attr = data.edge_attr
        edge_attr = edge_attr[mask_no_self]
        edge_attr = edge_attr[mask_canonical]
        data.edge_attr = edge_attr

    return data

