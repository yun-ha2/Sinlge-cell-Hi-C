# src/eval/clustering.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ClusteringMetrics:
    n_cells: int
    n_classes: int
    ari: float
    nmi: float
    ami: float
    homogeneity: float
    completeness: float
    v_measure: float
    silhouette: float
    hungarian: float


def load_label_maps(dataset: str, label_path: str) -> Tuple[Dict[str, str], Dict[str, int], Dict[int, str]]:
    """
    Returns
    -------
    cell_to_type: cell_id -> type(str)
    type_to_idx: type(str) -> int
    idx_to_type: int -> type(str)

    Assumptions
    ----------
    - lee: columns {'file_name','cell_type'}
    - nagano: columns {'cell_nm','cell_type'}
    """
    df = pd.read_csv(label_path, sep="\t")
    dataset = dataset.lower()

    if dataset == "lee":
        need = {"file_name", "cell_type"}
        if not need.issubset(df.columns):
            raise ValueError(f"Lee label file must contain columns {need}, got={set(df.columns)}")
        cell_to_type = dict(zip(df["file_name"].astype(str), df["cell_type"].astype(str)))

    elif dataset == "nagano":
        need = {"cell_nm", "cell_type"}
        if not need.issubset(df.columns):
            raise ValueError(f"Nagano label file must contain columns {need}, got={set(df.columns)}")
        cell_to_type = dict(zip(df["cell_nm"].astype(str), df["cell_type"].astype(str)))

    else:
        raise ValueError("dataset must be 'lee' or 'nagano'")

    types = sorted(pd.Series(list(cell_to_type.values())).unique().tolist())
    type_to_idx = {t: i for i, t in enumerate(types)}
    idx_to_type = {i: t for t, i in type_to_idx.items()}
    return cell_to_type, type_to_idx, idx_to_type


def align_embeddings_to_labels(
    X: np.ndarray,
    cell_ids: List[str],
    cell_to_type: Dict[str, str],
    type_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Filter/reorder embeddings to those that have labels.

    Returns
    -------
    X_use: (M, D)
    y_true: (M,)
    used_ids: length M
    used_types: length M
    """
    cid_to_row = {cid: i for i, cid in enumerate(cell_ids)}
    used_ids: List[str] = []
    used_types: List[str] = []
    y_true: List[int] = []
    rows: List[np.ndarray] = []

    for cid in cell_ids:
        if cid not in cell_to_type:
            continue
        t = cell_to_type[cid]
        used_ids.append(cid)
        used_types.append(t)
        y_true.append(type_to_idx[t])
        rows.append(X[cid_to_row[cid]])

    if len(rows) == 0:
        raise RuntimeError("No matched labels. Check cell_id naming vs label table keys.")

    X_use = np.vstack(rows).astype(np.float32)
    y_true = np.asarray(y_true, dtype=np.int64)
    return X_use, y_true, used_ids, used_types


def hungarian_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    for c in range(n_classes):
        for t in range(n_classes):
            conf[c, t] = np.sum((y_pred == c) & (y_true == t))
    r, c = linear_sum_assignment(-conf)
    return float(conf[r, c].sum() / max(1, len(y_true)))


def evaluate_kmeans(
    X: np.ndarray,
    y_true: np.ndarray,
    n_classes: int,
    *,
    random_state: int = 42,
) -> Tuple[np.ndarray, ClusteringMetrics, np.ndarray]:
    """
    Standardize -> KMeans -> compute metrics.

    Returns
    -------
    y_pred: (M,)
    metrics: ClusteringMetrics
    X_scaled: (M, D)
    """
    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=n_classes, random_state=random_state, n_init="auto")
    y_pred = km.fit_predict(X_scaled)

    ari = float(adjusted_rand_score(y_true, y_pred))
    nmi = float(normalized_mutual_info_score(y_true, y_pred))
    ami = float(adjusted_mutual_info_score(y_true, y_pred))
    homo = float(homogeneity_score(y_true, y_pred))
    comp = float(completeness_score(y_true, y_pred))
    vmeas = float(v_measure_score(y_true, y_pred))
    sil = float(silhouette_score(X_scaled, y_pred)) if len(np.unique(y_pred)) > 1 else float("nan")
    hung = float(hungarian_accuracy(y_true, y_pred, n_classes))

    metrics = ClusteringMetrics(
        n_cells=int(len(y_true)),
        n_classes=int(n_classes),
        ari=ari,
        nmi=nmi,
        ami=ami,
        homogeneity=homo,
        completeness=comp,
        v_measure=vmeas,
        silhouette=sil,
        hungarian=hung,
    )
    return y_pred, metrics, X_scaled


def build_embeddings_table(
    X: np.ndarray,
    cell_ids: List[str],
    cell_types: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "true_type": cell_types,
            "true_label": y_true.astype(int),
            "cluster": y_pred.astype(int),
        }
    )
    for j in range(X.shape[1]):
        df[f"z{j+1}"] = X[:, j]
    return df


def metrics_to_frame(metrics: ClusteringMetrics, extra: Dict[str, str] | None = None) -> pd.DataFrame:
    row = {
        "n_cells": metrics.n_cells,
        "n_classes": metrics.n_classes,
        "ARI": metrics.ari,
        "NMI": metrics.nmi,
        "AMI": metrics.ami,
        "Homogeneity": metrics.homogeneity,
        "Completeness": metrics.completeness,
        "Vmeasure": metrics.v_measure,
        "Silhouette": metrics.silhouette,
        "Hungarian": metrics.hungarian,
    }
    if extra:
        row.update(extra)
    return pd.DataFrame([row])

