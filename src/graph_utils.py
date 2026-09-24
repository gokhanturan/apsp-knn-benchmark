from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

DEFAULT_K_VALUES = (5, 10, 20)
SCALING_SIZES = (100, 150, 200, 250, 300, 400, 500, 750, 1000, 1250, 1500, 1797)
SCALING_SEEDS = (11, 23, 37, 53, 71)


def load_csv_dataset(path: str | Path):
    df = pd.read_csv(path)
    if "target" not in df.columns:
        raise ValueError(f"Expected a 'target' column in {path}")
    X = df.drop(columns="target").to_numpy(dtype=float)
    y = df["target"].to_numpy()
    return X, y


def build_union_knn_graph(X: np.ndarray, k: int) -> csr_matrix:
    """Build the exact positive-weight undirected union-kNN graph used in the study."""
    if k >= len(X):
        raise ValueError("k must be smaller than the number of observations")

    Xz = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(Xz)
    distances, indices = nn.kneighbors(Xz, return_distance=True)

    # Column 0 is the point itself. Keep exactly k non-self neighbors.
    rows = np.repeat(np.arange(len(Xz)), k)
    cols = indices[:, 1:].reshape(-1)
    vals = distances[:, 1:].reshape(-1)
    directed = csr_matrix((vals, (rows, cols)), shape=(len(Xz), len(Xz)))

    # Euclidean distance is symmetric. maximum() copies one-sided relations
    # and preserves the same weight when a relation is reciprocal.
    graph = directed.maximum(directed.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph


def graph_metadata(graph: csr_matrix) -> dict:
    n = graph.shape[0]
    undirected_edges = graph.nnz // 2
    deg = np.asarray((graph > 0).sum(axis=1)).ravel()
    comps, _ = connected_components(graph, directed=False)
    weights = graph.data
    return {
        "nodes": int(n),
        "edges": int(undirected_edges),
        "density": float(2 * undirected_edges / (n * (n - 1))) if n > 1 else 0.0,
        "mean_degree": float(deg.mean()),
        "min_degree": int(deg.min()) if len(deg) else 0,
        "max_degree": int(deg.max()) if len(deg) else 0,
        "components": int(comps),
        "min_weight": float(weights.min()) if len(weights) else np.nan,
        "max_weight": float(weights.max()) if len(weights) else np.nan,
        "all_weights_positive": bool(np.all(weights > 0)),
    }


def build_original_graphs(data_dir: str | Path, out_dir: str | Path,
                          datasets: Iterable[str] = ("wine", "diabetes", "breast_cancer", "digits"),
                          k_values: Iterable[int] = DEFAULT_K_VALUES) -> pd.DataFrame:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in datasets:
        X, _ = load_csv_dataset(data_dir / f"{dataset}.csv")
        for k in k_values:
            graph = build_union_knn_graph(X, int(k))
            path = out_dir / f"{dataset}_k{k}.npz"
            save_npz(path, graph)
            rows.append({"dataset": dataset, "k": int(k), **graph_metadata(graph)})
    return pd.DataFrame(rows)


def _stratified_indices(X: np.ndarray, y: np.ndarray, n: int, seed: int) -> np.ndarray:
    if n == len(X):
        return np.arange(len(X))
    split = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    idx, _ = next(split.split(X, y))
    return idx


def build_scaling_graphs(data_csv: str | Path, out_dir: str | Path,
                         sizes: Iterable[int] = SCALING_SIZES,
                         k_values: Iterable[int] = DEFAULT_K_VALUES,
                         seeds: Iterable[int] = SCALING_SEEDS) -> pd.DataFrame:
    """Recreate the controlled Digits scaling graphs used in the manuscript."""
    X, y = load_csv_dataset(data_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(seeds)
    rows = []

    for n in sizes:
        n = int(n)
        active_seeds = (seeds[0],) if n == len(X) else seeds
        for seed_id, seed in enumerate(active_seeds):
            idx = _stratified_indices(X, y, n, int(seed))
            Xn = X[idx]
            for k in k_values:
                k = int(k)
                if k >= n:
                    continue
                graph = build_union_knn_graph(Xn, k)
                path = out_dir / f"digits_n{n}_k{k}_s{seed_id}.npz"
                save_npz(path, graph)
                meta = graph_metadata(graph)
                rows.append({
                    "dataset": "digits_scaling",
                    "n": n,
                    "k": k,
                    "seed_id": seed_id,
                    "sampling_seed": int(seed),
                    "nodes": meta["nodes"],
                    "edges": meta["edges"],
                    "density": meta["density"],
                    "mean_degree": meta["mean_degree"],
                    "components": meta["components"],
                    "graph_path": str(path),
                })
    return pd.DataFrame(rows)
