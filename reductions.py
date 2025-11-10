"""Dimensionality reduction helpers with shared neighbor caching."""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.decomposition import FastICA, PCA
from sklearn.neighbors import NearestNeighbors

try:  # pragma: no cover - optional dependency guard
    from scipy import sparse  # type: ignore[import]
except ImportError:  # pragma: no cover
    sparse = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency guard
    import umap  # type: ignore[import]
except ImportError:  # pragma: no cover
    umap = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency guard
    import phate  # type: ignore[import]
except ImportError:  # pragma: no cover
    phate = None  # type: ignore[assignment]


class ReductionError(RuntimeError):
    """Raised when a reduction pipeline encounters an unrecoverable issue."""


class ReductionCancelled(RuntimeError):
    """Raised when an in-flight reduction job is cancelled."""


@dataclass(frozen=True)
class NeighborGraph:
    """Container for reusable nearest-neighbour results."""

    indices: np.ndarray
    distances: np.ndarray
    graph: Optional["sparse.csr_matrix"]


class NeighborCache:
    """Caches nearest-neighbour computations keyed by dataset signature."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, int, str], NeighborGraph] = {}

    @staticmethod
    def _dataset_signature(data: np.ndarray) -> str:
        contiguous = np.ascontiguousarray(data)
        digest = hashlib.sha1()
        digest.update(int(contiguous.shape[0]).to_bytes(8, "little"))
        digest.update(int(contiguous.shape[1]).to_bytes(8, "little"))
        digest.update(contiguous.view(np.uint8).tobytes())
        return digest.hexdigest()

    def clear(self) -> None:
        self._cache.clear()

    def get(
        self,
        data: np.ndarray,
        n_neighbors: int,
        metric: str,
        *,
        n_jobs: int,
        cancel_event: Optional["threading.Event"] = None,
    ) -> NeighborGraph:
        if cancel_event is not None and cancel_event.is_set():
            raise ReductionCancelled()

        signature = self._dataset_signature(data)
        key = (signature, int(n_neighbors), metric)
        if key in self._cache:
            return self._cache[key]

        nbrs = NearestNeighbors(
            n_neighbors=max(2, int(n_neighbors)),
            metric=metric,
            n_jobs=max(1, int(n_jobs)),
        )
        nbrs.fit(data)
        if cancel_event is not None and cancel_event.is_set():
            raise ReductionCancelled()
        distances, indices = nbrs.kneighbors(data)
        knn_graph = None
        if sparse is not None:
            graph = nbrs.kneighbors_graph(mode="distance")
            graph = graph.maximum(graph.T)
            graph.setdiag(0)
            graph.eliminate_zeros()
            knn_graph = graph.tocsr()
        graph_entry = NeighborGraph(indices=indices, distances=distances, graph=knn_graph)
        self._cache[key] = graph_entry
        return graph_entry


class ReductionRunner:
    """Runs dimensionality reduction pipelines with neighbour caching."""

    def __init__(self) -> None:
        self.neighbor_cache = NeighborCache()

    def run(
        self,
        method: str,
        data: np.ndarray,
        *,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        n_jobs: int = 1,
        random_state: int = 42,
        cancel_event: Optional["threading.Event"] = None,
    ) -> Tuple[np.ndarray, float]:
        start = time.time()
        method_lower = method.lower()
        if cancel_event is not None and cancel_event.is_set():
            raise ReductionCancelled()

        if method_lower == "pca":
            embedding = self._run_pca(data, n_components, random_state, cancel_event)
        elif method_lower == "ica":
            embedding = self._run_ica(data, n_components, random_state, cancel_event)
        elif method_lower == "umap":
            embedding = self._run_umap(
                data,
                n_components,
                n_neighbors,
                min_dist,
                metric,
                n_jobs,
                random_state,
                cancel_event,
            )
        elif method_lower == "phate":
            embedding = self._run_phate(
                data,
                n_components,
                n_neighbors,
                metric,
                n_jobs,
                random_state,
                cancel_event,
            )
        else:
            raise ReductionError(f"Unknown reduction method '{method}'.")
        elapsed = time.time() - start
        return embedding, elapsed

    def _run_pca(
        self,
        data: np.ndarray,
        n_components: int,
        random_state: int,
        cancel_event: Optional["threading.Event"],
    ) -> np.ndarray:
        if cancel_event is not None and cancel_event.is_set():
            raise ReductionCancelled()
        n_components = max(1, min(int(n_components), data.shape[1]))
        model = PCA(n_components=n_components, random_state=random_state)
        return model.fit_transform(data)

    def _run_ica(
        self,
        data: np.ndarray,
        n_components: int,
        random_state: int,
        cancel_event: Optional["threading.Event"],
    ) -> np.ndarray:
        if cancel_event is not None and cancel_event.is_set():
            raise ReductionCancelled()
        n_components = max(1, min(int(n_components), data.shape[1]))
        model = FastICA(n_components=n_components, random_state=random_state, max_iter=500)
        return model.fit_transform(data)

    def _run_umap(
        self,
        data: np.ndarray,
        n_components: int,
        n_neighbors: int,
        min_dist: float,
        metric: str,
        n_jobs: int,
        random_state: int,
        cancel_event: Optional["threading.Event"],
    ) -> np.ndarray:
        if umap is None:
            raise ReductionError("Install 'umap-learn' to run UMAP reductions.")
        graph = self.neighbor_cache.get(
            data,
            n_neighbors,
            metric,
            n_jobs=n_jobs,
            cancel_event=cancel_event,
        )
        if cancel_event is not None and cancel_event.is_set():
            raise ReductionCancelled()
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        try:
            embedding = reducer.fit_transform(
                data,
                knn_indices=graph.indices,
                knn_dists=graph.distances,
            )
        except TypeError:
            embedding = reducer.fit_transform(data)
        return embedding

    def _run_phate(
        self,
        data: np.ndarray,
        n_components: int,
        n_neighbors: int,
        metric: str,
        n_jobs: int,
        random_state: int,
        cancel_event: Optional["threading.Event"],
    ) -> np.ndarray:
        if phate is None:
            raise ReductionError("Install 'phate' to run PHATE reductions.")
        graph = self.neighbor_cache.get(
            data,
            n_neighbors,
            metric,
            n_jobs=n_jobs,
            cancel_event=cancel_event,
        )
        if cancel_event is not None and cancel_event.is_set():
            raise ReductionCancelled()
        reducer = phate.PHATE(
            knn=n_neighbors,
            decay=None,
            n_components=n_components,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        try:
            embedding = reducer.fit_transform(
                data,
                knn_distances=graph.distances,
                knn_indices=graph.indices,
            )
        except TypeError:
            embedding = reducer.fit_transform(data)
        return embedding


__all__ = [
    "NeighborCache",
    "NeighborGraph",
    "ReductionCancelled",
    "ReductionError",
    "ReductionRunner",
]
