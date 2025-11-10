import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from app import FlowDataApp, RANDOM_STATE

try:
    import hdbscan  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency in tests
    hdbscan = None  # type: ignore[assignment]


def _make_cluster_dataset(n_per_cluster: int = 40) -> tuple[np.ndarray, pd.Series]:
    rng = np.random.default_rng(RANDOM_STATE)
    cluster_a = rng.normal(loc=(-2.0, -2.0), scale=0.4, size=(n_per_cluster, 2))
    cluster_b = rng.normal(loc=(2.5, 2.5), scale=0.4, size=(n_per_cluster, 2))
    features = np.vstack([cluster_a, cluster_b])
    labels = pd.Series([0] * n_per_cluster + [1] * n_per_cluster)
    return features, labels


class ClusteringBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        features, labels = _make_cluster_dataset()
        scaler = StandardScaler()
        self.features_scaled = scaler.fit_transform(features)
        self.true_labels = labels
        self.predicted_labels = np.concatenate(
            [np.zeros(len(labels) // 2, dtype=int), np.ones(len(labels) // 2, dtype=int)]
        )

    def test_metric_evaluation_with_reference_labels(self) -> None:
        metrics = FlowDataApp._evaluate_clustering_metrics(
            self.predicted_labels,
            self.features_scaled,
            self.true_labels,
        )
        self.assertIsNotNone(metrics["silhouette"])
        self.assertGreater(metrics["silhouette"], 0.5)
        self.assertIsNotNone(metrics["ari"])
        self.assertGreater(metrics["ari"], 0.9)
        self.assertIsNotNone(metrics["nmi"])
        self.assertGreater(metrics["nmi"], 0.9)

    def test_metric_evaluation_without_reference_labels(self) -> None:
        metrics = FlowDataApp._evaluate_clustering_metrics(
            self.predicted_labels,
            self.features_scaled,
            None,
        )
        self.assertIsNotNone(metrics["silhouette"])
        self.assertGreater(metrics["silhouette"], 0.5)
        self.assertIsNone(metrics["ari"])
        self.assertIsNone(metrics["nmi"])

    def test_gaussian_mixture_backend_clusters(self) -> None:
        app = FlowDataApp.__new__(FlowDataApp)  # type: ignore[call-arg]
        labels = app._run_clustering_method(  # type: ignore[attr-defined]
            "gmm",
            self.features_scaled,
            {
                "n_components": 2,
                "covariance_type": "full",
                "reg_covar": 1e-4,
                "n_init": 5,
            },
            n_jobs=1,
        )
        self.assertEqual(len(labels), len(self.features_scaled))
        self.assertEqual(len(np.unique(labels)), 2)

    def test_agglomerative_backend_clusters(self) -> None:
        app = FlowDataApp.__new__(FlowDataApp)  # type: ignore[call-arg]
        labels = app._run_clustering_method(  # type: ignore[attr-defined]
            "agglomerative",
            self.features_scaled,
            {"n_clusters": 2, "linkage": "ward", "metric": "euclidean"},
            n_jobs=1,
        )
        self.assertEqual(len(labels), len(self.features_scaled))
        self.assertEqual(len(np.unique(labels)), 2)

    @unittest.skipIf(hdbscan is None, "hdbscan not installed")
    def test_hdbscan_backend_detects_multiple_clusters(self) -> None:
        app = FlowDataApp.__new__(FlowDataApp)  # type: ignore[call-arg]
        labels = app._run_clustering_method(  # type: ignore[attr-defined]
            "hdbscan",
            self.features_scaled,
            {
                "min_cluster_size": 10,
                "min_samples": 5,
                "cluster_selection_method": "eom",
            },
            n_jobs=1,
        )
        self.assertEqual(len(labels), len(self.features_scaled))
        unique_labels = np.unique(labels[labels >= 0])
        self.assertGreaterEqual(len(unique_labels), 2)


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    unittest.main()
