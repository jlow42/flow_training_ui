import unittest
from threading import Event

import numpy as np

import reductions
from reductions import ReductionRunner, ReductionCancelled


class ReductionRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.data = rng.normal(size=(120, 6)).astype(np.float32)
        self.runner = ReductionRunner()

    def test_neighbor_cache_reuse(self) -> None:
        cache = self.runner.neighbor_cache
        result_first = cache.get(self.data, 12, "euclidean", n_jobs=1)
        result_second = cache.get(self.data.copy(), 12, "euclidean", n_jobs=1)
        self.assertIs(result_first, result_second)
        self.assertEqual(result_first.indices.shape[1], 12)

    def test_pca_reduction_shape(self) -> None:
        embedding, elapsed = self.runner.run("pca", self.data, n_components=2)
        self.assertEqual(embedding.shape, (120, 2))
        self.assertGreaterEqual(elapsed, 0.0)

    def test_umap_reduction_runs(self) -> None:
        if reductions.umap is None:  # pragma: no cover - optional dependency
            self.skipTest("umap-learn not installed")
        embedding, _ = self.runner.run(
            "umap",
            self.data,
            n_neighbors=10,
            min_dist=0.2,
            metric="euclidean",
            n_jobs=1,
        )
        self.assertEqual(embedding.shape, (120, 2))

    def test_phate_reduction_runs(self) -> None:
        if reductions.phate is None:  # pragma: no cover - optional dependency
            self.skipTest("phate not installed")
        embedding, _ = self.runner.run(
            "phate",
            self.data,
            n_neighbors=10,
            metric="euclidean",
            n_jobs=1,
        )
        self.assertEqual(embedding.shape, (120, 2))

    def test_cancellation_raises(self) -> None:
        cancel_event = Event()
        cancel_event.set()
        with self.assertRaises(ReductionCancelled):
            self.runner.run("pca", self.data, cancel_event=cancel_event)


if __name__ == "__main__":
    unittest.main()
