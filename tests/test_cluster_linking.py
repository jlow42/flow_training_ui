import unittest

import numpy as np
import pandas as pd

from cluster_linking import (
    build_table_rows,
    filter_by_limits,
    select_points_within_path,
    summarize_clusters,
)


class ClusterLinkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(42)
        size = 100_000
        cls.large_df = pd.DataFrame(
            {
                "x": rng.normal(loc=0.0, scale=1.0, size=size),
                "y": rng.normal(loc=1.0, scale=2.0, size=size),
                "cluster": rng.integers(0, 5, size=size),
            }
        )

    def test_filter_by_limits_large_dataset(self) -> None:
        df = type(self).large_df
        subset = filter_by_limits(df, "x", "y", (-0.5, 0.5), (0.0, 2.0))
        self.assertGreater(len(subset), 0)
        self.assertTrue(((subset["x"] >= -0.5) & (subset["x"] <= 0.5)).all())
        self.assertTrue(((subset["y"] >= 0.0) & (subset["y"] <= 2.0)).all())

    def test_select_points_within_path(self) -> None:
        df = pd.DataFrame(
            {
                "x": [0.0, 1.0, 2.0, 3.0],
                "y": [0.0, 1.0, 0.0, 1.0],
                "cluster": ["a", "a", "b", "b"],
            }
        )
        vertices = [(0.0, 0.0), (2.0, -0.5), (2.0, 1.5), (0.0, 1.5)]
        mask = select_points_within_path(df, "x", "y", vertices)
        self.assertTrue(mask.dtype == bool)
        self.assertEqual(mask.sum(), 3)
        self.assertTrue(mask[0])
        self.assertTrue(mask[1])
        self.assertFalse(mask[3])

    def test_summarize_clusters(self) -> None:
        df = pd.DataFrame(
            {
                "cluster": ["a", "a", "b", "b", "b"],
                "x": [1, 2, 3, 4, 5],
                "y": [5, 4, 3, 2, 1],
            }
        )
        summary = summarize_clusters(df)
        self.assertEqual(summary.total_rows, 5)
        self.assertEqual(summary.cluster_counts["a"], 2)
        self.assertEqual(summary.cluster_counts["b"], 3)

    def test_build_table_rows_limit(self) -> None:
        df = pd.DataFrame(
            {
                "cluster": [0] * 300,
                "x": np.linspace(0, 1, 300),
                "y": np.linspace(1, 2, 300),
            }
        )
        rows = list(build_table_rows(df, "x", "y", max_rows=50))
        self.assertEqual(len(rows), 50)
        first_row = rows[0]
        self.assertIn(first_row[1], {"0", "0.0"})
        self.assertTrue(first_row[2])
        self.assertTrue(first_row[3])


if __name__ == "__main__":
    unittest.main()
