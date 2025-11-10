import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app import FlowDataApp, DataFile, RANDOM_STATE
from data_engine import DataEngine


class StreamingTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="streaming_test_"))
        self.cache_dir = self.temp_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Bypass the GUI-heavy FlowDataApp.__init__ to avoid requiring a display.
        self.app = FlowDataApp.__new__(FlowDataApp)
        self.app.data_engine = DataEngine(self.cache_dir)
        self.app.training_selection = []
        self.app.column_numeric_hints = {}

    def tearDown(self) -> None:
        self.app.data_engine.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _generate_large_dataset(self, n_rows: int = 120_000) -> pd.DataFrame:
        rng = np.random.default_rng(RANDOM_STATE)
        features = rng.normal(size=(n_rows, 4))
        weights = np.array([0.8, -0.3, 0.2, 0.5])
        logits = features @ weights
        probabilities = 1 / (1 + np.exp(-logits))
        # Create three classes with non-uniform probabilities.
        labels = np.where(probabilities > 0.65, "CD4", np.where(probabilities > 0.35, "CD8", "B"))
        dataframe = pd.DataFrame(features, columns=["F0", "F1", "F2", "F3"])
        dataframe["Target"] = labels
        return dataframe

    def _write_csv_parts(self, dataframe: pd.DataFrame, parts: int = 3) -> list[Path]:
        paths: list[Path] = []
        chunk_size = math.ceil(len(dataframe) / parts)
        for idx in range(parts):
            start = idx * chunk_size
            end = min((idx + 1) * chunk_size, len(dataframe))
            chunk = dataframe.iloc[start:end]
            path = self.temp_dir / f"stream_chunk_{idx}.csv"
            chunk.to_csv(path, index=False)
            paths.append(path)
        return paths

    def _build_data_files(self, paths: list[Path]) -> list[DataFile]:
        data_files: list[DataFile] = []
        for path in paths:
            sample = pd.read_csv(path, nrows=50)
            columns = list(sample.columns)
            dtype_hints = {col: sample[col].dtype.kind for col in columns}
            data_files.append(DataFile(path=path, columns=columns, dtype_hints=dtype_hints))
        return data_files

    def test_naive_bayes_streaming_large_dataset(self) -> None:
        dataframe = self._generate_large_dataset()
        csv_paths = self._write_csv_parts(dataframe)
        data_files = self._build_data_files(csv_paths)
        self.app.data_files = data_files
        self.app.training_selection = ["F0", "F1", "F2", "F3"]
        self.app.column_numeric_hints = {feature: True for feature in self.app.training_selection}
        self.app.data_engine.sync_files(data_files)

        required_columns = self.app.training_selection + ["Target"]
        signature = self.app.data_engine.build_signature(
            columns=required_columns,
            extra={"mode": "training"},
        )
        streaming_cfg = {
            "signature": signature,
            "columns": required_columns,
            "batch_size": 50_000,
            "seed": RANDOM_STATE,
        }

        payload = self.app._train_naive_bayes_streaming(
            params={"var_smoothing": 1e-9},
            features=self.app.training_selection,
            target="Target",
            streaming_cfg=streaming_cfg,
            test_size=0.2,
            class_balance_mode="Balanced",
        )

        self.assertEqual(payload["train_rows"] + payload["test_rows"], len(dataframe))
        metrics = payload["metrics"]
        self.assertGreater(metrics["accuracy"], 0.75)
        self.assertIn("Streaming partial_fit", metrics.get("extra_status", ""))
        self.assertIn("Cross-validation", payload["cv_warning"])
        confusion = payload["confusion_matrix"]
        self.assertEqual(confusion.shape[0], len(payload["classes"]))


if __name__ == "__main__":
    unittest.main()
