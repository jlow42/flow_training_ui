from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import polars as pl
import pytest

from data_engine import (
    CACHE_MANIFEST_SUFFIX,
    DataEngine,
    DataEngineConfig,
)


pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")


def _make_csv(tmp_path: Path, name: str, rows: int, start: int = 0) -> Path:
    frame = pd.DataFrame(
        {
            "index": list(range(start, start + rows)),
            "value": [float(i) for i in range(start, start + rows)],
            "label": ["even" if i % 2 == 0 else "odd" for i in range(start, start + rows)],
        }
    )
    csv_path = tmp_path / name
    frame.to_csv(csv_path, index=False)
    return csv_path


def _data_file(path: Path) -> SimpleNamespace:
    df = pd.read_csv(path, nrows=0)
    return SimpleNamespace(path=str(path), columns=list(df.columns), dtype_hints={})


@pytest.fixture()
def engine(tmp_path: Path) -> DataEngine:
    csv_one = _make_csv(tmp_path, "part_1.csv", rows=6, start=0)
    csv_two = _make_csv(tmp_path, "part_2.csv", rows=6, start=100)
    cache_dir = tmp_path / "cache"
    eng = DataEngine(cache_dir, DataEngineConfig())
    eng.sync_files([_data_file(csv_one), _data_file(csv_two)])
    yield eng
    eng.close()


def test_iter_batches_handles_large_dataset(engine: DataEngine) -> None:
    batches = list(engine.iter_batches(columns=["index", "value"], batch_size=4))
    assert batches, "Expected multiple batches from iterator"

    total_rows = sum(len(batch) for batch in batches)
    assert total_rows == 12

    for batch in batches:
        assert "__source_file" in batch.columns
        assert len(batch) <= 4

    lazy_frame = engine.load_lazy_dataset(columns=["index", "value", "label"])
    assert isinstance(lazy_frame, pl.LazyFrame)
    collected = lazy_frame.collect()
    assert collected.shape == (12, 4)
    assert "__source_file" in collected.columns


def test_cache_reuse_across_sessions(tmp_path: Path) -> None:
    csv_one = _make_csv(tmp_path, "part_a.csv", rows=5, start=50)
    csv_two = _make_csv(tmp_path, "part_b.csv", rows=5, start=200)
    cache_dir = tmp_path / "cache"

    files = [_data_file(csv_one), _data_file(csv_two)]
    engine_first = DataEngine(cache_dir, DataEngineConfig())
    engine_first.sync_files(files)
    signature = engine_first.build_signature(columns=["index", "value"], extra={"test": "reuse"})
    cache_path = engine_first.ensure_cached_dataset(signature, columns=["index", "value"])
    manifest_path = cache_path.with_suffix(cache_path.suffix + CACHE_MANIFEST_SUFFIX)
    initial_cache_mtime = cache_path.stat().st_mtime_ns
    initial_manifest_mtime = manifest_path.stat().st_mtime_ns
    engine_first.close()

    engine_second = DataEngine(cache_dir, DataEngineConfig())
    engine_second.sync_files(files)
    cache_path_second = engine_second.ensure_cached_dataset(
        signature, columns=["index", "value"]
    )
    assert cache_path_second == cache_path
    assert cache_path_second.stat().st_mtime_ns == initial_cache_mtime
    assert manifest_path.stat().st_mtime_ns == initial_manifest_mtime

    # Confirm cache metadata can be inspected and batches reuse the stored parquet.
    metadata = engine_second.get_cached_dataset_metadata(cache_path_second)
    assert metadata is not None
    assert metadata["signature"] == signature

    batches = list(engine_second.iter_batches(columns=["index", "value"], batch_size=3))
    assert sum(len(batch) for batch in batches) == 10
    engine_second.close()
