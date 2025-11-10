"""Tests for prepared dataset caching in the data engine."""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from data_engine import DataEngine, PYARROW_AVAILABLE


def _create_engine(cache_dir: Path) -> DataEngine:
    engine = DataEngine(cache_dir)
    engine._duckdb_available = False  # force pandas fallback in tests
    engine.conn = None
    return engine


def _data_file(path: Path, frame: pd.DataFrame) -> SimpleNamespace:
    dtype_hints = {column: frame[column].dtype.kind for column in frame.columns}
    return SimpleNamespace(path=path, columns=list(frame.columns), dtype_hints=dtype_hints)


def _prepare_file(tmp_path: Path, name: str) -> tuple[SimpleNamespace, pd.DataFrame]:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    csv_path = tmp_path / name
    frame.to_csv(csv_path, index=False)
    return _data_file(csv_path, frame), frame


def test_ensure_cached_dataset_creates_metadata(tmp_path: Path) -> None:
    engine = _create_engine(tmp_path / "cache")
    data_file, frame = _prepare_file(tmp_path, "data.csv")
    engine.sync_files([data_file])

    columns = list(frame.columns)
    signature = engine.build_signature(columns=columns, extra={"mode": "training"})
    cache_path = engine.ensure_cached_dataset(
        signature=signature,
        columns=columns,
        extra={"mode": "training"},
    )

    assert cache_path.exists()
    metadata_path = engine.prepared_dir / signature[:2] / signature / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["signature"] == signature
    assert metadata["format"] == "parquet"
    assert metadata["columns"] == columns
    assert metadata["files"][0]["path"] == str(data_file.path)


def test_ensure_cached_dataset_reuses_existing_file(tmp_path: Path) -> None:
    engine = _create_engine(tmp_path / "cache")
    data_file, frame = _prepare_file(tmp_path, "reusable.csv")
    engine.sync_files([data_file])
    columns = list(frame.columns)

    signature = engine.build_signature(columns=columns, extra={"mode": "training"})
    cache_path = engine.ensure_cached_dataset(
        signature=signature,
        columns=columns,
        extra={"mode": "training"},
    )
    metadata_path = engine.prepared_dir / signature[:2] / signature / "metadata.json"
    metadata_before = json.loads(metadata_path.read_text())

    second_path = engine.ensure_cached_dataset(
        signature=signature,
        columns=columns,
        extra={"mode": "training"},
    )
    metadata_after = json.loads(metadata_path.read_text())

    assert cache_path == second_path
    assert metadata_after["created_at"] == metadata_before["created_at"]


def test_lookup_cached_dataset_respects_file_changes(tmp_path: Path) -> None:
    engine = _create_engine(tmp_path / "cache")
    data_file, frame = _prepare_file(tmp_path, "changing.csv")
    engine.sync_files([data_file])
    columns = list(frame.columns)

    signature = engine.build_signature(columns=columns, extra={"mode": "training"})
    original_path = engine.ensure_cached_dataset(
        signature=signature,
        columns=columns,
        extra={"mode": "training"},
    )
    assert original_path.exists()

    # Modify the underlying file to change its modification time/signature.
    time.sleep(1.1)
    updated_frame = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 5, 6, 7]})
    updated_frame.to_csv(data_file.path, index=False)
    updated_file = _data_file(Path(data_file.path), updated_frame)
    engine.sync_files([updated_file])

    new_columns = list(updated_frame.columns)
    new_signature = engine.build_signature(columns=new_columns, extra={"mode": "training"})
    assert new_signature != signature

    new_path = engine.ensure_cached_dataset(
        signature=new_signature,
        columns=new_columns,
        extra={"mode": "training"},
    )
    assert new_path.exists()
    assert new_path != original_path


def test_cached_dataset_reused_by_new_engine_instance(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    engine = _create_engine(cache_dir)
    data_file, frame = _prepare_file(tmp_path, "workflow.csv")
    engine.sync_files([data_file])
    columns = list(frame.columns)

    signature = engine.build_signature(columns=columns, extra={"mode": "training"})
    created_path = engine.ensure_cached_dataset(
        signature=signature,
        columns=columns,
        extra={"mode": "training"},
    )
    assert created_path.exists()

    # Simulate a new workflow by creating a fresh engine instance.
    engine.close()
    successor = _create_engine(cache_dir)
    successor.sync_files([data_file])
    reused_path = successor.lookup_cached_dataset(
        signature=signature,
        columns=columns,
        extra={"mode": "training"},
    )
    assert reused_path == created_path


@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow is required for Arrow cache format")
def test_arrow_format_cache_creation(tmp_path: Path) -> None:
    engine = _create_engine(tmp_path / "cache")
    data_file, frame = _prepare_file(tmp_path, "arrow.csv")
    engine.sync_files([data_file])
    columns = list(frame.columns)

    signature = engine.build_signature(columns=columns, extra={"mode": "training"})
    cache_path = engine.ensure_cached_dataset(
        signature=signature,
        columns=columns,
        extra={"mode": "training"},
        storage_format="arrow",
    )

    assert cache_path.suffix == ".arrow"
    metadata_path = engine.prepared_dir / signature[:2] / signature / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["format"] == "arrow"
