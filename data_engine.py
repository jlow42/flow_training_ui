"""Out-of-core data engine built on DuckDB for lazy CSV access and caching."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

try:  # pragma: no cover - import guard for environments without duckdb
    import duckdb  # type: ignore
    DUCKDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    duckdb = None  # type: ignore
    DUCKDB_AVAILABLE = False

try:  # pragma: no cover - import guard for optional polars support
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:  # pragma: no cover
    pl = None  # type: ignore
    POLARS_AVAILABLE = False

try:  # pragma: no cover - pyarrow is optional but recommended for chunking
    import pyarrow.dataset as pa_dataset
    PYARROW_DATASET_AVAILABLE = True
except ImportError:  # pragma: no cover
    pa_dataset = None  # type: ignore
    PYARROW_DATASET_AVAILABLE = False

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency during runtime
    import pyarrow.parquet as pq  # type: ignore[import]
    PYARROW_AVAILABLE = True
except ImportError:  # pragma: no cover
    pq = None  # type: ignore[assignment]
    PYARROW_AVAILABLE = False


DEFAULT_CACHE_CHUNK_SIZE = 200_000
CACHE_MANIFEST_SUFFIX = ".json"
CACHE_VERSION = 1


class DataEngineError(RuntimeError):
    """Raised when the data engine encounters a critical issue."""


def _env_flag(value: Optional[str], default: bool = True) -> bool:
    """Interpret an environment variable flag."""

    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DataEngineConfig:
    """Configuration flags controlling backend and caching behaviour."""

    enable_duckdb: bool = True
    enable_polars: bool = True
    spill_to_disk: bool = True
    cache_compression: str = "zstd"
    default_chunk_size: int = DEFAULT_CACHE_CHUNK_SIZE

    @classmethod
    def from_env(cls) -> "DataEngineConfig":
        """Initialise configuration from environment variables."""

        chunk_size_env = os.environ.get(
            "FLOW_ENGINE_DEFAULT_CHUNK_SIZE", str(DEFAULT_CACHE_CHUNK_SIZE)
        )
        try:
            chunk_size = int(chunk_size_env)
        except (TypeError, ValueError):
            chunk_size = DEFAULT_CACHE_CHUNK_SIZE

        return cls(
            enable_duckdb=_env_flag(os.environ.get("FLOW_ENGINE_ENABLE_DUCKDB"), True),
            enable_polars=_env_flag(os.environ.get("FLOW_ENGINE_ENABLE_POLARS"), True),
            spill_to_disk=_env_flag(os.environ.get("FLOW_ENGINE_SPILL_TO_DISK"), True),
            cache_compression=os.environ.get("FLOW_ENGINE_CACHE_COMPRESSION", "zstd"),
            default_chunk_size=chunk_size,
        )


class DataEngine:
    """Helper that routes multi-file CSV queries through DuckDB/Polars backends.

    The engine combines arbitrarily large CSV collections without keeping every
    row in memory. DuckDB powers SQL querying while Polars and PyArrow provide a
    lazy interface and chunked/batched materialisation. A spill-to-disk parquet
    cache enables reuse across sessions.
    """

    def __init__(self, cache_dir: Path, config: Optional[DataEngineConfig] = None) -> None:
        self.config = config or DataEngineConfig.from_env()
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.prepared_dir = self.cache_dir / "prepared"
        self.prepared_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "engine.duckdb"
        self._duckdb_available = DUCKDB_AVAILABLE and self.config.enable_duckdb
        self._polars_available = POLARS_AVAILABLE and self.config.enable_polars
        self.conn = duckdb.connect(str(self.db_path)) if self._duckdb_available else None
        self.file_views: Dict[str, Dict[str, object]] = {}

    # ------------------------------------------------------------------
    # File registration
    # ------------------------------------------------------------------
    def reset(self) -> None:
        if self._duckdb_available and self.conn is not None:
            for view_info in self.file_views.values():
                self.conn.execute(
                    f"DROP VIEW IF EXISTS {self._quote_identifier(view_info['view'])}"
                )
        self.file_views.clear()

    def sync_files(self, data_files: Sequence[object]) -> None:
        paths = {str(getattr(df, "path")) for df in data_files}
        to_remove = [path for path in self.file_views if path not in paths]
        for path in to_remove:
            if self._duckdb_available and self.conn is not None:
                view_name = self.file_views[path]["view"]
                self.conn.execute(
                    f"DROP VIEW IF EXISTS {self._quote_identifier(view_name)}"
                )
            self.file_views.pop(path, None)

        for df in data_files:
            path = str(getattr(df, "path"))
            columns = list(getattr(df, "columns", []))
            dtype_hints = dict(getattr(df, "dtype_hints", {}))
            mtime = Path(path).stat().st_mtime if Path(path).exists() else 0.0
            view_name = self.file_views.get(path, {}).get("view")
            if not view_name:
                view_name = self._make_view_name(path)
            if self._duckdb_available and self.conn is not None:
                escaped = path.replace("'", "''")
                self.conn.execute(
                    f"CREATE OR REPLACE VIEW {self._quote_identifier(view_name)} AS "
                    f"SELECT * FROM read_csv_auto('{escaped}', SAMPLE_SIZE=-1)"
                )
            self.file_views[path] = {
                "view": view_name,
                "columns": columns,
                "dtype_hints": dtype_hints,
                "mtime": mtime,
                "path": path,
            }

    def _make_view_name(self, path: str) -> str:
        digest = hashlib.sha1(path.encode()).hexdigest()[:12]
        return f"view_{digest}"

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def fetch_dataframe(
        self,
        columns: Optional[Sequence[str]] = None,
        filters: Optional[str] = None,
    ) -> pd.DataFrame:
        if not self._duckdb_available or self.conn is None:
            return self._fallback_fetch_dataframe(columns, filters)
        relation = self._build_union_relation(columns, filters)
        return relation.df()

    def ensure_cached_dataset(
        self,
        signature: str,
        columns: Optional[Sequence[str]] = None,
        filters: Optional[str] = None,
    ) -> Path:
        if not self.config.spill_to_disk:
            raise DataEngineError(
                "Spill-to-disk caching is disabled via configuration; enable it to create reusable datasets."
            )

        requested_columns = list(columns) if columns else self._all_columns()
        if "__source_file" not in requested_columns:
            requested_columns.append("__source_file")
        cache_path = self.prepared_dir / f"prepared_{signature}.parquet"
        manifest_path = self._cache_manifest_path(cache_path)
        if cache_path.exists() and manifest_path.exists():
            return cache_path

        if not self._duckdb_available or self.conn is None:
            df = self._fallback_fetch_dataframe(requested_columns, filters)
            df.to_parquet(cache_path, index=False)
        else:
            relation = self._build_union_relation(requested_columns, filters)
            relation.write_parquet(
                str(cache_path), compression=self.config.cache_compression
            )

        self._write_cache_manifest(
            cache_path,
            signature,
            requested_columns,
            filters,
        )
        return cache_path

    def stream_dataset(
        self,
        signature: str,
        columns: Sequence[str],
        filters: Optional[str] = None,
        batch_size: int = 50_000,
    ) -> Iterator[pd.DataFrame]:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        cache_path = self.ensure_cached_dataset(signature, columns, filters)
        requested = list(columns)
        if PYARROW_AVAILABLE:
            parquet_file = pq.ParquetFile(cache_path)
            for batch in parquet_file.iter_batches(columns=requested, batch_size=batch_size):
                yield batch.to_pandas()
            return
        dataframe = pd.read_parquet(cache_path, columns=requested)
        total_rows = len(dataframe)
        for start in range(0, total_rows, batch_size):
            stop = start + batch_size
            yield dataframe.iloc[start:stop].copy()

    def column_stats(self, column: str, numeric: bool) -> Dict[str, object]:
        if not self.file_views:
            raise DataEngineError("No files registered in the data engine.")
        if not self._duckdb_available or self.conn is None:
            return self._fallback_column_stats(column, numeric)
        identifier = self._quote_identifier(column)
        union_sql = self._build_union_sql([column], None)
        numeric_expr = []
        if numeric:
            numeric_expr = [
                f"AVG({identifier}) AS avg_value",
                f"STDDEV_POP({identifier}) AS std_value",
                f"quantile_cont({identifier}, 0.25) AS q1_value",
                f"quantile_cont({identifier}, 0.75) AS q3_value",
            ]
        agg_sql = (
            "SELECT "
            + ", ".join(
                [
                    "COUNT(*) AS total_rows",
                    f"COUNT({identifier}) AS non_null",
                    f"SUM(CASE WHEN {identifier} IS NULL THEN 1 ELSE 0 END) AS missing",
                    f"COUNT(DISTINCT {identifier}) AS unique_vals",
                    f"MIN({identifier}) AS min_value",
                    f"MAX({identifier}) AS max_value",
                ]
                + numeric_expr
            )
            + f" FROM ({union_sql})"
        )
        result = self.conn.sql(agg_sql).fetchone()
        stats = {
            "total_rows": result[0],
            "non_null": result[1],
            "missing": result[2],
            "unique_vals": result[3],
            "min_value": result[4],
            "max_value": result[5],
        }
        if numeric:
            stats.update(
                {
                    "avg_value": result[6],
                    "std_value": result[7],
                    "q1_value": result[8],
                    "q3_value": result[9],
                }
            )
        return stats

    def overall_row_stats(self) -> Dict[str, Optional[int]]:
        if not self.file_views:
            return {"total_rows": 0, "distinct_rows": 0, "duplicate_rows": 0}
        if not self._duckdb_available or self.conn is None:
            total = sum(info.get("rows", 0) or 0 for info in self.file_views.values())
            return {"total_rows": total, "distinct_rows": None, "duplicate_rows": None}
        union_sql = self._build_union_sql(None, None)
        total_rows = self.conn.sql(f"SELECT COUNT(*) FROM ({union_sql})").fetchone()[0]
        distinct_rows = self.conn.sql(
            f"SELECT COUNT(*) FROM (SELECT DISTINCT * FROM ({union_sql}))"
        ).fetchone()[0]
        duplicate_rows = total_rows - distinct_rows
        return {
            "total_rows": total_rows,
            "distinct_rows": distinct_rows,
            "duplicate_rows": duplicate_rows,
        }

    def build_signature(
        self,
        columns: Optional[Sequence[str]] = None,
        filters: Optional[str] = None,
        extra: Optional[Dict[str, object]] = None,
    ) -> str:
        requested_columns = list(columns) if columns else self._all_columns()
        if "__source_file" not in requested_columns:
            requested_columns.append("__source_file")
        payload = {
            "columns": requested_columns,
            "filters": filters or "",
            "files": [
                {
                    "path": info["path"],
                    "mtime": info.get("mtime", 0.0),
                }
                for info in sorted(self.file_views.values(), key=lambda item: item["path"])
            ],
            "extra": extra or {},
        }
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()

    def load_lazy_dataset(
        self,
        columns: Optional[Sequence[str]] = None,
        filters: Optional[str] = None,
        use_cache: bool = True,
    ) -> "pl.LazyFrame":
        if not self._polars_available:
            raise DataEngineError(
                "Polars support is disabled or unavailable; enable it to request lazy datasets."
            )

        if not use_cache:
            raise DataEngineError(
                "Lazy datasets require the spill-to-disk cache; call with use_cache=True."
            )

        signature = self.build_signature(columns, filters, extra={"cache_version": CACHE_VERSION})
        cache_path = self.ensure_cached_dataset(signature, columns, filters)
        return pl.scan_parquet(str(cache_path))

    def iter_batches(
        self,
        columns: Optional[Sequence[str]] = None,
        filters: Optional[str] = None,
        batch_size: Optional[int] = None,
        use_cache: bool = True,
    ) -> Iterator[pd.DataFrame]:
        if batch_size is None:
            batch_size = self.config.default_chunk_size

        if batch_size <= 0:
            raise DataEngineError("batch_size must be a positive integer")

        requested_columns = list(columns) if columns else None

        if use_cache and self.config.spill_to_disk:
            signature = self.build_signature(
                requested_columns, filters, extra={"cache_version": CACHE_VERSION}
            )
            cache_path = self.ensure_cached_dataset(signature, requested_columns, filters)
            yield from self._iter_batches_from_cache(cache_path, batch_size)
            return

        if not self._duckdb_available or self.conn is None:
            yield from self._fallback_iter_batches(requested_columns, batch_size)
            return

        offset = 0
        while True:
            base_sql = self._build_union_sql(requested_columns, filters)
            limit_clause = f" LIMIT {int(batch_size)} OFFSET {int(offset)}"
            chunk_relation = self.conn.sql(
                f"SELECT * FROM ({base_sql}) AS unioned{limit_clause}"
            )
            chunk_df = chunk_relation.df()
            if chunk_df.empty:
                break
            yield chunk_df
            offset += len(chunk_df)

    def get_cached_dataset_metadata(self, cache_path: Path) -> Optional[Dict[str, object]]:
        manifest_path = self._cache_manifest_path(cache_path)
        if not manifest_path.exists():
            return None
        with manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _build_union_relation(
        self,
        columns: Optional[Sequence[str]],
        filters: Optional[str],
    ) -> "duckdb.DuckDBPyRelation":
        if not self.file_views:
            raise DataEngineError("No files registered in the data engine.")
        if not self._duckdb_available or self.conn is None:
            raise DataEngineError(
                "DuckDB is not available. Install the 'duckdb' package to enable out-of-core loading."
            )
        union_sql = self._build_union_sql(columns, filters)
        return self.conn.sql(union_sql)

    def _build_union_sql(
        self,
        columns: Optional[Sequence[str]],
        filters: Optional[str],
    ) -> str:
        requested_columns = list(columns) if columns else self._all_columns()
        if "__source_file" not in requested_columns:
            requested_columns.append("__source_file")

        selects: List[str] = []
        for info in self.file_views.values():
            view_name = info["view"]
            file_columns = set(info["columns"])
            parts: List[str] = []
            for column in requested_columns:
                if column == "__source_file":
                    literal = Path(info["path"]).name.replace("'", "''")
                    parts.append(f"'{literal}' AS {self._quote_identifier('__source_file')}")
                elif column in file_columns:
                    parts.append(self._quote_identifier(column))
                else:
                    parts.append(f"NULL AS {self._quote_identifier(column)}")
            where_clause = f" WHERE {filters}" if filters else ""
            selects.append(
                f"SELECT {', '.join(parts)} FROM {self._quote_identifier(view_name)}{where_clause}"
            )
        return " UNION ALL ".join(selects)

    def _all_columns(self) -> List[str]:
        columns: List[str] = []
        for info in self.file_views.values():
            for column in info["columns"]:
                if column not in columns:
                    columns.append(column)
        return columns

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self) -> None:
        try:
            if self.conn is not None:
                self.conn.close()
        except Exception:  # pragma: no cover - best effort cleanup
            pass

    @staticmethod
    def _quote_identifier(name: str) -> str:
        escaped = name.replace('"', '""')
        return f'"{escaped}"'

    def _fallback_fetch_dataframe(
        self,
        columns: Optional[Sequence[str]],
        filters: Optional[str] = None,
    ) -> pd.DataFrame:
        if filters:
            raise DataEngineError(
                "Filtering requires DuckDB; install the 'duckdb' package to enable this feature."
            )
        frames: List[pd.DataFrame] = []
        requested_columns = list(columns) if columns else self._all_columns()
        for info in self.file_views.values():
            path = Path(info["path"])
            usecols = requested_columns if requested_columns else None
            try:
                df = pd.read_csv(path, usecols=usecols, low_memory=False)
            except ValueError as exc:
                raise DataEngineError(str(exc)) from exc
            df["__source_file"] = path.name
            if requested_columns:
                for column in requested_columns:
                    if column not in df.columns:
                        df[column] = pd.NA
                df = df[requested_columns]
            frames.append(df)
        if not frames:
            raise DataEngineError("No data available.")
        return pd.concat(frames, ignore_index=True, copy=False)

    def _fallback_column_stats(self, column: str, numeric: bool) -> Dict[str, object]:
        total_rows = 0
        non_null = 0
        unique_values: set = set()
        min_val = None
        max_val = None
        values: List[float] = [] if numeric else []
        for series in self._fallback_column_series(column):
            total_rows += len(series)
            not_null_series = series.dropna()
            non_null += len(not_null_series)
            unique_values.update(not_null_series.unique().tolist())
            if not not_null_series.empty:
                series_min = not_null_series.min()
                series_max = not_null_series.max()
                min_val = series_min if min_val is None else min(min_val, series_min)
                max_val = series_max if max_val is None else max(max_val, series_max)
                if numeric:
                    values.extend(pd.to_numeric(not_null_series, errors="coerce").dropna().tolist())
        missing = total_rows - non_null
        stats: Dict[str, object] = {
            "total_rows": total_rows,
            "non_null": non_null,
            "missing": missing,
            "unique_vals": len(unique_values),
            "min_value": min_val,
            "max_value": max_val,
        }
        if numeric and values:
            arr = np.array(values, dtype=float)
            stats.update(
                {
                    "avg_value": float(arr.mean()),
                    "std_value": float(arr.std()) if arr.size > 1 else 0.0,
                    "q1_value": float(np.quantile(arr, 0.25)),
                    "q3_value": float(np.quantile(arr, 0.75)),
                }
            )
        return stats

    def _fallback_column_series(self, column: str) -> Iterator[pd.Series]:
        for info in self.file_views.values():
            path = Path(info["path"])
            if column not in info.get("columns", []):
                continue
            series = pd.read_csv(path, usecols=[column], low_memory=False)[column]
            yield series

    def _iter_batches_from_cache(
        self, cache_path: Path, batch_size: int
    ) -> Iterator[pd.DataFrame]:
        if self._duckdb_available and self.conn is not None:
            offset = 0
            escaped = cache_path.as_posix().replace("'", "''")
            while True:
                sql = (
                    f"SELECT * FROM read_parquet('{escaped}')"
                    f" LIMIT {int(batch_size)} OFFSET {int(offset)}"
                )
                df = self.conn.sql(sql).df()
                if df.empty:
                    break
                yield df
                offset += len(df)
            return

        if PYARROW_DATASET_AVAILABLE:
            dataset = pa_dataset.dataset(str(cache_path))
            for batch in dataset.to_batches(batch_size=batch_size):
                yield batch.to_pandas()
            return

        df = pd.read_parquet(cache_path)
        for start in range(0, len(df), batch_size):
            yield df.iloc[start : start + batch_size].copy()

    def _fallback_iter_batches(
        self, columns: Optional[Sequence[str]], batch_size: int
    ) -> Iterator[pd.DataFrame]:
        if batch_size <= 0:
            raise DataEngineError("batch_size must be a positive integer")

        requested_columns = list(columns) if columns else self._all_columns()
        if "__source_file" not in requested_columns:
            requested_columns.append("__source_file")

        for info in self.file_views.values():
            path = Path(info["path"])
            usecols = [col for col in requested_columns if col != "__source_file"] or None
            try:
                iterator = pd.read_csv(
                    path,
                    usecols=usecols,
                    low_memory=False,
                    chunksize=batch_size,
                )
            except ValueError as exc:
                raise DataEngineError(str(exc)) from exc
            for chunk in iterator:
                chunk["__source_file"] = path.name
                for column in requested_columns:
                    if column not in chunk.columns:
                        chunk[column] = pd.NA
                chunk = chunk[requested_columns]
                yield chunk

    def _cache_manifest_path(self, cache_path: Path) -> Path:
        return cache_path.with_suffix(cache_path.suffix + CACHE_MANIFEST_SUFFIX)

    def _write_cache_manifest(
        self,
        cache_path: Path,
        signature: str,
        columns: Sequence[str],
        filters: Optional[str],
    ) -> None:
        manifest_path = self._cache_manifest_path(cache_path)
        manifest = {
            "signature": signature,
            "columns": list(columns),
            "filters": filters or "",
            "cache_version": CACHE_VERSION,
            "created_at": time.time(),
            "files": [
                {
                    "path": info["path"],
                    "mtime": info.get("mtime", 0.0),
                }
                for info in sorted(self.file_views.values(), key=lambda item: item["path"])
            ],
        }
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True)
