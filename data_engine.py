"""Out-of-core data engine built on DuckDB for lazy CSV access and caching."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

try:  # pragma: no cover - import guard for environments without duckdb
    import duckdb  # type: ignore
    DUCKDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    duckdb = None  # type: ignore
    DUCKDB_AVAILABLE = False

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency for Arrow output
    import pyarrow as pa  # type: ignore
    import pyarrow.feather as feather  # type: ignore
    PYARROW_AVAILABLE = True
except ImportError:  # pragma: no cover
    pa = None  # type: ignore
    feather = None  # type: ignore
    PYARROW_AVAILABLE = False


class DataEngineError(RuntimeError):
    """Raised when the data engine encounters a critical issue."""


class DataEngine:
    """Minimal helper that routes multi-file CSV queries through DuckDB.

    This lets the application combine arbitrarily large CSV collections without
    keeping every row in memory and provides a shared cache for expensive
    combinations (e.g., training feature sets).
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.prepared_dir = self.cache_dir / "prepared"
        self.prepared_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "engine.duckdb"
        self._duckdb_available = DUCKDB_AVAILABLE
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
        columns: Sequence[str],
        filters: Optional[str] = None,
        *,
        extra: Optional[Dict[str, object]] = None,
        storage_format: str = "parquet",
    ) -> Path:
        context = self._build_metadata_context(
            signature=signature,
            columns=columns,
            filters=filters,
            extra=extra or {},
            storage_format=storage_format,
        )
        cached = self._resolve_cached_dataset(context)
        if cached is not None:
            return cached

        dataset_dir = self._prepared_dataset_dir(signature)
        self._drop_cached_dataset(dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        data_path = dataset_dir / self._dataset_filename(storage_format)

        row_count: Optional[int] = None
        if not self._duckdb_available or self.conn is None:
            df = self._fallback_fetch_dataframe(columns, filters)
            row_count = len(df.index)
            if storage_format == "parquet":
                df.to_parquet(data_path, index=False)
            elif storage_format == "arrow":
                if not PYARROW_AVAILABLE:
                    raise DataEngineError(
                        "pyarrow is required to cache datasets in Arrow format."
                    )
                table = pa.Table.from_pandas(df, preserve_index=False)
                feather.write_feather(table, data_path)
            else:
                raise DataEngineError(f"Unsupported storage format '{storage_format}'.")
        else:
            relation = self._build_union_relation(columns, filters)
            if storage_format == "parquet":
                relation.write_parquet(str(data_path), compression="zstd")
                row_count = relation.aggregate("COUNT(*) AS rows").fetchone()[0]
            elif storage_format == "arrow":
                if not PYARROW_AVAILABLE:
                    raise DataEngineError(
                        "pyarrow is required to cache datasets in Arrow format."
                    )
                table = relation.arrow()
                feather.write_feather(table, data_path)
                row_count = table.num_rows
            else:
                raise DataEngineError(f"Unsupported storage format '{storage_format}'.")

        metadata = dict(context)
        metadata.update(
            {
                "version": 1,
                "created_at": time.time(),
                "row_count": row_count,
            }
        )
        self._write_metadata(dataset_dir, metadata)
        return data_path

    def lookup_cached_dataset(
        self,
        signature: str,
        columns: Sequence[str],
        filters: Optional[str] = None,
        *,
        extra: Optional[Dict[str, object]] = None,
        storage_format: str = "parquet",
    ) -> Optional[Path]:
        context = self._build_metadata_context(
            signature=signature,
            columns=columns,
            filters=filters,
            extra=extra or {},
            storage_format=storage_format,
        )
        return self._resolve_cached_dataset(context)

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
        columns: Sequence[str],
        filters: Optional[str] = None,
        extra: Optional[Dict[str, object]] = None,
    ) -> str:
        payload = {
            "columns": list(columns),
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

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _build_metadata_context(
        self,
        *,
        signature: str,
        columns: Sequence[str],
        filters: Optional[str],
        extra: Dict[str, object],
        storage_format: str,
    ) -> Dict[str, object]:
        normalized_columns = list(columns)
        normalized_filters = filters or ""
        files = self._current_file_descriptors()
        return {
            "signature": signature,
            "columns": normalized_columns,
            "filters": normalized_filters,
            "extra": extra,
            "files": files,
            "format": storage_format,
        }

    def _prepared_dataset_dir(self, signature: str) -> Path:
        prefix = signature[:2]
        return self.prepared_dir / prefix / signature

    def _dataset_filename(self, storage_format: str) -> str:
        if storage_format == "parquet":
            return "data.parquet"
        if storage_format == "arrow":
            return "data.arrow"
        raise DataEngineError(f"Unsupported storage format '{storage_format}'.")

    def _resolve_cached_dataset(self, context: Dict[str, object]) -> Optional[Path]:
        signature = str(context["signature"])
        dataset_dir = self._prepared_dataset_dir(signature)
        metadata = self._read_metadata(dataset_dir)
        if not metadata:
            return None
        if not self._metadata_matches(metadata, context):
            self._drop_cached_dataset(dataset_dir)
            return None
        format_value = metadata.get("format")
        if not isinstance(format_value, str):
            self._drop_cached_dataset(dataset_dir)
            return None
        dataset_path = dataset_dir / self._dataset_filename(format_value)
        if not dataset_path.exists():
            self._drop_cached_dataset(dataset_dir)
            return None
        return dataset_path

    def _metadata_matches(
        self,
        metadata: Dict[str, object],
        context: Dict[str, object],
    ) -> bool:
        if metadata.get("signature") != context.get("signature"):
            return False
        if metadata.get("format") != context.get("format"):
            return False
        if metadata.get("filters", "") != context.get("filters", ""):
            return False
        if metadata.get("columns") != context.get("columns"):
            return False
        if metadata.get("extra", {}) != context.get("extra", {}):
            return False
        meta_files = metadata.get("files") or []
        ctx_files = context.get("files") or []
        if len(meta_files) != len(ctx_files):
            return False
        meta_sorted = sorted(meta_files, key=lambda item: item["path"])  # type: ignore[index]
        ctx_sorted = sorted(ctx_files, key=lambda item: item["path"])
        for meta_file, ctx_file in zip(meta_sorted, ctx_sorted):
            if meta_file["path"] != ctx_file["path"]:
                return False
            if abs(float(meta_file.get("mtime", 0.0)) - float(ctx_file["mtime"])) > 1e-9:
                return False
        return True

    def _current_file_descriptors(self) -> List[Dict[str, object]]:
        files: List[Tuple[str, float]] = []
        for info in self.file_views.values():
            files.append((info["path"], float(info.get("mtime", 0.0))))
        return [
            {"path": path, "mtime": mtime}
            for path, mtime in sorted(files, key=lambda item: item[0])
        ]

    def _read_metadata(self, dataset_dir: Path) -> Optional[Dict[str, object]]:
        metadata_path = dataset_dir / "metadata.json"
        if not metadata_path.exists():
            return None
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except Exception:  # pragma: no cover - corrupted metadata
            return None
        return metadata

    def _write_metadata(self, dataset_dir: Path, metadata: Dict[str, object]) -> None:
        metadata_path = dataset_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

    @staticmethod
    def _drop_cached_dataset(dataset_dir: Path) -> None:
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir, ignore_errors=True)

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
