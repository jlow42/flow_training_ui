"""Schema inference and reporting for tabular ingestion."""
from __future__ import annotations

import datetime as _dt
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)


@dataclass
class ColumnReport:
    """Summary information for a single column."""

    name: str
    inferred_type: str
    non_null: int
    missing: int
    unique: int
    stats: Dict[str, Optional[float]] = field(default_factory=dict)
    sample_values: List[Any] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "inferred_type": self.inferred_type,
            "non_null": self.non_null,
            "missing": self.missing,
            "unique": self.unique,
            "stats": self.stats,
            "sample_values": self.sample_values,
            "anomalies": self.anomalies,
        }


@dataclass
class FileReport:
    """Summary for a single ingested file."""

    path: str
    row_count: int
    columns: List[ColumnReport]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "row_count": self.row_count,
            "columns": [column.to_dict() for column in self.columns],
            "warnings": self.warnings,
        }


@dataclass
class SchemaReport:
    """Aggregate schema report for all ingested files."""

    files: List[FileReport]
    summary: Dict[str, Any]
    generated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "files": [file_report.to_dict() for file_report in self.files],
            "summary": self.summary,
            "generated_at": self.generated_at,
        }


class SchemaAnalyzer:
    """Analyze CSV files to infer schema and highlight anomalies."""

    def __init__(
        self,
        *,
        sample_rows: Optional[int] = None,
        high_missing_threshold: float = 0.2,
        high_cardinality_threshold: float = 0.8,
        extreme_zscore: float = 6.0,
    ) -> None:
        self.sample_rows = sample_rows
        self.high_missing_threshold = high_missing_threshold
        self.high_cardinality_threshold = high_cardinality_threshold
        self.extreme_zscore = extreme_zscore

    def generate_report(self, files: Sequence[Path]) -> SchemaReport:
        file_reports: List[FileReport] = []
        column_summary: Dict[str, Dict[str, Any]] = {}
        global_warnings: List[str] = []

        for path in files:
            file_report = self._analyze_file(path)
            file_reports.append(file_report)
            for column in file_report.columns:
                summary_entry = column_summary.setdefault(
                    column.name,
                    {
                        "inferred_type": column.inferred_type,
                        "files_present": 0,
                        "anomalies": [],
                    },
                )
                summary_entry["files_present"] += 1
                summary_entry["anomalies"].extend(column.anomalies)
                if summary_entry["inferred_type"] != column.inferred_type:
                    summary_entry.setdefault("dtype_conflicts", set()).add(column.inferred_type)
            global_warnings.extend(file_report.warnings)

        summary_payload = self._build_summary(column_summary, len(files), global_warnings)
        timestamp = _dt.datetime.now(_dt.timezone.utc).isoformat()
        return SchemaReport(
            files=file_reports,
            summary=summary_payload,
            generated_at=timestamp,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_summary(
        self,
        column_summary: Dict[str, Dict[str, Any]],
        total_files: int,
        global_warnings: Iterable[str],
    ) -> Dict[str, Any]:
        columns_payload: Dict[str, Any] = {}
        aggregated_warnings: List[str] = list(dict.fromkeys(global_warnings))

        for name, data in column_summary.items():
            entry = {
                "inferred_type": data["inferred_type"],
                "files_present": data["files_present"],
                "coverage_pct": (data["files_present"] / total_files * 100.0)
                if total_files
                else 0.0,
                "anomalies": sorted(set(data.get("anomalies", []))),
            }
            if data.get("dtype_conflicts"):
                entry["dtype_conflicts"] = sorted(data["dtype_conflicts"])
            columns_payload[name] = entry

        aggregated_warnings.extend(
            warning
            for warning in self._detect_summary_warnings(column_summary, total_files)
            if warning not in aggregated_warnings
        )
        return {
            "total_files": total_files,
            "columns": columns_payload,
            "warnings": aggregated_warnings,
        }

    def _detect_summary_warnings(
        self, column_summary: Dict[str, Dict[str, Any]], total_files: int
    ) -> List[str]:
        warnings: List[str] = []
        if not total_files:
            return warnings
        for name, data in column_summary.items():
            if data["files_present"] == total_files:
                continue
            coverage_pct = data["files_present"] / total_files
            if coverage_pct < 0.5:
                warnings.append(f"Column '{name}' is only present in {coverage_pct*100:.1f}% of files")
        return warnings

    def _analyze_file(self, path: Path) -> FileReport:
        if not path.exists():
            return FileReport(
                path=str(path),
                row_count=0,
                columns=[],
                warnings=["file_missing"],
            )

        frame = pd.read_csv(path, nrows=self.sample_rows)
        row_count = len(frame.index)
        columns: List[ColumnReport] = []
        warnings: List[str] = []

        if row_count == 0:
            warnings.append("empty_file")

        for column_name in frame.columns:
            series = frame[column_name]
            column_report = self._analyze_column(series, column_name, row_count)
            columns.append(column_report)
            warnings.extend(column_report.anomalies)

        return FileReport(
            path=str(path),
            row_count=row_count,
            columns=columns,
            warnings=sorted(set(warnings)),
        )

    def _analyze_column(
        self, series: pd.Series, name: str, row_count: int
    ) -> ColumnReport:
        non_null = int(series.notna().sum())
        missing = int(row_count - non_null)
        unique = int(series.nunique(dropna=True))

        inferred_type = self._infer_type(series)
        stats: Dict[str, Optional[float]] = {}
        anomalies: List[str] = []

        missing_pct = (missing / row_count) if row_count else 0.0
        if missing_pct > self.high_missing_threshold:
            anomalies.append("high_missing_rate")

        if inferred_type == "numeric":
            numeric_series = pd.to_numeric(series, errors="coerce")
            non_null_numeric = numeric_series.dropna()
            stats = self._numeric_stats(non_null_numeric)
            anomalies.extend(self._numeric_anomalies(numeric_series, stats))
        elif inferred_type == "datetime":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Could not infer format",
                    category=UserWarning,
                )
                datetime_series = pd.to_datetime(series, errors="coerce", utc=True)
            non_null_datetime = datetime_series.dropna()
            if not non_null_datetime.empty:
                stats = {
                    "min": non_null_datetime.min().timestamp(),
                    "max": non_null_datetime.max().timestamp(),
                }
        else:
            stats = self._categorical_stats(series)
            anomalies.extend(self._categorical_anomalies(series, non_null, unique))

        sample_values = self._sample_values(series)
        return ColumnReport(
            name=name,
            inferred_type=inferred_type,
            non_null=non_null,
            missing=missing,
            unique=unique,
            stats=stats,
            sample_values=sample_values,
            anomalies=sorted(set(anomalies)),
        )

    def _infer_type(self, series: pd.Series) -> str:
        if is_numeric_dtype(series):
            return "numeric"
        if is_datetime64_any_dtype(series):
            return "datetime"
        if is_bool_dtype(series):
            return "boolean"
        # Attempt coercion for numeric
        coerced = pd.to_numeric(series.dropna(), errors="coerce")
        if not coerced.dropna().empty and float(coerced.dropna().shape[0]) >= 0.9 * max(1, series.dropna().shape[0]):
            return "numeric"
        # Attempt coercion for datetime
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format",
                category=UserWarning,
            )
            coerced_dt = pd.to_datetime(series.dropna(), errors="coerce", utc=True)
        if not coerced_dt.dropna().empty and float(coerced_dt.dropna().shape[0]) >= 0.9 * max(1, series.dropna().shape[0]):
            return "datetime"
        return "categorical"

    def _numeric_stats(self, series: pd.Series) -> Dict[str, Optional[float]]:
        if series.empty:
            return {"mean": None, "std": None, "min": None, "max": None, "q1": None, "q3": None}
        mean = float(series.mean()) if not np.isnan(series.mean()) else None
        std = float(series.std(ddof=0)) if not np.isnan(series.std(ddof=0)) else None
        quantiles = series.quantile([0.25, 0.75])
        q1 = float(quantiles.iloc[0]) if not np.isnan(quantiles.iloc[0]) else None
        q3 = float(quantiles.iloc[1]) if not np.isnan(quantiles.iloc[1]) else None
        minimum = float(series.min()) if not np.isnan(series.min()) else None
        maximum = float(series.max()) if not np.isnan(series.max()) else None
        return {
            "mean": mean,
            "std": std,
            "min": minimum,
            "max": maximum,
            "q1": q1,
            "q3": q3,
        }

    def _numeric_anomalies(self, series: pd.Series, stats: Dict[str, Optional[float]]) -> List[str]:
        anomalies: List[str] = []
        std = stats.get("std")
        mean = stats.get("mean")
        maximum = stats.get("max")
        minimum = stats.get("min")
        q1 = stats.get("q1")
        q3 = stats.get("q3")

        non_null = series.dropna()
        if non_null.empty:
            return anomalies
        if std is not None and std == 0:
            anomalies.append("no_variation")
        if std is not None and std > 0 and maximum is not None and mean is not None:
            zscore = (maximum - mean) / std
            if zscore > self.extreme_zscore:
                anomalies.append("extreme_upper_values")
        if std is not None and std > 0 and minimum is not None and mean is not None:
            zscore = (mean - minimum) / std
            if zscore > self.extreme_zscore:
                anomalies.append("extreme_lower_values")
        if q1 is not None and q3 is not None:
            iqr = q3 - q1
            if iqr > 0:
                upper = q3 + 1.5 * iqr
                lower = q1 - 1.5 * iqr
                if maximum is not None and maximum > upper:
                    anomalies.append("upper_outliers")
                if minimum is not None and minimum < lower:
                    anomalies.append("lower_outliers")
        return anomalies

    def _categorical_stats(self, series: pd.Series) -> Dict[str, Optional[float]]:
        top = series.mode(dropna=True)
        top_value = top.iloc[0] if not top.empty else None
        return {"top": top_value}

    def _categorical_anomalies(
        self, series: pd.Series, non_null: int, unique: int
    ) -> List[str]:
        anomalies: List[str] = []
        if non_null == 0:
            return anomalies
        if unique == 1:
            anomalies.append("single_value")
        unique_ratio = unique / non_null if non_null else 0.0
        if unique_ratio > self.high_cardinality_threshold:
            anomalies.append("high_cardinality")
        return anomalies

    def _sample_values(self, series: pd.Series, sample_size: int = 3) -> List[Any]:
        if series.empty:
            return []
        dropna = series.dropna().unique()
        if len(dropna) == 0:
            return []
        selected = dropna[:sample_size]
        return [self._coerce_jsonable(value) for value in selected]

    @staticmethod
    def _coerce_jsonable(value: Any) -> Any:
        if isinstance(value, (np.generic,)):
            return value.item()
        if isinstance(value, _dt.datetime):
            return value.isoformat()
        return value
