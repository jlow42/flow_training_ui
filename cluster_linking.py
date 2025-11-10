from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.path import Path


@dataclass(frozen=True)
class SelectionSummary:
    """Compact summary of a linked selection."""

    total_rows: int
    cluster_counts: Dict[str, int]


def _format_numeric(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return ""
        return f"{float(value):.4g}"
    return str(value)


def select_points_within_path(
    dataframe: pd.DataFrame,
    x_col: str,
    y_col: str,
    vertices: Sequence[Tuple[float, float]],
) -> np.ndarray:
    """Return a boolean mask for points enclosed by a polygon."""

    if dataframe.empty or not vertices:
        return np.zeros(len(dataframe), dtype=bool)
    coords = np.column_stack(
        (
            dataframe[x_col].to_numpy(dtype=float, copy=False),
            dataframe[y_col].to_numpy(dtype=float, copy=False),
        )
    )
    polygon = Path(vertices)
    return polygon.contains_points(coords)


def filter_by_limits(
    dataframe: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
) -> pd.DataFrame:
    """Return the subset of rows within the provided axis limits."""

    if dataframe.empty:
        return dataframe.iloc[0:0]
    x_min, x_max = sorted(x_limits)
    y_min, y_max = sorted(y_limits)
    x_data = dataframe[x_col].to_numpy(dtype=float, copy=False)
    y_data = dataframe[y_col].to_numpy(dtype=float, copy=False)
    mask = (x_data >= x_min) & (x_data <= x_max) & (y_data >= y_min) & (y_data <= y_max)
    return dataframe.loc[mask]


def summarize_clusters(
    dataframe: pd.DataFrame,
    cluster_col: str = "cluster",
) -> SelectionSummary:
    """Aggregate per-cluster counts for a selection."""

    if dataframe is None or dataframe.empty:
        return SelectionSummary(total_rows=0, cluster_counts={})
    clusters = dataframe[cluster_col].astype(str)
    counts = clusters.value_counts(sort=False).sort_index()
    return SelectionSummary(total_rows=int(counts.sum()), cluster_counts=counts.to_dict())


def build_table_rows(
    dataframe: pd.DataFrame,
    x_col: str,
    y_col: str,
    max_rows: int = 200,
) -> Iterable[Tuple[str, str, str, str]]:
    """Yield tuple rows for a selection details table."""

    if dataframe is None or dataframe.empty:
        return []
    safe_cols = [col for col in (x_col, y_col) if col in dataframe.columns]
    if len(safe_cols) < 2 or "cluster" not in dataframe.columns:
        return []
    limited = dataframe[["cluster", x_col, y_col]].head(max_rows)
    rows = []
    for index, row in limited.iterrows():
        rows.append(
            (
                str(index),
                str(row["cluster"]),
                _format_numeric(row[x_col]),
                _format_numeric(row[y_col]),
            )
        )
    return rows


__all__ = [
    "SelectionSummary",
    "build_table_rows",
    "filter_by_limits",
    "select_points_within_path",
    "summarize_clusters",
]
