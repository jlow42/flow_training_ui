"""Utility helpers for clustering state management and transformations.

These helpers intentionally avoid any tkinter dependencies so that they can be
tested in isolation. The main application wires them into the UI layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


@dataclass
class IncrementalUpdateResult:
    """Container for incremental update outcomes."""

    assignments: Dict[str, pd.DataFrame]
    metadata: Dict[str, Dict[str, object]]
    incremental_state: Dict[str, Dict[str, object]]
    summary: List[Dict[str, object]]
    clusters: List[Dict[str, object]]
    feature_matrix: np.ndarray


def _safe_scale(values: Sequence[float]) -> np.ndarray:
    scale = np.asarray(values, dtype=float)
    scale[scale == 0] = 1.0
    return scale


def _ensure_features_exist(dataset: pd.DataFrame, features: Sequence[str]) -> None:
    missing = [feature for feature in features if feature not in dataset.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required feature columns: " + ", ".join(missing)
        )


def scale_feature_matrix(
    dataset: pd.DataFrame, features: Sequence[str], mean: Sequence[float], scale: Sequence[float]
) -> np.ndarray:
    """Scale the feature matrix using a pre-computed mean/scale."""

    _ensure_features_exist(dataset, features)
    matrix = dataset.loc[:, list(features)].to_numpy(dtype=float, copy=False)
    mean_arr = np.asarray(mean, dtype=float)
    if mean_arr.shape[0] != matrix.shape[1]:
        raise ValueError("Scaler mean vector shape does not match feature matrix.")
    scale_arr = _safe_scale(scale)
    if scale_arr.shape[0] != matrix.shape[1]:
        raise ValueError("Scaler scale vector shape does not match feature matrix.")
    return (matrix - mean_arr) / scale_arr


def _validate_dataset_alignment(
    previous: pd.DataFrame,
    new: pd.DataFrame,
    features: Sequence[str],
    tolerance: float = 1e-8,
) -> int:
    if previous.empty:
        raise ValueError("Previous clustering dataset is unavailable.")
    _ensure_features_exist(previous, features)
    _ensure_features_exist(new, features)

    old_rows = previous.shape[0]
    if new.shape[0] <= old_rows:
        raise ValueError("No new rows detected for incremental clustering.")

    old_matrix = previous.loc[:, list(features)].to_numpy(dtype=float, copy=False)
    new_prefix = new.loc[:, list(features)].iloc[:old_rows].to_numpy(dtype=float, copy=False)
    if old_matrix.shape != new_prefix.shape:
        raise ValueError("Existing rows changed shape; incremental update aborted.")

    if not np.allclose(old_matrix, new_prefix, atol=tolerance, equal_nan=True):
        raise ValueError(
            "Existing rows differ from the prior clustering dataset; run a full clustering pass."
        )

    return old_rows


def _compute_cluster_counts(assignment_df: pd.DataFrame) -> Mapping[object, int]:
    if "cluster" not in assignment_df.columns:
        raise ValueError("Assignment dataframe is missing the 'cluster' column.")
    counts = (
        assignment_df["cluster"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )
    return counts


def _compute_scaled_centroids(
    assignment_df: pd.DataFrame,
    scaled_matrix: np.ndarray,
) -> Dict[str, List[float]]:
    clusters = assignment_df["cluster"].to_numpy()
    if scaled_matrix.shape[0] != clusters.shape[0]:
        raise ValueError("Scaled matrix and assignments are misaligned.")
    centroids: Dict[str, List[float]] = {}
    for cluster_value in pd.unique(clusters):
        mask = clusters == cluster_value
        if not np.any(mask):
            continue
        centroid = scaled_matrix[mask].mean(axis=0)
        centroids[str(cluster_value)] = centroid.tolist()
    return centroids


def incremental_update_assignments(
    assignments: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Mapping[str, object]],
    incremental_state: Mapping[str, Mapping[str, object]],
    scaler_params: Mapping[str, Sequence[float]],
    previous_dataset: pd.DataFrame,
    new_dataset: pd.DataFrame,
    features: Sequence[str],
    method_labels: Mapping[str, str],
) -> IncrementalUpdateResult:
    """Apply an incremental update for the provided clustering assignments."""

    if not assignments:
        raise ValueError("No clustering assignments available for incremental update.")
    if not scaler_params:
        raise ValueError("Scaler parameters are required for incremental clustering updates.")

    mean = scaler_params.get("mean")
    scale = scaler_params.get("scale")
    if mean is None or scale is None:
        raise ValueError("Scaler parameters must include 'mean' and 'scale'.")

    old_rows = _validate_dataset_alignment(previous_dataset, new_dataset, features)
    feature_matrix = scale_feature_matrix(new_dataset, features, mean, scale)
    if feature_matrix.shape[0] <= old_rows:
        raise ValueError("Incremental update detected zero new rows.")

    new_assignments: Dict[str, pd.DataFrame] = {}
    new_metadata: Dict[str, Dict[str, object]] = {}
    new_state: Dict[str, Dict[str, object]] = {}
    summary_rows: List[Dict[str, object]] = []
    cluster_rows: List[Dict[str, object]] = []

    base_tail_df = new_dataset.iloc[old_rows:].copy()
    feature_tail = feature_matrix[old_rows:, :]

    for run_key, assignment_df in assignments.items():
        if run_key not in incremental_state:
            raise ValueError(f"Incremental state missing for run '{run_key}'.")
        state_entry = dict(incremental_state[run_key])
        supports_incremental = bool(state_entry.get("supports_incremental"))
        if not supports_incremental:
            raise ValueError(
                f"Run '{run_key}' does not support incremental updates; rerun clustering."
            )

        centroids = state_entry.get("centroids_scaled") or {}
        if not centroids:
            raise ValueError(f"Centroid data is unavailable for run '{run_key}'.")

        existing_clusters = assignment_df["cluster"].unique()
        lookup = {str(value): value for value in existing_clusters}
        centroid_labels: List[object] = []
        centroid_matrix: List[List[float]] = []
        for key, centroid in centroids.items():
            if key not in lookup:
                raise ValueError(
                    f"Centroid '{key}' does not correspond to an existing cluster for '{run_key}'."
                )
            centroid_labels.append(lookup[key])
            centroid_matrix.append(list(centroid))
        centroid_array = np.asarray(centroid_matrix, dtype=float)
        if centroid_array.size == 0:
            raise ValueError(f"No centroids available for run '{run_key}'.")

        distances = np.linalg.norm(
            feature_tail[:, np.newaxis, :] - centroid_array[np.newaxis, :, :], axis=2
        )
        closest = distances.argmin(axis=1)
        assigned_clusters = [centroid_labels[idx] for idx in closest]

        updated_df = pd.concat(
            [assignment_df, base_tail_df.assign(cluster=assigned_clusters)],
            ignore_index=True,
        )
        new_assignments[run_key] = updated_df

        counts = _compute_cluster_counts(updated_df)
        metadata_entry = dict(metadata.get(run_key, {}))
        metadata_entry.update(
            {
                "cluster_sizes": {str(k): int(v) for k, v in counts.items()},
                "cluster_count": len(counts),
                "rows_used": int(updated_df.shape[0]),
            }
        )
        new_metadata[run_key] = metadata_entry

        state_entry.update(
            {
                "cluster_sizes": metadata_entry["cluster_sizes"],
                "rows": int(updated_df.shape[0]),
            }
        )
        state_entry["centroids_scaled"] = _compute_scaled_centroids(
            updated_df, feature_matrix
        )
        new_state[run_key] = state_entry

        method_label = method_labels.get(run_key, run_key)
        summary_rows.append(
            {
                "method_key": metadata_entry.get("method_key", run_key),
                "method_label": method_label,
                "run_key": run_key,
                "cluster_count": len(counts),
                "rows": int(updated_df.shape[0]),
            }
        )
        for cluster_value, cluster_size in counts.items():
            cluster_rows.append(
                {
                    "run_key": run_key,
                    "method_key": metadata_entry.get("method_key", run_key),
                    "method_label": method_label,
                    "cluster": str(cluster_value),
                    "count": int(cluster_size),
                }
            )

    return IncrementalUpdateResult(
        assignments=new_assignments,
        metadata=new_metadata,
        incremental_state=new_state,
        summary=summary_rows,
        clusters=cluster_rows,
        feature_matrix=feature_matrix,
    )


def merge_clusters(assignment_df: pd.DataFrame, clusters: Sequence[object], new_label: object) -> pd.DataFrame:
    if "cluster" not in assignment_df.columns:
        raise ValueError("Assignment dataframe must contain a 'cluster' column for merging.")
    if len(clusters) < 2:
        raise ValueError("Select at least two clusters to merge.")
    updated = assignment_df.copy()
    updated.loc[updated["cluster"].isin(clusters), "cluster"] = new_label
    return updated


def split_cluster(
    assignment_df: pd.DataFrame,
    features: Sequence[str],
    scaler_params: Optional[Mapping[str, Sequence[float]]],
    cluster_label: object,
    n_splits: int,
    prefix: str,
    random_state: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    if n_splits < 2:
        raise ValueError("Split count must be at least 2.")
    if "cluster" not in assignment_df.columns:
        raise ValueError("Assignment dataframe must contain a 'cluster' column for splitting.")
    mask = assignment_df["cluster"] == cluster_label
    if not mask.any():
        raise ValueError(f"Cluster '{cluster_label}' not found in assignment dataframe.")

    if scaler_params and scaler_params.get("mean") is not None and scaler_params.get("scale") is not None:
        scaled = scale_feature_matrix(
            assignment_df, features, scaler_params["mean"], scaler_params["scale"]
        )
    else:
        matrix = assignment_df.loc[:, list(features)].to_numpy(dtype=float, copy=False)
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0, ddof=0)
        scaled = scale_feature_matrix(assignment_df, features, mean, scale)

    subset = scaled[mask]
    if subset.shape[0] < n_splits:
        raise ValueError("Not enough rows to split the cluster into the requested parts.")

    model = KMeans(n_clusters=n_splits, random_state=random_state, n_init=10)
    split_labels = model.fit_predict(subset)

    updated = assignment_df.copy()
    for idx, sub_label in zip(updated.index[mask], split_labels):
        updated.at[idx, "cluster"] = f"{prefix}{sub_label + 1}"

    return updated, scaled


def _sanitize(value: str) -> str:
    sanitized = value.strip().lower()
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in sanitized)
    sanitized = sanitized.strip("_")
    return sanitized or "value"


def prepare_imported_assignments(
    dataframe: pd.DataFrame, metadata: Mapping[str, object]
) -> Tuple[
    Dict[str, pd.DataFrame],
    Dict[str, Dict[str, object]],
    Dict[str, str],
    Dict[str, Dict[str, object]],
    Dict[str, pd.DataFrame],
    List[str],
    pd.DataFrame,
    Optional[Dict[str, Sequence[float]]],
    np.ndarray,
]:
    if not isinstance(metadata, Mapping):
        raise ValueError("Metadata must be a mapping when importing clustering assignments.")
    base_meta = metadata.get("base")
    methods_meta = metadata.get("methods")
    if not isinstance(base_meta, Mapping) or not isinstance(methods_meta, Mapping):
        raise ValueError("Metadata is missing base or methods information.")

    features = base_meta.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("Metadata does not describe clustering feature columns.")

    column_map: Dict[str, str] = {}
    method_labels: Dict[str, str] = {}
    assignments: Dict[str, pd.DataFrame] = {}
    metadata_map: Dict[str, Dict[str, object]] = {}
    annotations: Dict[str, pd.DataFrame] = {}

    cluster_columns: List[str] = []
    all_annotation_columns: List[str] = []

    for run_key, info in methods_meta.items():
        if not isinstance(info, Mapping):
            raise ValueError("Malformed method entry in metadata.")
        method_label = str(info.get("method_label", run_key))
        method_labels[run_key] = method_label
        column_name = info.get("column_name")
        if not column_name:
            column_name = f"cluster_{_sanitize(method_label)}"
        if column_name not in dataframe.columns:
            raise ValueError(
                f"Assignment column '{column_name}' is missing from the provided dataframe."
            )
        column_map[run_key] = column_name
        cluster_columns.append(column_name)
        metadata_map[run_key] = dict(info)

        annotation_meta = info.get("annotation_columns")
        if isinstance(annotation_meta, list):
            for entry in annotation_meta:
                sanitized = entry.get("column") if isinstance(entry, Mapping) else None
                if sanitized:
                    all_annotation_columns.append(sanitized)

    base_columns = [
        column
        for column in dataframe.columns
        if column not in cluster_columns and column not in all_annotation_columns
    ]
    base_df = dataframe.loc[:, base_columns].copy()

    for run_key, column_name in column_map.items():
        assignment_df = base_df.copy()
        assignment_df["cluster"] = dataframe[column_name]
        assignments[run_key] = assignment_df

        info = metadata_map[run_key]
        annotation_meta = info.get("annotation_columns")
        annotation_entries: List[Dict[str, object]] = []
        if isinstance(annotation_meta, list):
            for entry in annotation_meta:
                if not isinstance(entry, Mapping):
                    continue
                sanitized = entry.get("column")
                original = entry.get("name")
                if not sanitized or sanitized not in dataframe.columns or not original:
                    continue
                annotation_entries.append({"name": original, "column": sanitized})

        if annotation_entries:
            cluster_series = dataframe[column_name]
            annotation_rows: Dict[str, List[object]] = {"cluster": []}
            for ann in annotation_entries:
                annotation_rows[ann["name"]] = []
            seen: Dict[str, bool] = {}
            for cluster_value, row in zip(cluster_series, dataframe.itertuples(index=False)):
                cluster_key = str(cluster_value)
                if cluster_key in seen:
                    continue
                seen[cluster_key] = True
                annotation_rows["cluster"].append(cluster_key)
                for ann in annotation_entries:
                    annotation_rows[ann["name"]].append(getattr(row, ann["column"], ""))
            annotations[run_key] = pd.DataFrame(annotation_rows)

    scaler_params = metadata.get("scaler_params")
    feature_matrix: np.ndarray
    if isinstance(scaler_params, Mapping) and scaler_params.get("mean") is not None and scaler_params.get("scale") is not None:
        feature_matrix = scale_feature_matrix(base_df, features, scaler_params["mean"], scaler_params["scale"])
    else:
        scaler_params = None
        feature_matrix = base_df.loc[:, list(features)].to_numpy(dtype=float, copy=False)

    incremental_state = metadata.get("incremental_state")
    if not isinstance(incremental_state, Mapping):
        incremental_state = {}

    return (
        assignments,
        metadata_map,
        method_labels,
        dict(incremental_state),
        annotations,
        list(features),
        base_df,
        scaler_params if scaler_params is not None else None,
        feature_matrix,
    )

