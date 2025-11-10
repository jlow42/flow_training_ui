import pandas as pd
from sklearn.preprocessing import StandardScaler

from clustering_ops import (
    incremental_update_assignments,
    merge_clusters,
    prepare_imported_assignments,
    split_cluster,
)


def _build_initial_state() -> tuple[dict, dict, dict, dict, dict, dict]:
    base_data = pd.DataFrame({"f1": [0.0, 1.0], "f2": [0.0, 1.0]})
    scaler = StandardScaler().fit(base_data)
    scaled = scaler.transform(base_data)
    assignments = {
        "run::kmeans": pd.DataFrame({"f1": base_data["f1"], "f2": base_data["f2"], "cluster": [0, 1]})
    }
    metadata = {
        "run::kmeans": {
            "method_key": "kmeans",
            "params": {"n_clusters": 2},
        }
    }
    incremental_state = {
        "run::kmeans": {
            "supports_incremental": True,
            "cluster_sizes": {"0": 1, "1": 1},
            "centroids_scaled": {
                "0": scaled[0].tolist(),
                "1": scaled[1].tolist(),
            },
        }
    }
    method_labels = {"run::kmeans": "KMeans"}
    scaler_params = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    return assignments, metadata, incremental_state, method_labels, scaler_params, base_data


def test_incremental_update_assignments_appends_rows():
    (
        assignments,
        metadata,
        incremental_state,
        method_labels,
        scaler_params,
        base_data,
    ) = _build_initial_state()

    new_dataset = pd.DataFrame({"f1": [0.0, 1.0, 0.1], "f2": [0.0, 1.0, 0.2]})

    result = incremental_update_assignments(
        assignments,
        metadata,
        incremental_state,
        scaler_params,
        base_data,
        new_dataset,
        ["f1", "f2"],
        method_labels,
    )

    updated = result.assignments["run::kmeans"]
    assert len(updated) == 3
    # New point is closer to cluster 0 centroid
    assert int(updated.iloc[-1]["cluster"]) == 0
    assert result.metadata["run::kmeans"]["cluster_sizes"]["0"] == 2
    assert result.metadata["run::kmeans"]["cluster_sizes"]["1"] == 1


def test_merge_clusters_combines_values():
    df = pd.DataFrame({"value": [1, 2, 3, 4], "cluster": ["A", "A", "B", "C"]})
    merged = merge_clusters(df, ["A", "B"], "merged")
    assert set(merged["cluster"]) == {"merged", "C"}
    assert (merged["cluster"] == "merged").sum() == 3


def test_split_cluster_creates_new_labels():
    df = pd.DataFrame(
        {
            "f1": [0.0, 0.1, 0.2, 0.3, 10.0],
            "f2": [0.0, 0.1, 0.2, 0.3, 10.0],
            "cluster": ["X", "X", "X", "X", "Y"],
        }
    )
    scaler = StandardScaler().fit(df[["f1", "f2"]])
    updated, _ = split_cluster(
        df,
        ["f1", "f2"],
        {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "X",
        2,
        "X_split_",
        random_state=0,
    )
    new_labels = sorted(label for label in updated["cluster"].unique() if label.startswith("X_split_"))
    assert len(new_labels) == 2
    assert "Y" in updated["cluster"].unique()


def test_prepare_imported_assignments_round_trip():
    data = pd.DataFrame(
        {
            "f1": [0.0, 1.0],
            "f2": [0.0, 1.0],
            "cluster_kmeans": [0, 1],
            "annotation_kmeans_label": ["alpha", "beta"],
        }
    )
    metadata = {
        "base": {"features": ["f1", "f2"]},
        "methods": {
            "run::kmeans": {
                "method_label": "KMeans",
                "column_name": "cluster_kmeans",
                "annotation_columns": [
                    {"name": "label", "column": "annotation_kmeans_label"}
                ],
            }
        },
        "incremental_state": {
            "run::kmeans": {"supports_incremental": True}
        },
        "scaler_params": {"mean": [0.5, 0.5], "scale": [0.5, 0.5]},
    }

    (
        assignments,
        metadata_map,
        method_labels,
        incremental_state,
        annotations,
        features,
        base_df,
        scaler_params,
        feature_matrix,
    ) = prepare_imported_assignments(data, metadata)

    assert set(assignments.keys()) == {"run::kmeans"}
    assert method_labels["run::kmeans"] == "KMeans"
    assert incremental_state["run::kmeans"]["supports_incremental"] is True
    assert annotations["run::kmeans"].iloc[0]["label"] in {"alpha", "beta"}
    assert features == ["f1", "f2"]
    assert scaler_params == metadata["scaler_params"]
    assert feature_matrix.shape == (2, 2)
