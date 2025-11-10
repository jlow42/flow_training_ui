import json
import sys
from pathlib import Path
from typing import Dict

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model_cards import EnvironmentSnapshot, ModelCardStore, build_model_card


def _sample_payload() -> Dict[str, object]:
    return {
        "metrics": {
            "accuracy": 0.91,
            "f1_macro": 0.87,
            "f1_weighted": 0.9,
            "report_dict": {
                "class_a": {
                    "precision": 0.9,
                    "recall": 0.88,
                    "f1-score": 0.89,
                    "support": 45,
                },
                "class_b": {
                    "precision": 0.92,
                    "recall": 0.9,
                    "f1-score": 0.91,
                    "support": 55,
                },
                "accuracy": {"support": 100},
                "macro avg": {
                    "precision": 0.91,
                    "recall": 0.89,
                    "f1-score": 0.9,
                    "support": 100,
                },
            },
        },
        "confusion_matrix": [[40, 5], [6, 49]],
        "cv_scores": [0.85, 0.9, 0.88, 0.87, 0.89],
        "cv_warning": "",
        "train_rows": 800,
        "test_rows": 200,
        "classes": ["class_a", "class_b"],
        "artifacts": {"explanation": "available"},
    }


def _snapshot() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(python_version="3.10", platform="test", libraries={"numpy": "1.0"})


def test_model_card_store_roundtrip(tmp_path: Path) -> None:
    payload = _sample_payload()
    training_config = {"model_params": {"n_estimators": 100}, "test_size": 0.2, "cv_folds": 5, "n_jobs": 2}
    card = build_model_card(
        model_name="Random Forest",
        payload=payload,
        features=["f1", "f2", "f3"],
        target="label",
        dataset_sources=["/data/file1.csv", "/data/file2.csv"],
        dataset_signature="sig-123",
        class_balance="balanced",
        tags=["baseline"],
        notes="Initial run",
        downsampling={"method": "Random", "value": "500"},
        training_config=training_config,
        environment_snapshot=_snapshot(),
    )

    store_path = tmp_path / "cards.json"
    store = ModelCardStore(store_path)
    store.add_card(card)

    loaded = store.get_card(card.id)
    assert loaded is not None
    assert pytest.approx(loaded.metrics.summary["accuracy"], rel=1e-6) == 0.91
    assert loaded.dataset.class_distribution["class_a"] == 45
    assert "class_b" in loaded.metrics.per_class

    export_path = tmp_path / "card_export.json"
    store.export_card(card.id, export_path)
    exported = json.loads(export_path.read_text(encoding="utf-8"))
    assert exported["id"] == card.id
    assert exported["dataset"]["signature"] == "sig-123"

    reloaded_store = ModelCardStore(store_path)
    assert reloaded_store.get_card(card.id) is not None


def test_model_card_store_orders_by_recency(tmp_path: Path) -> None:
    payload = _sample_payload()
    card_a = build_model_card(
        model_name="ModelA",
        payload=payload,
        features=["a"],
        target="label",
        dataset_sources=["/tmp/a.csv"],
        dataset_signature=None,
        class_balance=None,
        tags=[],
        notes="",
        downsampling=None,
        training_config={"model_params": {}},
        environment_snapshot=_snapshot(),
    )
    card_b = build_model_card(
        model_name="ModelB",
        payload=payload,
        features=["a"],
        target="label",
        dataset_sources=["/tmp/a.csv"],
        dataset_signature=None,
        class_balance=None,
        tags=[],
        notes="",
        downsampling=None,
        training_config={"model_params": {}},
        environment_snapshot=_snapshot(),
    )

    card_a.created_at = 10
    card_b.created_at = 20

    store = ModelCardStore(tmp_path / "cards.json")
    store.add_card(card_a)
    store.add_card(card_b)

    ids = [card.id for card in store.all_cards()]
    assert ids[0] == card_b.id
