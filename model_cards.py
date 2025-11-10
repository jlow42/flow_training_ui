"""Utilities for capturing, storing, and exporting training model cards."""
from __future__ import annotations

import json
import platform
import sys
import time
import uuid
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _safe_float(value: object) -> Optional[float]:
    """Coerce a value to ``float`` when possible."""

    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        return None


def _safe_int(value: object) -> Optional[int]:
    """Coerce a value to ``int`` when possible."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        coerced = int(value)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        return None
    return coerced


def _sanitize(value: object) -> object:
    """Convert complex/numpy values into JSON-serialisable objects."""

    if isinstance(value, dict):
        return {str(key): _sanitize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # noqa: BLE001
            return str(value)
    if isinstance(value, (str, int, float)) or value is None:
        return value
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _module_version(module_name: str) -> Optional[str]:
    """Return the version string for a module if available."""

    try:
        module = import_module(module_name)
    except Exception:  # noqa: BLE001
        return None
    for attr in ("__version__", "VERSION", "__about__.__version__"):
        parts = attr.split(".")
        root = module
        try:
            for part in parts:
                root = getattr(root, part)
        except AttributeError:
            continue
        if isinstance(root, str):
            return root
    return None


@dataclass
class EnvironmentSnapshot:
    """Capture a lightweight view of the runtime environment."""

    python_version: str
    platform: str
    libraries: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def capture(extra_modules: Optional[Iterable[str]] = None) -> "EnvironmentSnapshot":
        """Collect version information for key dependencies."""

        baseline = [
            "numpy",
            "pandas",
            "scipy",
            "sklearn",
            "seaborn",
            "xgboost",
            "lightgbm",
            "torch",
        ]
        modules = list(dict.fromkeys([*baseline, *(extra_modules or [])]))
        versions: Dict[str, str] = {}
        for name in modules:
            version = _module_version(name)
            if version:
                versions[name] = version
        return EnvironmentSnapshot(
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            libraries=versions,
        )


@dataclass
class ModelCardMetrics:
    """Primary and per-class evaluation metrics."""

    summary: Dict[str, float] = field(default_factory=dict)
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion_matrix: List[List[float]] = field(default_factory=list)
    cv_scores: Optional[List[float]] = None
    cv_warning: str = ""

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ModelCardMetrics":
        return ModelCardMetrics(
            summary={k: float(v) for k, v in (data.get("summary") or {}).items()},
            per_class={
                str(label): {metric: float(value) for metric, value in metrics.items()}
                for label, metrics in (data.get("per_class") or {}).items()
            },
            confusion_matrix=[
                [float(item) for item in row]
                for row in (data.get("confusion_matrix") or [])
            ],
            cv_scores=[float(item) for item in data.get("cv_scores", [])]
            if data.get("cv_scores") is not None
            else None,
            cv_warning=str(data.get("cv_warning", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": _sanitize(self.summary),
            "per_class": _sanitize(self.per_class),
            "confusion_matrix": _sanitize(self.confusion_matrix),
            "cv_scores": _sanitize(self.cv_scores),
            "cv_warning": self.cv_warning,
        }


@dataclass
class ModelCardHyperparameters:
    """Hyperparameter and training configuration snapshot."""

    parameters: Dict[str, Any] = field(default_factory=dict)
    test_size: Optional[float] = None
    cv_folds: Optional[int] = None
    n_jobs: Optional[int] = None
    training_time: Optional[float] = None

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ModelCardHyperparameters":
        return ModelCardHyperparameters(
            parameters=dict(data.get("parameters", {})),
            test_size=_safe_float(data.get("test_size")),
            cv_folds=_safe_int(data.get("cv_folds")),
            n_jobs=_safe_int(data.get("n_jobs")),
            training_time=_safe_float(data.get("training_time")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameters": _sanitize(self.parameters),
            "test_size": self.test_size,
            "cv_folds": self.cv_folds,
            "n_jobs": self.n_jobs,
            "training_time": self.training_time,
        }


@dataclass
class DatasetMetadata:
    """Describe the dataset used during training."""

    features: List[str]
    target: str
    class_names: List[str] = field(default_factory=list)
    class_distribution: Dict[str, int] = field(default_factory=dict)
    train_rows: Optional[int] = None
    test_rows: Optional[int] = None
    sources: List[str] = field(default_factory=list)
    signature: Optional[str] = None
    downsampling: Optional[Dict[str, Any]] = None
    class_balance: Optional[str] = None

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "DatasetMetadata":
        return DatasetMetadata(
            features=list(data.get("features", [])),
            target=str(data.get("target", "")),
            class_names=list(data.get("class_names", [])),
            class_distribution={
                str(label): int(count)
                for label, count in (data.get("class_distribution") or {}).items()
            },
            train_rows=_safe_int(data.get("train_rows")),
            test_rows=_safe_int(data.get("test_rows")),
            sources=[str(path) for path in data.get("sources", [])],
            signature=data.get("signature"),
            downsampling=data.get("downsampling"),
            class_balance=data.get("class_balance"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "features": list(self.features),
            "target": self.target,
            "class_names": list(self.class_names),
            "class_distribution": _sanitize(self.class_distribution),
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "sources": list(self.sources),
            "signature": self.signature,
            "downsampling": _sanitize(self.downsampling),
            "class_balance": self.class_balance,
        }


@dataclass
class ModelCard:
    """Complete model card representation."""

    id: str
    created_at: float
    model_name: str
    metrics: ModelCardMetrics
    hyperparameters: ModelCardHyperparameters
    dataset: DatasetMetadata
    environment: EnvironmentSnapshot
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    artifacts: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ModelCard":
        return ModelCard(
            id=str(data.get("id", uuid.uuid4().hex)),
            created_at=float(data.get("created_at", time.time())),
            model_name=str(data.get("model_name", "")),
            metrics=ModelCardMetrics.from_dict(data.get("metrics", {})),
            hyperparameters=ModelCardHyperparameters.from_dict(
                data.get("hyperparameters", {})
            ),
            dataset=DatasetMetadata.from_dict(data.get("dataset", {})),
            environment=EnvironmentSnapshot(
                python_version=str(data.get("environment", {}).get("python_version", "")),
                platform=str(data.get("environment", {}).get("platform", "")),
                libraries=dict(data.get("environment", {}).get("libraries", {})),
            ),
            tags=list(data.get("tags", [])),
            notes=str(data.get("notes", "")),
            artifacts=dict(data.get("artifacts", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "model_name": self.model_name,
            "metrics": self.metrics.to_dict(),
            "hyperparameters": self.hyperparameters.to_dict(),
            "dataset": self.dataset.to_dict(),
            "environment": {
                "python_version": self.environment.python_version,
                "platform": self.environment.platform,
                "libraries": dict(self.environment.libraries),
            },
            "tags": list(self.tags),
            "notes": self.notes,
            "artifacts": _sanitize(self.artifacts),
        }


class ModelCardStore:
    """Persist and retrieve :class:`ModelCard` instances."""

    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._cards: List[ModelCard] = []
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            self._cards = []
            return
        try:
            with self.storage_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:  # noqa: BLE001
            self._cards = []
            return
        raw_cards = data.get("cards") if isinstance(data, dict) else data
        if not isinstance(raw_cards, list):
            raw_cards = []
        self._cards = [ModelCard.from_dict(item) for item in raw_cards]

    def _persist(self) -> None:
        payload = {"cards": [card.to_dict() for card in self._cards]}
        with self.storage_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def all_cards(self) -> List[ModelCard]:
        return sorted(self._cards, key=lambda card: card.created_at, reverse=True)

    def add_card(self, card: ModelCard) -> None:
        self._cards.append(card)
        self._persist()

    def get_card(self, card_id: str) -> Optional[ModelCard]:
        for card in self._cards:
            if card.id == card_id:
                return card
        return None

    def export_card(self, card_id: str, destination: Path) -> Path:
        card = self.get_card(card_id)
        if card is None:
            raise KeyError(f"Unknown model card id: {card_id}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(card.to_dict(), handle, indent=2)
        return destination


def build_model_card(
    *,
    model_name: str,
    payload: Dict[str, Any],
    features: Sequence[str],
    target: str,
    dataset_sources: Sequence[str],
    dataset_signature: Optional[str],
    class_balance: Optional[str],
    tags: Sequence[str],
    notes: str,
    downsampling: Optional[Dict[str, Any]],
    training_config: Dict[str, Any],
    environment_snapshot: Optional[EnvironmentSnapshot] = None,
) -> ModelCard:
    """Assemble a :class:`ModelCard` from training artefacts."""

    metrics_dict = dict(payload.get("metrics", {}))
    summary: Dict[str, float] = {}
    for key in ("accuracy", "f1_macro", "f1_weighted"):
        value = _safe_float(metrics_dict.get(key))
        if value is not None:
            summary[key] = value

    report_dict = metrics_dict.get("report_dict") if isinstance(metrics_dict, dict) else None
    per_class: Dict[str, Dict[str, float]] = {}
    class_distribution: Dict[str, int] = {}
    if isinstance(report_dict, dict):
        for label, class_metrics in report_dict.items():
            if not isinstance(class_metrics, dict):
                continue
            label_str = str(label)
            class_entry: Dict[str, float] = {}
            for metric_name in ("precision", "recall", "f1-score"):
                value = _safe_float(class_metrics.get(metric_name))
                if value is not None:
                    class_entry[metric_name] = value
            if class_entry:
                per_class[label_str] = class_entry
            support = _safe_int(class_metrics.get("support"))
            if support is not None and label_str not in {"accuracy", "macro avg", "weighted avg"}:
                class_distribution[label_str] = support

    confusion_matrix = payload.get("confusion_matrix")
    confusion_list: List[List[float]] = []
    if confusion_matrix is not None:
        confusion_list = _sanitize(confusion_matrix)  # type: ignore[assignment]
        if not isinstance(confusion_list, list):
            confusion_list = []

    cv_scores_raw = payload.get("cv_scores")
    cv_scores: Optional[List[float]]
    if cv_scores_raw is None:
        cv_scores = None
    else:
        scores = _sanitize(cv_scores_raw)
        if isinstance(scores, list):
            cleaned_scores: List[float] = []
            for item in scores:
                value = _safe_float(item)
                if value is not None:
                    cleaned_scores.append(value)
            cv_scores = cleaned_scores
        else:
            cv_scores = None

    dataset = DatasetMetadata(
        features=list(features),
        target=target,
        class_names=[str(label) for label in (payload.get("classes") or [])],
        class_distribution=class_distribution,
        train_rows=_safe_int(payload.get("train_rows")),
        test_rows=_safe_int(payload.get("test_rows")),
        sources=[str(path) for path in dataset_sources],
        signature=dataset_signature,
        downsampling=downsampling,
        class_balance=class_balance,
    )

    hyperparameters = ModelCardHyperparameters(
        parameters=dict(training_config.get("model_params", {})),
        test_size=_safe_float(training_config.get("test_size")),
        cv_folds=_safe_int(training_config.get("cv_folds")),
        n_jobs=_safe_int(training_config.get("n_jobs")),
        training_time=_safe_float(payload.get("training_time")),
    )

    environment = environment_snapshot or EnvironmentSnapshot.capture()

    card = ModelCard(
        id=uuid.uuid4().hex,
        created_at=time.time(),
        model_name=model_name,
        metrics=ModelCardMetrics(
            summary=summary,
            per_class=per_class,
            confusion_matrix=confusion_list,
            cv_scores=cv_scores,
            cv_warning=str(payload.get("cv_warning", "")),
        ),
        hyperparameters=hyperparameters,
        dataset=dataset,
        environment=environment,
        tags=list(tags),
        notes=notes,
        artifacts=dict(payload.get("artifacts", {})),
    )
    return card

