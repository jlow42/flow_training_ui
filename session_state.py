"""Session autosave helpers for Flow Training UI."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SessionSnapshot:
    """Serializable representation of the application's session state."""

    files: List[str]
    datasets: List[Dict[str, Any]]
    training: Dict[str, Any]
    saved_at: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "files": list(self.files),
            "datasets": [dict(dataset) for dataset in self.datasets],
            "training": dict(self.training),
            "saved_at": self.saved_at,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SessionSnapshot":
        files = [str(path) for path in payload.get("files", [])]
        datasets_raw = payload.get("datasets") or []
        datasets = [dict(dataset) for dataset in datasets_raw if isinstance(dataset, dict)]
        training_raw = payload.get("training") or {}
        training = dict(training_raw) if isinstance(training_raw, dict) else {}
        saved_at = float(payload.get("saved_at", time.time()))
        return cls(files=files, datasets=datasets, training=training, saved_at=saved_at)


class SessionStateStore:
    """Durable storage wrapper for session snapshots."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Optional[SessionSnapshot]:
        if not self.path.exists():
            return None
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError, TypeError):
            return None
        return SessionSnapshot.from_dict(payload)

    def save(self, snapshot: SessionSnapshot) -> None:
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = snapshot.to_dict()
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            tmp_path.replace(self.path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
