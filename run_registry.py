"""Persistent run registry helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class RunRegistry:
    """Thin wrapper around the on-disk registry JSON document."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._runs: List[Dict[str, Any]] = []
        self.load()

    @property
    def runs(self) -> List[Dict[str, Any]]:
        return list(self._runs)

    def load(self) -> None:
        if not self.path.exists():
            self._runs = []
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError, TypeError):
            self._runs = []
            return
        runs_raw = payload.get("runs") if isinstance(payload, dict) else []
        self._runs = [dict(run) for run in runs_raw if isinstance(run, dict)]

    def save(self) -> None:
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump({"runs": self._runs}, handle, indent=2)
            tmp_path.replace(self.path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def list_runs(self) -> List[Dict[str, Any]]:
        return sorted(self._runs, key=lambda item: item.get("timestamp", 0), reverse=True)

    def add_run(self, record: Dict[str, Any]) -> Dict[str, Any]:
        record_copy = dict(record)
        self._runs.append(record_copy)
        self.save()
        return record_copy

    def find(self, run_id: str) -> Optional[Dict[str, Any]]:
        for run in self._runs:
            if run.get("id") == run_id:
                return run
        return None

    def update_metadata(
        self,
        run_id: str,
        *,
        tags: Optional[Iterable[str]] = None,
        notes: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> bool:
        record = self.find(run_id)
        if not record:
            return False
        if tags is not None:
            record["tags"] = [str(tag) for tag in tags]
        if notes is not None:
            record["notes"] = notes
        if seed is not None:
            record["seed"] = int(seed)
        self.save()
        return True
