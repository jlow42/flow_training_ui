"""Helper for appending CLI runs to the shared registry file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


class RunRegistry:
    """Simple JSON-backed registry shared with the GUI application."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        root = base_dir or Path(__file__).resolve().parent
        self.cache_dir = root / ".flow_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.cache_dir / "run_registry.json"

    def load(self) -> List[Dict[str, object]]:
        if not self.registry_path.exists():
            return []
        with self.registry_path.open("r", encoding="utf-8") as handle:
            try:
                data = json.load(handle)
            except json.JSONDecodeError:
                return []
        if isinstance(data, dict) and "runs" in data and isinstance(data["runs"], list):
            return list(data["runs"])
        if isinstance(data, list):
            return list(data)
        return []

    def append(self, record: Dict[str, object]) -> None:
        runs = self.load()
        runs.append(record)
        with self.registry_path.open("w", encoding="utf-8") as handle:
            json.dump({"runs": runs}, handle, indent=2)

