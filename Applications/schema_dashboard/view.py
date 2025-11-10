"""Frontend helper for rendering schema report summaries."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from urllib.request import Request, urlopen


FetchFn = Callable[[str], Dict[str, Any]]


@dataclass
class SchemaDashboard:
    """Load schema report data and render lightweight summaries."""

    base_url: str = "http://localhost:8000"
    fetcher: Optional[FetchFn] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    files: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def load(self) -> None:
        """Retrieve schema report JSON from the configured API."""

        fetch = self.fetcher or self._default_fetch
        payload = fetch(self._build_url("/reports/schema"))
        self.summary = payload.get("summary", {})
        self.files = payload.get("files", [])
        self.warnings = list(dict.fromkeys(self.summary.get("warnings", [])))

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def render_summary(self) -> str:
        """Return a multi-line string summarising the report."""

        if not self.summary:
            return "Schema report unavailable."
        lines = [
            f"Total files analysed: {self.summary.get('total_files', 0)}",
        ]
        columns = self.summary.get("columns", {})
        if columns:
            lines.append("Columns:")
            for name, info in sorted(columns.items()):
                coverage = info.get("coverage_pct", 0.0)
                anomalies = ", ".join(info.get("anomalies", [])) or "none"
                lines.append(f"- {name}: coverage {coverage:.1f}% (anomalies: {anomalies})")
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"* {warning}")
        return "\n".join(lines)

    def render_file_table(self) -> List[Dict[str, Any]]:
        """Return a simplified table of per-file statistics."""

        table: List[Dict[str, Any]] = []
        for file_entry in self.files:
            table.append(
                {
                    "path": file_entry.get("path"),
                    "row_count": file_entry.get("row_count", 0),
                    "warnings": ", ".join(file_entry.get("warnings", [])) or "none",
                }
            )
        return table

    # ------------------------------------------------------------------
    # Networking helpers
    # ------------------------------------------------------------------
    def _build_url(self, path: str) -> str:
        base = self.base_url.rstrip("/")
        suffix = path if path.startswith("/") else f"/{path}"
        return f"{base}{suffix}"

    def _default_fetch(self, url: str) -> Dict[str, Any]:
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request) as response:  # pragma: no cover - network integration
            data = response.read()
        return json.loads(data)
