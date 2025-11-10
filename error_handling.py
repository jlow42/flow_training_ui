"""Utilities for presenting actionable error guidance in the UI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class ErrorRecord:
    """Captured UI error along with suggested guidance."""

    title: str
    message: str
    guidance: str
    timestamp: datetime
    details: Optional[str] = None

    @property
    def formatted_message(self) -> str:
        """Return a formatted message including actionable guidance."""

        guidance_block = f"Guidance: {self.guidance}" if self.guidance else ""
        detail_block = f"\nDetails: {self.details}" if self.details else ""
        parts = [self.message]
        if guidance_block:
            parts.append("\n\n" + guidance_block)
        if detail_block:
            parts.append(detail_block)
        return "".join(parts)


class ErrorManager:
    """Maintain structured error records for display within the UI."""

    def __init__(self, *, max_entries: int = 100) -> None:
        self.max_entries = max_entries
        self._records: List[ErrorRecord] = []
        self._log_path: Optional[Path] = None

    def set_log_path(self, log_path: Path) -> None:
        """Associate an application log path for troubleshooting guidance."""

        self._log_path = log_path

    def register_error(
        self,
        title: str,
        message: str,
        *,
        details: Optional[str] = None,
        hints: Optional[Iterable[str]] = None,
    ) -> ErrorRecord:
        """Capture an error and derive helpful remediation steps."""

        normalized = f"{title} {message}".lower()
        suggestions: List[str] = list(hints or [])

        if "csv" in normalized:
            suggestions.append(
                "Confirm the file is a valid CSV and not locked by another application."
            )
        if "cluster" in normalized:
            suggestions.append(
                "Review clustering feature selections and ensure required packages are installed."
            )
        if "model" in normalized or "train" in normalized:
            suggestions.append(
                "Verify training parameters and ensure the dataset contains enough labeled rows."
            )
        if "downsample" in normalized:
            suggestions.append(
                "Adjust the downsampling configuration or disable it to isolate the issue."
            )
        if "umap" in normalized:
            suggestions.append("Reduce the feature count or sample size before rerunning UMAP.")
        if not suggestions:
            suggestions.append("Review the input values and retry the operation.")

        if self._log_path is not None:
            suggestions.append(f"Check the application log at '{self._log_path}' for details.")

        guidance = " ".join(suggestions)
        record = ErrorRecord(
            title=title,
            message=message,
            guidance=guidance,
            details=details,
            timestamp=datetime.now(UTC),
        )
        self._records.append(record)
        self._records = self._records[-self.max_entries :]
        return record

    def get_recent(self) -> List[ErrorRecord]:
        """Return a copy of the captured error records."""

        return list(self._records)

    def clear(self) -> None:
        """Forget all captured errors."""

        self._records.clear()

