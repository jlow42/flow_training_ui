"""Local telemetry pipeline with opt-in controls and privacy safeguards."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class TelemetryError(RuntimeError):
    """Raised when telemetry preferences cannot be persisted."""


@dataclass
class TelemetryEvent:
    """Representation of a telemetry event stored on disk."""

    name: str
    timestamp: float
    session_id: str
    client_id: str
    properties: Dict[str, object]

    def to_json(self) -> str:
        """Serialize the event to a JSON line."""

        payload = asdict(self)
        payload["timestamp_iso"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp)
        )
        return json.dumps(payload, ensure_ascii=False)


class TelemetryManager:
    """Persist telemetry preferences and events locally."""

    _CONFIG_FILE = "telemetry_settings.json"
    _EVENT_FILE = "telemetry_events.jsonl"

    def __init__(self, storage_dir: Path, logger: Optional[logging.Logger] = None) -> None:
        self.storage_dir = storage_dir
        self.logger = logger or logging.getLogger(__name__)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.config_path = self.storage_dir / self._CONFIG_FILE
        self.events_path = self.storage_dir / self._EVENT_FILE

        self._config = self._load_config()
        self.client_id = self._config.get("client_id") or uuid.uuid4().hex
        self.opted_in = bool(self._config.get("opted_in", False))
        self.session_id = uuid.uuid4().hex

        if self._config.get("client_id") != self.client_id:
            self._config["client_id"] = self.client_id
            self._persist_config()

    # ------------------------------------------------------------------
    # Preference management
    # ------------------------------------------------------------------
    def set_opt_in(self, opted_in: bool) -> None:
        """Persist telemetry opt-in preference and enforce privacy safeguards."""

        if self.opted_in == opted_in:
            return

        self.opted_in = opted_in
        self._config["opted_in"] = opted_in
        try:
            self._persist_config()
        except OSError as exc:  # pragma: no cover - exercised via TelemetryError
            self.opted_in = not opted_in
            raise TelemetryError(f"Failed to update telemetry preferences: {exc}") from exc

        if not opted_in:
            self._purge_events()
            self.logger.info("Telemetry disabled by user preference.")
        else:
            self.logger.info("Telemetry enabled by user preference.")
            self.record_event("telemetry_enabled", {"from_state": "disabled"})

    # ------------------------------------------------------------------
    # Event collection
    # ------------------------------------------------------------------
    def record_event(
        self,
        name: str,
        properties: Optional[Dict[str, object]] = None,
        *,
        sensitive_keys: Optional[Iterable[str]] = None,
    ) -> bool:
        """Record a telemetry event if the user has opted in."""

        if not self.opted_in:
            self.logger.debug("Telemetry event '%s' dropped because user is opted out.", name)
            return False

        safe_properties = self._sanitize_properties(properties or {}, sensitive_keys)

        event = TelemetryEvent(
            name=name,
            timestamp=time.time(),
            session_id=self.session_id,
            client_id=self.client_id,
            properties=safe_properties,
        )

        try:
            with self.events_path.open("a", encoding="utf-8") as fh:
                fh.write(event.to_json())
                fh.write("\n")
        except OSError as exc:  # pragma: no cover - depends on OS failures
            self.logger.warning("Failed to persist telemetry event '%s': %s", name, exc)
            return False

        self.logger.debug("Telemetry event '%s' captured.", name)
        return True

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, object]]:
        """Return the most recent telemetry events for inspection."""

        if not self.events_path.exists():
            return []

        events: List[Dict[str, object]] = []
        with self.events_path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()

        for line in lines[-limit:]:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        return events

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_config(self) -> Dict[str, object]:
        if not self.config_path.exists():
            return {}

        try:
            with self.config_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return {}

    def _persist_config(self) -> None:
        temp_path = self.config_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as fh:
            json.dump(self._config, fh, indent=2, sort_keys=True)
        temp_path.replace(self.config_path)

    def _purge_events(self) -> None:
        if self.events_path.exists():
            try:
                self.events_path.unlink()
            except OSError as exc:  # pragma: no cover - depends on OS failures
                self.logger.warning("Failed to purge telemetry events: %s", exc)

    def _sanitize_properties(
        self, properties: Dict[str, object], sensitive_keys: Optional[Iterable[str]]
    ) -> Dict[str, object]:
        sensitive = {key for key in sensitive_keys or []}
        safe_properties: Dict[str, object] = {}
        for key, value in properties.items():
            if key in sensitive or value is None:
                continue
            if isinstance(value, (bool, int, float)):
                safe_properties[key] = value
            elif isinstance(value, str):
                safe_properties[key] = value[:200]
            else:
                safe_properties[key] = str(value)[:200]
        return safe_properties

