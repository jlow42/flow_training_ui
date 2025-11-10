from pathlib import Path

from telemetry import TelemetryManager


def test_default_opt_out(tmp_path: Path) -> None:
    manager = TelemetryManager(tmp_path)

    assert manager.opted_in is False
    manager.record_event("test", {"value": 1})
    assert manager.get_recent_events() == []


def test_opt_in_records_sanitized_event(tmp_path: Path) -> None:
    manager = TelemetryManager(tmp_path)
    manager.set_opt_in(True)

    manager.record_event(
        "files_loaded",
        {"requested_files": 3, "path": "/tmp/secret", "note": "ok"},
        sensitive_keys={"path"},
    )

    events = manager.get_recent_events()
    assert events
    event = events[-1]
    assert event["name"] == "files_loaded"
    assert event["properties"]["requested_files"] == 3
    assert "path" not in event["properties"]


def test_opt_out_purges_events(tmp_path: Path) -> None:
    manager = TelemetryManager(tmp_path)
    manager.set_opt_in(True)
    manager.record_event("sample", {})
    assert manager.get_recent_events()

    manager.set_opt_in(False)
    assert manager.get_recent_events() == []


def test_preference_persists_between_sessions(tmp_path: Path) -> None:
    manager = TelemetryManager(tmp_path)
    manager.set_opt_in(True)

    second = TelemetryManager(tmp_path)
    assert second.opted_in is True
