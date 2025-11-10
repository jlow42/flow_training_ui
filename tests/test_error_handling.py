from pathlib import Path

from error_handling import ErrorManager


def test_error_manager_records_guidance(tmp_path: Path) -> None:
    log_path = tmp_path / "app.log"
    log_path.write_text("log")

    manager = ErrorManager()
    manager.set_log_path(log_path)
    record = manager.register_error("CSV load", "Failed to load CSV file")

    assert "csv" in record.guidance.lower()
    assert str(log_path) in record.guidance
    assert "Guidance" in record.formatted_message


def test_error_history_clear(tmp_path: Path) -> None:
    manager = ErrorManager(max_entries=2)
    manager.register_error("A", "First")
    manager.register_error("B", "Second")
    assert len(manager.get_recent()) == 2

    manager.clear()
    assert manager.get_recent() == []


def test_error_history_trims(tmp_path: Path) -> None:
    manager = ErrorManager(max_entries=2)
    manager.register_error("One", "First")
    manager.register_error("Two", "Second")
    manager.register_error("Three", "Third")

    recent = manager.get_recent()
    assert len(recent) == 2
    assert recent[0].title == "Two"
    assert recent[1].title == "Three"
