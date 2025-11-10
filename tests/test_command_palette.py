"""Unit tests for the command palette helpers."""

from app import CommandDefinition, filter_commands


def _noop() -> None:
    """Simple callback used for command definitions in tests."""
    return None


def test_filter_commands_returns_all_for_blank_query() -> None:
    commands = [
        CommandDefinition(
            label="Load CSV files",
            callback=_noop,
            description="Open a file chooser",
            category="Data",
            accelerator="Ctrl+O",
        ),
        CommandDefinition(
            label="Go to Training setup",
            callback=_noop,
            description="Navigate to the training configuration tab",
            category="Navigation",
            accelerator="Ctrl+Shift+S",
        ),
    ]

    result = filter_commands(commands, "   ")

    assert result == commands


def test_filter_commands_matches_multiple_tokens() -> None:
    commands = [
        CommandDefinition(
            label="Load CSV files",
            callback=_noop,
            description="Open a file chooser",
            category="Data",
            accelerator="Ctrl+O",
        ),
        CommandDefinition(
            label="Go to Training setup",
            callback=_noop,
            description="Navigate to the training configuration tab",
            category="Navigation",
            accelerator="Ctrl+Shift+S",
        ),
        CommandDefinition(
            label="Go to Training results",
            callback=_noop,
            description="Review evaluation metrics",
            category="Navigation",
            accelerator="Ctrl+Shift+R",
        ),
    ]

    result = filter_commands(commands, "training results")

    assert [command.label for command in result] == ["Go to Training results"]


def test_filter_commands_searches_accelerator_and_description() -> None:
    commands = [
        CommandDefinition(
            label="Show keyboard shortcuts",
            callback=_noop,
            description="Display quick reference",
            category="Help",
            accelerator="F1",
        ),
        CommandDefinition(
            label="Open command palette",
            callback=_noop,
            description="Search available commands",
            category="Navigation",
            accelerator="Ctrl+Shift+P",
        ),
    ]

    result = filter_commands(commands, "shift p")

    assert [command.label for command in result] == ["Open command palette"]
