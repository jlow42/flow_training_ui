"""Environment snapshot utilities for Flow Training UI.

This module collects metadata about the current Python runtime, installed
packages, and important configuration files. The resulting snapshot is meant to
make it easier to reproduce the environment that generated a training bundle or
model export.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional, Sequence

TEXT_SAMPLE_BYTES = 4096
DEFAULT_ENV_PREFIXES: tuple[str, ...] = (
    "FLOW_",
    "PYTHON",
    "CONDA",
    "VIRTUAL_ENV",
)


@dataclass
class FileMetadata:
    """Small metadata payload for a configuration file."""

    path: str
    size_bytes: int
    sha256: str
    modified_utc: str
    text_preview: Optional[str] = None

    @classmethod
    def from_path(cls, path: Path, project_root: Path) -> "FileMetadata":
        stat = path.stat()
        hash_obj = sha256()
        data = path.read_bytes()
        hash_obj.update(data)
        preview: Optional[str]
        try:
            preview_text = data[:TEXT_SAMPLE_BYTES].decode("utf-8")
            preview = preview_text if preview_text.strip() else None
        except UnicodeDecodeError:
            preview = None
        try:
            rel_path = path.relative_to(project_root)
        except ValueError:
            rel_path = path
        return cls(
            path=str(rel_path),
            size_bytes=stat.st_size,
            sha256=hash_obj.hexdigest(),
            modified_utc=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            text_preview=preview,
        )


@dataclass
class EnvironmentSnapshot:
    """Serializable representation of the runtime environment."""

    python_version: str
    executable: str
    platform: str
    timestamp_utc: str
    working_directory: str
    dependencies: List[str] = field(default_factory=list)
    requirements_file: Optional[str] = None
    environment_variables: Dict[str, str] = field(default_factory=dict)
    config_files: List[FileMetadata] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        config_payload = [metadata.__dict__ for metadata in self.config_files]
        return {
            "python_version": self.python_version,
            "executable": self.executable,
            "platform": self.platform,
            "timestamp_utc": self.timestamp_utc,
            "working_directory": self.working_directory,
            "dependencies": self.dependencies,
            "requirements_file": self.requirements_file,
            "environment_variables": self.environment_variables,
            "config_files": config_payload,
        }


def _default_config_candidates(project_root: Path) -> List[Path]:
    candidates: List[Path] = []
    for pattern in ("*.json", "*.yaml", "*.yml", "*.toml", "*.ini"):
        candidates.extend(project_root.glob(pattern))
    return candidates


def _resolve_config_paths(
    project_root: Path, config_paths: Optional[Sequence[Path]]
) -> List[Path]:
    if config_paths:
        resolved = [project_root / path if not path.is_absolute() else path for path in config_paths]
    else:
        resolved = _default_config_candidates(project_root)
    normalized: List[Path] = []
    for path in resolved:
        if path.exists():
            if path.is_dir():
                for child in path.rglob("*"):
                    if child.is_file():
                        normalized.append(child)
            else:
                normalized.append(path)
    unique_paths: List[Path] = []
    seen: set[Path] = set()
    for path in normalized:
        resolved_path = path.resolve()
        if resolved_path not in seen:
            seen.add(resolved_path)
            unique_paths.append(resolved_path)
    return unique_paths


def _collect_environment_variables(prefixes: Sequence[str]) -> Dict[str, str]:
    captured: Dict[str, str] = {}
    for key, value in os.environ.items():
        if any(key.startswith(prefix) for prefix in prefixes):
            captured[key] = value
    return dict(sorted(captured.items()))


def _read_requirements(project_root: Path, requirements: Optional[Path]) -> Optional[str]:
    target = requirements or project_root / "requirements.txt"
    if target.exists():
        return target.read_text(encoding="utf-8")
    return None


def collect_environment_snapshot(
    config_paths: Optional[Sequence[Path]] = None,
    env_prefixes: Sequence[str] = DEFAULT_ENV_PREFIXES,
    requirements_path: Optional[Path] = None,
    cwd: Optional[Path] = None,
) -> EnvironmentSnapshot:
    """Collect environment metadata and return an :class:`EnvironmentSnapshot`.

    Parameters
    ----------
    config_paths:
        Optional sequence of file or directory paths to include in the snapshot.
        If omitted, a heuristic search captures project-level JSON/YAML/TOML/INI
        files.
    env_prefixes:
        Prefixes of environment variables to capture. Defaults to a short list of
        virtual-environment related prefixes.
    requirements_path:
        Optional path to a requirements file. Defaults to ``requirements.txt`` in
        the project root if available.
    cwd:
        Working directory to treat as the project root. Defaults to the current
        working directory.
    """

    project_root = cwd or Path.cwd()
    timestamp = datetime.now(timezone.utc).isoformat()
    dependencies: List[str] = []
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        )
        dependencies = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.CalledProcessError):
        dependencies = []

    config_metadata: List[FileMetadata] = []
    for path in _resolve_config_paths(project_root, config_paths):
        try:
            config_metadata.append(FileMetadata.from_path(path, project_root))
        except (OSError, PermissionError):
            continue

    snapshot = EnvironmentSnapshot(
        python_version=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        timestamp_utc=timestamp,
        working_directory=str(project_root),
        dependencies=dependencies,
        requirements_file=_read_requirements(project_root, requirements_path),
        environment_variables=_collect_environment_variables(env_prefixes),
        config_files=config_metadata,
    )
    return snapshot


def write_environment_snapshot(snapshot: EnvironmentSnapshot, output_path: Path) -> Path:
    """Serialize *snapshot* to *output_path* in JSON format."""

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return output_path


__all__ = [
    "EnvironmentSnapshot",
    "collect_environment_snapshot",
    "write_environment_snapshot",
]
