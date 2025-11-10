"""Command line entry-points for packaging utilities."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from .bundler import (
    build_pyinstaller_bundle,
    detect_default_data_paths,
    validate_bundle_artifact,
)
from .environment import (
    DEFAULT_ENV_PREFIXES,
    collect_environment_snapshot,
    write_environment_snapshot,
)


def _add_snapshot_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "snapshot", help="Export a JSON snapshot of the current Python environment."
    )
    parser.add_argument("output", type=Path, help="Path to write the JSON snapshot")
    parser.add_argument(
        "--config", action="append", type=Path, dest="config_paths", help="Configuration files or directories to include"
    )
    parser.add_argument(
        "--env-prefix",
        action="append",
        dest="env_prefixes",
        help="Environment variable prefixes to capture (can be supplied multiple times)",
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        dest="requirements_path",
        help="Optional requirements.txt path to capture",
    )


def _add_bundle_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("bundle", help="Build a desktop bundle using PyInstaller")
    parser.add_argument(
        "--app", type=Path, default=Path("app.py"), help="Entry point script (default: app.py)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("dist_bundle"), help="Directory for PyInstaller output"
    )
    parser.add_argument("--name", default="FlowTrainingUI", help="Bundle name")
    parser.add_argument(
        "--onefile", action="store_true", help="Produce a single-file executable (default: onedir)"
    )
    parser.add_argument("--console", action="store_true", help="Keep console window (default: hidden)")
    parser.add_argument(
        "--no-clean", action="store_true", help="Skip cleaning PyInstaller build directories"
    )
    parser.add_argument(
        "--icon",
        type=Path,
        help="Optional icon file to include in the bundle",
    )
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Skip writing an environment snapshot alongside the bundle",
    )
    parser.add_argument(
        "--add-data",
        action="append",
        type=str,
        dest="add_data",
        help=(
            "Additional data paths in the form 'source:dest'. Use the platform-specific separator 'source;dest' on Windows."
        ),
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        dest="extra_args",
        help="Extra arguments forwarded to PyInstaller",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the generated bundle artifact after building",
    )


def _add_validate_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("validate", help="Validate a previously created bundle")
    parser.add_argument("artifact", type=Path, help="Path to the bundle artifact")
    parser.add_argument(
        "--platform",
        action="append",
        dest="platforms",
        help="Expected platforms to validate against (linux, darwin, win32)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flow-packager", description="Packaging helpers for Flow Training UI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_snapshot_subparser(subparsers)
    _add_bundle_subparser(subparsers)
    _add_validate_subparser(subparsers)
    return parser


def run_snapshot_command(args: argparse.Namespace) -> int:
    prefixes = tuple(args.env_prefixes) if args.env_prefixes else DEFAULT_ENV_PREFIXES
    snapshot = collect_environment_snapshot(
        config_paths=args.config_paths,
        env_prefixes=prefixes,
        requirements_path=args.requirements_path,
    )
    write_environment_snapshot(snapshot, args.output)
    print(f"Environment snapshot written to {args.output}")
    return 0


def _parse_add_data(values: Sequence[str]) -> Sequence[tuple[Path, str]]:
    parsed = []
    for value in values:
        if os.pathsep in value:
            source, destination = value.split(os.pathsep, 1)
        elif ":" in value:
            source, destination = value.split(":", 1)
        elif ";" in value:
            source, destination = value.split(";", 1)
        else:
            raise ValueError(
                "Invalid --add-data value. Use 'source:dest' (or 'source;dest' on Windows)."
            )
        parsed.append((Path(source), destination))
    return parsed


def run_bundle_command(args: argparse.Namespace) -> int:
    add_data = detect_default_data_paths(args.app.resolve().parent)
    if args.add_data:
        add_data.extend(_parse_add_data(args.add_data))

    result = build_pyinstaller_bundle(
        args.app.resolve(),
        args.output_dir.resolve(),
        name=args.name,
        onefile=args.onefile,
        windowed=not args.console,
        clean=not args.no_clean,
        icon=args.icon,
        add_data=add_data,
        extra_args=args.extra_args,
        snapshot=not args.no_snapshot,
    )

    print(f"Bundle artifact created at {result.artifact_path}")

    if args.validate:
        validate_bundle_artifact(result.artifact_path)
        print("Bundle validation succeeded")

    return 0


def run_validate_command(args: argparse.Namespace) -> int:
    validate_bundle_artifact(args.artifact.resolve(), args.platforms)
    print("Bundle validation succeeded")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "snapshot":
        return run_snapshot_command(args)
    if args.command == "bundle":
        return run_bundle_command(args)
    if args.command == "validate":
        return run_validate_command(args)
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
