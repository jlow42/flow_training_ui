import stat
import tempfile
import unittest
from pathlib import Path

from packaging_tools import (
    collect_environment_snapshot,
    validate_bundle_artifact,
    write_environment_snapshot,
)
from packaging_tools.environment import EnvironmentSnapshot


class EnvironmentSnapshotTests(unittest.TestCase):
    def test_collect_snapshot_produces_metadata(self) -> None:
        snapshot = collect_environment_snapshot(config_paths=[], env_prefixes=("PYTHON",))
        self.assertIsInstance(snapshot, EnvironmentSnapshot)
        self.assertTrue(snapshot.python_version)
        self.assertTrue(snapshot.dependencies)

    def test_snapshot_round_trip(self) -> None:
        snapshot = collect_environment_snapshot(config_paths=[], env_prefixes=("PYTHON",))
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "snapshot.json"
            write_environment_snapshot(snapshot, output_path)
            self.assertTrue(output_path.exists())
            content = output_path.read_text(encoding="utf-8")
            self.assertIn("python_version", content)
            self.assertIn("dependencies", content)


class BundleValidationTests(unittest.TestCase):
    def test_linux_validation_requires_executable_bit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "FlowTrainingUI"
            artifact.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            # Initially missing executable permission should fail
            with self.assertRaises(PermissionError):
                validate_bundle_artifact(artifact, ["linux"])
            artifact.chmod(artifact.stat().st_mode | stat.S_IXUSR)
            # Should pass once executable bit is set
            validate_bundle_artifact(artifact, ["linux"])

    def test_windows_validation_requires_exe_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "FlowTrainingUI.exe"
            artifact.write_text("binary", encoding="utf-8")
            validate_bundle_artifact(artifact, ["win32"])

    def test_macos_validation_accepts_app_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app_dir = Path(tmpdir) / "FlowTrainingUI.app"
            contents = app_dir / "Contents" / "MacOS"
            contents.mkdir(parents=True)
            validate_bundle_artifact(app_dir, ["darwin"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
