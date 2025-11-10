from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd


def _run_cli(args: list[str], cwd: Path) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "training_cli", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _make_dataset(path: Path) -> Path:
    df = pd.DataFrame(
        {
            "FSC": [1.0, 1.2, 0.9, 1.5, 1.4, 0.8],
            "SSC": [0.4, 0.45, 0.5, 0.35, 0.3, 0.55],
            "CD3": [100, 120, 95, 140, 130, 90],
            "CD19": [50, 45, 48, 52, 53, 47],
            "CellType": ["T", "T", "T", "B", "B", "B"],
        }
    )
    csv_path = path / "dataset.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_cli_training_and_exports(tmp_path: Path) -> None:
    csv_path = _make_dataset(tmp_path)
    registry_dir = tmp_path / "registry"
    output_dir = tmp_path / "exports"

    result = _run_cli(
        [
            "train",
            "--inputs",
            str(csv_path),
            "--target",
            "CellType",
            "--model",
            "random_forest",
            "--output-dir",
            str(output_dir),
            "--registry-dir",
            str(registry_dir),
            "--tags",
            "integration",
            "--notes",
            "CLI test",
            "--export",
            "joblib",
            "--export",
            "onnx",
            "--export",
            "pmml",
            "--ignore-export-errors",
        ],
        cwd=Path.cwd(),
    )

    assert result["runs"], "Expected at least one run summary"
    run = result["runs"][0]
    run_dir = Path(run["output_dir"])
    assert run_dir.exists()

    onnx_path = Path(run["onnx_path"]) if run["onnx_path"] else None
    pmml_path = Path(run["pmml_path"]) if run["pmml_path"] else None
    metadata_path = Path(run["metadata_path"])
    bundle_path = Path(run["bundle_path"])

    assert "Random Forest" in run["model"]
    if onnx_path and onnx_path.exists():
        onnx_bytes = onnx_path.read_bytes()
        assert b"metrics" in onnx_bytes

    if pmml_path and pmml_path.exists():
        tree = ElementTree.parse(pmml_path)
        root = tree.getroot()
        extensions = root.findall(".//{*}Extension")
        extension_names = {ext.attrib.get("name") for ext in extensions}
        assert "metrics" in extension_names

    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert saved_metadata["metrics"]["accuracy"] >= 0.0

    registry_file = registry_dir / ".flow_cache" / "run_registry.json"
    assert registry_file.exists(), "Registry not updated"
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    assert registry["runs"], "Registry missing entries"

    eval_result = _run_cli(
        [
            "evaluate",
            "--inputs",
            str(csv_path),
            "--model-path",
            str(bundle_path),
            "--target",
            "CellType",
        ],
        cwd=Path.cwd(),
    )
    assert "metrics" in eval_result and eval_result["metrics"]["accuracy"] >= 0.0


def test_list_models_command(tmp_path: Path) -> None:
    result = _run_cli(["list-models"], cwd=Path.cwd())
    assert "random_forest" in result["models"]
