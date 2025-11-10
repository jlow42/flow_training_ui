"""Model export helpers for ONNX and PMML formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping
from xml.etree import ElementTree

import numpy as np


class ExportError(RuntimeError):
    """Raised when an export fails (missing dependency or conversion error)."""


def _ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _serialise_metadata(metadata: Mapping[str, Any]) -> Dict[str, str]:
    serialised: Dict[str, str] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float)) or value is None:
            serialised[key] = "" if value is None else str(value)
        else:
            serialised[key] = json.dumps(value, default=_json_default)
    return serialised


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serialisable")


def export_to_onnx(
    model: object,
    feature_names: Iterable[str],
    output_path: Path,
    metadata: Mapping[str, Any],
) -> Path:
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError as exc:  # pragma: no cover - exercised in environments without skl2onnx
        raise ExportError("Install 'skl2onnx' to enable ONNX export.") from exc

    feature_list = list(feature_names)
    initial_type = [("input", FloatTensorType([None, len(feature_list)]))]
    try:
        onnx_model = convert_sklearn(model, initial_types=initial_type)
    except Exception as exc:  # noqa: BLE001
        raise ExportError(f"Failed to convert model to ONNX: {exc}") from exc

    serialised_metadata = _serialise_metadata(metadata)
    meta_props = onnx_model.metadata_props.add
    for key, value in serialised_metadata.items():
        meta_props(key, value)

    _ensure_directory(output_path)
    with output_path.open("wb") as handle:
        handle.write(onnx_model.SerializeToString())
    return output_path


def export_to_pmml(
    model: object,
    feature_names: Iterable[str],
    target: str,
    output_path: Path,
    metadata: Mapping[str, Any],
) -> Path:
    try:
        from nyoka import skl_to_pmml
    except ImportError as exc:  # pragma: no cover - exercised in environments without nyoka
        raise ExportError("Install 'nyoka' to enable PMML export.") from exc

    feature_list = list(feature_names)
    _ensure_directory(output_path)
    try:
        skl_to_pmml(model, feature_list, target, str(output_path))
    except Exception as exc:  # noqa: BLE001
        raise ExportError(f"Failed to convert model to PMML: {exc}") from exc

    _inject_pmml_metadata(output_path, metadata)
    return output_path


def _inject_pmml_metadata(output_path: Path, metadata: Mapping[str, Any]) -> None:
    tree = ElementTree.parse(output_path)
    root = tree.getroot()
    if root.tag.startswith("{") and "}" in root.tag:
        namespace = root.tag.split("}", 1)[0][1:]
        prefix = f"{{{namespace}}}"
    else:
        namespace = ""
        prefix = ""

    header = root.find(f"{prefix}Header")
    if header is None:
        header = ElementTree.SubElement(root, f"{prefix}Header")

    serialised_metadata = _serialise_metadata(metadata)
    for key, value in serialised_metadata.items():
        extension = ElementTree.SubElement(header, f"{prefix}Extension")
        extension.set("name", key)
        extension.text = value

    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def write_metadata_file(output_path: Path, metadata: Mapping[str, Any]) -> Path:
    _ensure_directory(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=_json_default)
    return output_path

