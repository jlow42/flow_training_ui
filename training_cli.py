"""Command-line interface for batch model training and evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

from exporters import ExportError, export_to_onnx, export_to_pmml, write_metadata_file
from run_registry import RunRegistry
from training_pipeline import (
    DEFAULT_RANDOM_STATE,
    TrainingResult,
    available_models,
    prepare_metadata,
    train_and_evaluate,
    validate_features,
)

LOGGER = logging.getLogger("flow-training-cli")


@dataclass
class ModelSpec:
    name: str
    params: Dict[str, object]


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _load_config(path: Optional[Path]) -> List[ModelSpec]:
    if not path:
        return []
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    models_raw = data.get("models") if isinstance(data, dict) else data
    if not isinstance(models_raw, list):
        raise ValueError("Configuration file must define a 'models' list.")
    specs: List[ModelSpec] = []
    for entry in models_raw:
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError("Each model config entry requires a 'name'.")
        params = entry.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError("Model 'params' must be a mapping of hyperparameters.")
        specs.append(ModelSpec(name=str(entry["name"]), params=params))
    return specs


def _parse_model_args(models: Sequence[str], config: Optional[Path]) -> List[ModelSpec]:
    specs = _load_config(config)
    for name in models:
        specs.append(ModelSpec(name=name, params={}))
    if not specs:
        specs.append(ModelSpec(name="random_forest", params={}))
    return specs


def _load_dataset(paths: Sequence[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["__source_file"] = path.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _normalise_features(data: pd.DataFrame, target: str, features: Optional[Sequence[str]]) -> List[str]:
    if features:
        return validate_features(data.columns, features)
    return [col for col in data.columns if col not in {target, "__source_file"}]


def _dump_bundle(
    result: TrainingResult,
    output_dir: Path,
    metadata: Dict[str, object],
) -> Path:
    bundle = {
        "model_name": result.model_name,
        "model": result.estimator,
        "features": list(result.features),
        "target": result.target,
        "metrics": result.metrics,
        "classes": result.classes,
        "confusion_matrix": result.confusion_matrix,
        "cv_scores": result.cv_scores,
        "cv_warning": result.cv_warning,
        "artifacts": result.artifacts,
        "config": result.config,
        "training_time": result.training_time,
        "timestamp": metadata.get("timestamp"),
    }
    bundle_path = output_dir / "model.joblib"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_path)
    return bundle_path


def _slugify(name: str) -> str:
    return "-".join(filter(None, "".join(ch if ch.isalnum() else " " for ch in name).split())).lower()


def _build_registry_record(
    result: TrainingResult,
    metadata: Dict[str, object],
    *,
    tags: Sequence[str],
    notes: str,
    class_balance: str,
) -> Dict[str, object]:
    metrics = result.metrics
    return {
        "id": uuid.uuid4().hex,
        "timestamp": metadata.get("timestamp"),
        "model_name": result.model_name,
        "target": result.target,
        "features": list(result.features),
        "accuracy": metrics.get("accuracy"),
        "f1_macro": metrics.get("f1_macro"),
        "model_params": result.config.get("model_params", {}),
        "class_balance": class_balance,
        "tags": list(tags),
        "notes": notes,
        "seed": result.config.get("random_state", DEFAULT_RANDOM_STATE),
        "source": "cli",
    }


def _append_registry(
    registry: RunRegistry,
    result: TrainingResult,
    metadata: Dict[str, object],
    *,
    tags: Sequence[str],
    notes: str,
    class_balance: str,
) -> None:
    record = _build_registry_record(
        result,
        metadata,
        tags=tags,
        notes=notes,
        class_balance=class_balance,
    )
    registry.append(record)


def run_train(args: argparse.Namespace) -> Dict[str, object]:
    paths = [Path(p) for p in args.inputs]
    data = _load_dataset(paths)
    features = _normalise_features(data, args.target, args.features)
    if not features:
        raise ValueError("No feature columns available after excluding the target column.")

    specs = _parse_model_args(args.model or [], args.config)
    export_formats = set(args.export or [])
    if not export_formats:
        export_formats = {"joblib", "onnx", "pmml"}

    registry = None if args.no_registry else RunRegistry(Path(args.registry_dir) if args.registry_dir else None)
    timestamp = datetime.now(timezone.utc).isoformat()

    summaries = []
    for spec in specs:
        LOGGER.info("Training model: %s", spec.name)
        result = train_and_evaluate(
            data,
            args.target,
            features,
            spec.name,
            params=spec.params,
            test_size=args.test_size,
            random_state=args.random_state,
            cv_folds=args.cv_folds,
            n_jobs=args.n_jobs,
        )
        metadata = prepare_metadata(result)
        metadata["timestamp"] = timestamp
        metadata["notes"] = args.notes
        metadata["tags"] = args.tags

        model_slug = _slugify(result.model_name or spec.name)
        output_dir = Path(args.output_dir) / model_slug
        output_dir.mkdir(parents=True, exist_ok=True)

        bundle_path = _dump_bundle(result, output_dir, metadata)
        metadata_path = write_metadata_file(output_dir / "metadata.json", metadata)

        onnx_path = None
        if "onnx" in export_formats:
            try:
                onnx_path = export_to_onnx(result.estimator, features, output_dir / "model.onnx", metadata)
            except ExportError as exc:
                if not args.ignore_export_errors:
                    raise
                LOGGER.warning("ONNX export skipped: %s", exc)

        pmml_path = None
        if "pmml" in export_formats:
            try:
                pmml_path = export_to_pmml(result.estimator, features, args.target, output_dir / "model.pmml", metadata)
            except ExportError as exc:
                if not args.ignore_export_errors:
                    raise
                LOGGER.warning("PMML export skipped: %s", exc)

        if "joblib" not in export_formats:
            bundle_path.unlink(missing_ok=True)

        if registry is not None:
            _append_registry(
                registry,
                result,
                metadata,
                tags=args.tags,
                notes=args.notes,
                class_balance=args.class_balance,
            )

        summary = {
            "model": result.model_name,
            "metrics": metadata["metrics"],
            "output_dir": str(output_dir),
            "bundle_path": str(bundle_path),
            "onnx_path": str(onnx_path) if onnx_path else None,
            "pmml_path": str(pmml_path) if pmml_path else None,
            "metadata_path": str(metadata_path),
        }
        summaries.append(summary)

    return {"runs": summaries, "target": args.target, "features": features}


def run_evaluate(args: argparse.Namespace) -> Dict[str, object]:
    bundle = joblib.load(Path(args.model_path))
    model = bundle["model"]
    target = args.target or bundle.get("target")
    if not target:
        raise ValueError("Target column must be provided either via the CLI or the bundle metadata.")

    paths = [Path(p) for p in args.inputs]
    data = _load_dataset(paths)
    features = args.features or bundle.get("features")
    if not features:
        features = [col for col in data.columns if col not in {target, "__source_file"}]
    features = validate_features(data.columns, features)

    from training_pipeline import _classification_metrics  # Local import to avoid circular

    X = data.loc[:, features]
    y_true = data.loc[:, target]
    predictions = model.predict(X)
    metrics, matrix, classes = _classification_metrics(y_true, predictions)
    return {
        "model": bundle.get("model_name"),
        "metrics": {
            "accuracy": metrics.get("accuracy"),
            "f1_macro": metrics.get("f1_macro"),
            "f1_weighted": metrics.get("f1_weighted"),
        },
        "confusion_matrix": matrix.tolist(),
        "classes": classes,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flow Training CLI")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train one or more models.")
    train_parser.add_argument("--inputs", nargs="+", required=True, help="CSV files to load.")
    train_parser.add_argument("--target", required=True, help="Target column name.")
    train_parser.add_argument("--features", nargs="*", help="Explicit feature column list.")
    train_parser.add_argument(
        "--model",
        action="append",
        help="Model identifier (can be repeated). Defaults to random_forest.",
    )
    train_parser.add_argument("--config", type=Path, help="JSON config describing models to train.")
    train_parser.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    train_parser.add_argument("--cv-folds", type=int, default=0, help="Number of CV folds.")
    train_parser.add_argument("--n-jobs", type=int, default=1, dest="n_jobs", help="Parallel jobs for estimators.")
    train_parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    train_parser.add_argument("--output-dir", type=Path, default=Path("cli_runs"))
    train_parser.add_argument(
        "--export",
        action="append",
        choices=["joblib", "onnx", "pmml"],
        help="Export formats to produce (default: all).",
    )
    train_parser.add_argument("--ignore-export-errors", action="store_true", help="Continue even if an export fails.")
    train_parser.add_argument("--no-registry", action="store_true", help="Skip registry updates.")
    train_parser.add_argument("--registry-dir", type=Path, help="Override registry directory (for testing).")
    train_parser.add_argument("--tags", nargs="*", default=[], help="Optional tags stored alongside the run.")
    train_parser.add_argument("--notes", default="", help="Free-form notes stored with the run.")
    train_parser.add_argument("--class-balance", default="none", help="Description of the class balancing strategy.")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate an exported bundle on new data.")
    eval_parser.add_argument("--inputs", nargs="+", required=True, help="CSV files to load.")
    eval_parser.add_argument("--model-path", required=True, help="Path to a joblib bundle.")
    eval_parser.add_argument("--target", help="Target column name (defaults to bundle metadata).")
    eval_parser.add_argument("--features", nargs="*", help="Optional feature override.")

    subparsers.add_parser("list-models", help="List supported model identifiers.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    try:
        if args.command == "train":
            result = run_train(args)
        elif args.command == "evaluate":
            result = run_evaluate(args)
        elif args.command == "list-models":
            result = {"models": available_models()}
        else:  # pragma: no cover - defensive fallback
            parser.error("Unknown command")
            return 2
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("%s", exc)
        if args.verbose:
            raise
        return 1

    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    sys.exit(main())

