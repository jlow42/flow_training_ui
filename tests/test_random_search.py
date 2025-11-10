from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest

pytest.importorskip("optuna")

from random_search import OptunaRandomSearchRunner


def test_clamp_jobs_enforces_cpu_budget(tmp_path: Path) -> None:
    runner = OptunaRandomSearchRunner(tmp_path)
    assert runner.clamp_jobs(8, 2, 4) == 2
    assert runner.clamp_jobs(0, None, 4) == 1
    assert runner.clamp_jobs(4, 10, 3) == 3


def test_run_returns_best_payload(tmp_path: Path) -> None:
    runner = OptunaRandomSearchRunner(tmp_path)
    search_space = {"x": {"type": "int", "low": 0, "high": 4}}
    history = []
    progress_events: list[Dict[str, object]] = []

    def evaluate(params: Dict[str, object]):
        x = int(params["x"])  # type: ignore[index]
        metric = 1.0 - abs(x - 3) * 0.1
        payload = {
            "metrics": {"f1_macro": metric, "accuracy": metric},
            "confusion_matrix": [],
            "classes": [],
            "feature_importances": [],
            "train_rows": 0,
            "test_rows": 0,
            "cv_scores": None,
            "cv_warning": "",
            "artifacts": {},
            "model": object(),
            "model_name": "dummy",
            "features": ["f"],
            "target": "y",
            "config": {
                "model_params": dict(params),
                "test_size": 0.2,
                "cv_folds": 3,
                "n_jobs": 1,
            },
            "training_time": 0.0,
        }
        history.append(metric)
        return metric, payload

    best_payload = runner.run(
        study_name="demo",
        base_params={},
        search_space=search_space,
        evaluate=evaluate,
        n_trials=4,
        timeout=10.0,
        progress_callback=progress_events.append,
    )

    assert best_payload["metrics"]["f1_macro"] == pytest.approx(max(history))
    summary = best_payload.get("search_summary", {})
    assert summary.get("best_score") == pytest.approx(max(history))
    assert progress_events, "progress callback should be invoked"
    assert progress_events[-1]["best_metric"] == pytest.approx(max(history))
    assert len(best_payload.get("search_trials", [])) >= len(progress_events)
