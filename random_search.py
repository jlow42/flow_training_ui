"""Utilities for running Optuna-backed randomized hyperparameter searches."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:  # pragma: no cover - optuna is an optional dependency at runtime
    import optuna  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully by the app
    optuna = None  # type: ignore[assignment]


class RandomizedSearchUnavailableError(RuntimeError):
    """Raised when randomized search features are unavailable."""


class SearchSpaceValidationError(ValueError):
    """Raised when the configured search space is invalid."""


@dataclass
class TrialInfo:
    """Lightweight representation of a completed trial."""

    number: int
    value: float
    params: Dict[str, Any]


class OptunaRandomSearchRunner:
    """Helper that encapsulates Optuna study management and execution."""

    def __init__(self, storage_dir: Path) -> None:
        if optuna is None:  # pragma: no cover - exercised via integration
            raise RandomizedSearchUnavailableError(
                "Optuna is not installed; randomized search is unavailable."
            )
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def clamp_jobs(requested: int, cpu_budget: Optional[int], total_cpu: int) -> int:
        """Clamp parallel job count against CPU availability and budgets."""

        if total_cpu < 1:
            total_cpu = 1
        requested = max(int(requested), 1)
        maximum = min(requested, total_cpu)
        if cpu_budget is not None:
            maximum = min(maximum, max(int(cpu_budget), 1))
        return maximum

    def run(
        self,
        *,
        study_name: str,
        base_params: Dict[str, Any],
        search_space: Dict[str, Dict[str, Any]],
        evaluate: Callable[[Dict[str, Any]], Tuple[float, Dict[str, Any]]],
        n_trials: int,
        timeout: Optional[float] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Execute a randomized search and return the best payload."""

        if n_trials < 1:
            raise SearchSpaceValidationError("n_trials must be at least 1.")
        if not search_space:
            raise SearchSpaceValidationError("Search space must contain parameters.")

        db_path = self.storage_dir / f"{study_name}.db"
        storage = f"sqlite:///{db_path}"
        study = optuna.create_study(  # type: ignore[union-attr]
            study_name=study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )

        best_payload: Optional[Dict[str, Any]] = None
        best_metric = float("-inf")
        completed_trials = 0
        start_time = time.time()

        def objective(trial: "optuna.trial.Trial") -> float:  # type: ignore[name-defined]
            nonlocal best_payload, best_metric, completed_trials
            trial_params = self._suggest_params(trial, search_space)
            params = dict(base_params)
            params.update(trial_params)
            metric, payload = evaluate(params)
            completed_trials += 1
            if metric > best_metric:
                best_metric = metric
                best_payload = payload
            if progress_callback is not None:
                progress_callback(
                    {
                        "trial": trial.number,
                        "metric": metric,
                        "best_metric": best_metric,
                        "params": params,
                        "elapsed": time.time() - start_time,
                        "trials_completed": completed_trials,
                        "total_trials": n_trials,
                    }
                )
            return metric

        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        if best_payload is None:
            raise RuntimeError("Randomized search completed without any successful trials.")

        trials: List[TrialInfo] = [
            TrialInfo(number=t.number, value=t.value or float("nan"), params=dict(t.params))
            for t in study.trials
            if t.value is not None
        ]
        best_payload.setdefault("search_summary", {})
        best_payload["search_summary"].update(
            {
                "study_name": study_name,
                "trials_completed": completed_trials,
                "best_score": best_metric,
                "best_trial": study.best_trial.number if study.best_trial else None,
            }
        )
        best_payload["search_trials"] = [
            {"number": t.number, "value": t.value, "params": t.params} for t in trials
        ]
        return best_payload

    def _suggest_params(
        self, trial: "optuna.trial.Trial", space: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        suggested: Dict[str, Any] = {}
        for name, definition in space.items():
            param_type = definition.get("type")
            if param_type == "int":
                low = int(definition["low"])
                high = int(definition["high"])
                if low > high:
                    raise SearchSpaceValidationError(
                        f"Invalid range for '{name}': {low}..{high}."
                    )
                suggested[name] = trial.suggest_int(name, low, high)
            elif param_type == "int_or_none":
                low = int(definition["low"])
                high = int(definition["high"])
                if low > high:
                    raise SearchSpaceValidationError(
                        f"Invalid range for '{name}': {low}..{high}."
                    )
                if definition.get("allow_none", False):
                    choices: List[Any] = list(range(low, high + 1))
                    choices.append(None)
                    suggested[name] = trial.suggest_categorical(name, choices)
                else:
                    suggested[name] = trial.suggest_int(name, low, high)
            elif param_type == "float":
                low = float(definition["low"])
                high = float(definition["high"])
                if low > high:
                    raise SearchSpaceValidationError(
                        f"Invalid range for '{name}': {low}..{high}."
                    )
                suggested[name] = trial.suggest_float(
                    name,
                    low,
                    high,
                    log=bool(definition.get("log", False)),
                )
            elif param_type == "categorical":
                choices = definition.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise SearchSpaceValidationError(
                        f"Categorical parameter '{name}' requires at least one choice."
                    )
                suggested[name] = trial.suggest_categorical(name, choices)
            else:
                raise SearchSpaceValidationError(
                    f"Unsupported search parameter type '{param_type}' for '{name}'."
                )
        return suggested

