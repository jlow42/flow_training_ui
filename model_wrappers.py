"""Training helpers and model wrappers for supervised classifiers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # pragma: no cover - optional dependency
    from imblearn.over_sampling import SMOTE  # type: ignore[import]
except ImportError:  # pragma: no cover - runtime fallback
    SMOTE = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from xgboost import XGBClassifier  # type: ignore[import]
except ImportError:  # pragma: no cover
    XGBClassifier = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import lightgbm as lgb  # type: ignore[import]
except ImportError:  # pragma: no cover
    lgb = None  # type: ignore[assignment]


@dataclass
class BalancePayload:
    """Container describing class balancing adjustments."""

    X_train: pd.DataFrame | np.ndarray
    y_train: pd.Series | np.ndarray
    strategy: str = "None"
    class_weights: Optional[Dict[object, float]] = None
    sample_weight: Optional[np.ndarray] = None


@dataclass
class TrainResult:
    """Return value for wrapper training routines."""

    model: object
    predictions: np.ndarray
    probabilities: Optional[np.ndarray]
    cv_estimator: object


def compute_class_weight_dict(y: pd.Series) -> Dict[object, float]:
    """Return inverse-frequency class weights for a target series."""

    counts = y.value_counts(dropna=False)
    total = len(y)
    n_classes = len(counts)
    weights: Dict[object, float] = {}
    for cls, count in counts.items():
        if count == 0:
            continue
        weights[cls] = total / (n_classes * count)
    return weights


def class_weight_sample_array(y: pd.Series, class_weights: Dict[object, float]) -> np.ndarray:
    """Map class weights onto an array aligned with ``y``."""

    mapped = y.map(class_weights).astype(float)
    return mapped.to_numpy(dtype=float, copy=False)


def focal_sample_weight_array(
    y: pd.Series, class_weights: Dict[object, float], gamma: float = 2.0
) -> np.ndarray:
    """Return per-row weights following the focal loss heuristic."""

    counts = y.value_counts(dropna=False)
    total = len(y)
    frequencies = counts / float(total)
    weights = {}
    for cls, base_weight in class_weights.items():
        freq = float(frequencies.get(cls, 0.0))
        weights[cls] = base_weight * float((1.0 - freq) ** gamma)
    return class_weight_sample_array(y, weights)


def apply_smote(
    X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, random_state: int
) -> tuple[pd.DataFrame | np.ndarray, pd.Series | np.ndarray]:
    """Apply SMOTE oversampling to rebalance ``X`` and ``y``."""

    if SMOTE is None:  # pragma: no cover - handled via requirements
        raise ValueError("Install 'imbalanced-learn' to use SMOTE balancing.")
    smote = SMOTE(random_state=random_state)
    X_input = X
    y_input = y
    if isinstance(X, pd.DataFrame):
        X_input = X.to_numpy(copy=False)
    if isinstance(y, pd.Series):
        y_input = y.to_numpy(copy=False)
    X_resampled, y_resampled = smote.fit_resample(X_input, y_input)
    if isinstance(X, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    if isinstance(y, pd.Series):
        y_resampled = pd.Series(y_resampled, name=y.name)
    return X_resampled, y_resampled


def _calibration_method(calibration: str) -> Optional[str]:
    calibration = calibration.lower()
    if calibration.startswith("platt") or calibration.startswith("sigmoid"):
        return "sigmoid"
    if calibration.startswith("isotonic"):
        return "isotonic"
    return None


def train_logistic_regression(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, object],
    balance: BalancePayload,
    calibration: str,
    cv_folds: int,
    n_jobs: int,
) -> TrainResult:
    """Train logistic regression with optional calibration."""

    class_weight = balance.class_weights if balance.strategy == "Class Weights" else None
    logistic_params = {
        "solver": params["solver"],
        "penalty": params["penalty"],
        "C": params["C"],
        "max_iter": params["max_iter"],
        "class_weight": class_weight,
        "n_jobs": n_jobs,
        "random_state": params.get("random_state", 42),
    }
    if params["penalty"] == "elasticnet":
        logistic_params["l1_ratio"] = params.get("l1_ratio", 0.5)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(**logistic_params)),
        ]
    )

    method = _calibration_method(calibration)
    fit_kwargs: Dict[str, np.ndarray] = {}
    if balance.strategy == "Focal Loss":
        if method is not None:
            raise ValueError(
                "Focal Loss balancing is not supported together with calibration for Logistic Regression."
            )
        if balance.sample_weight is not None:
            fit_kwargs["logreg__sample_weight"] = balance.sample_weight

    if method is None:
        pipeline.fit(balance.X_train, balance.y_train, **fit_kwargs)
        model = pipeline
    else:
        calibrator = CalibratedClassifierCV(
            pipeline,
            method=method,
            cv=max(2, min(cv_folds, 5)),
            n_jobs=n_jobs,
        )
        calibrator.fit(balance.X_train, balance.y_train)
        model = calibrator

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    return TrainResult(model=model, predictions=predictions, probabilities=probabilities, cv_estimator=pipeline)


def train_naive_bayes(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, object],
    balance: BalancePayload,
    calibration: str,
    cv_folds: int,
    n_jobs: int,
) -> TrainResult:
    """Train Gaussian Naive Bayes with optional calibration."""

    model = GaussianNB(var_smoothing=params["var_smoothing"])
    method = _calibration_method(calibration)
    fit_kwargs = {}
    if balance.sample_weight is not None:
        fit_kwargs["sample_weight"] = balance.sample_weight

    if method is None:
        model.fit(balance.X_train, balance.y_train, **fit_kwargs)
        final_model = model
        cv_estimator = model
    else:
        calibrator = CalibratedClassifierCV(
            model,
            method=method,
            cv=max(2, min(cv_folds, 5)),
            n_jobs=n_jobs,
        )
        calibrator.fit(balance.X_train, balance.y_train, **fit_kwargs)
        final_model = calibrator
        cv_estimator = calibrator

    predictions = final_model.predict(X_test)
    probabilities = final_model.predict_proba(X_test) if hasattr(final_model, "predict_proba") else None
    return TrainResult(model=final_model, predictions=predictions, probabilities=probabilities, cv_estimator=cv_estimator)


def train_xgboost(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, object],
    balance: BalancePayload,
    calibration: str,
    n_jobs: int,
) -> TrainResult:
    """Train an XGBoost classifier."""

    if XGBClassifier is None:  # pragma: no cover - validated in tests
        raise ValueError("Install the 'xgboost' package to use this model.")
    X_train_np = X_train.to_numpy(dtype=np.float32, copy=False)
    X_test_np = X_test.to_numpy(dtype=np.float32, copy=False)
    num_classes = y_train.nunique()
    objective = "multi:softprob" if num_classes > 2 else "binary:logistic"
    estimator = XGBClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        objective=objective,
        eval_metric="mlogloss",
        n_jobs=n_jobs,
        random_state=params.get("random_state", 42),
        tree_method="hist",
    )
    method = _calibration_method(calibration)
    fit_kwargs = {}
    if balance.sample_weight is not None:
        fit_kwargs["sample_weight"] = balance.sample_weight

    if method is None:
        estimator.fit(X_train_np, y_train, **fit_kwargs)
        model = estimator
    else:
        calibrator = CalibratedClassifierCV(
            estimator,
            method=method,
            cv=5,
            n_jobs=n_jobs,
        )
        calibrator.fit(X_train_np, y_train, **fit_kwargs)
        model = calibrator
    predictions = model.predict(X_test_np)
    probabilities = model.predict_proba(X_test_np) if hasattr(model, "predict_proba") else None
    cv_estimator = model if method is not None else estimator
    return TrainResult(model=model, predictions=predictions, probabilities=probabilities, cv_estimator=cv_estimator)


def train_lightgbm(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, object],
    balance: BalancePayload,
    calibration: str,
    n_jobs: int,
) -> TrainResult:
    """Train a LightGBM classifier."""

    if lgb is None:  # pragma: no cover - validated in tests
        raise ValueError("Install the 'lightgbm' package to use this model.")
    X_train_np = X_train.to_numpy(dtype=np.float32, copy=False)
    X_test_np = X_test.to_numpy(dtype=np.float32, copy=False)
    estimator = lgb.LGBMClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        num_leaves=params["num_leaves"],
        subsample=params["subsample"],
        class_weight=balance.class_weights if balance.strategy == "Class Weights" else None,
        n_jobs=n_jobs,
        random_state=params.get("random_state", 42),
    )
    method = _calibration_method(calibration)
    fit_kwargs = {}
    if balance.sample_weight is not None and balance.strategy != "Class Weights":
        fit_kwargs["sample_weight"] = balance.sample_weight

    if method is None:
        estimator.fit(X_train_np, y_train, **fit_kwargs)
        model = estimator
    else:
        calibrator = CalibratedClassifierCV(
            estimator,
            method=method,
            cv=5,
            n_jobs=n_jobs,
        )
        calibrator.fit(X_train_np, y_train, **fit_kwargs)
        model = calibrator
    predictions = model.predict(X_test_np)
    probabilities = model.predict_proba(X_test_np) if hasattr(model, "predict_proba") else None
    cv_estimator = model if method is not None else estimator
    return TrainResult(model=model, predictions=predictions, probabilities=probabilities, cv_estimator=cv_estimator)
