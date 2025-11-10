"""Reusable training pipeline utilities for headless workflows.

This module mirrors the model-training behaviours offered by the Tk UI but in a
library-friendly form so that automated tools (for example the CLI and tests)
can invoke the same model recipes without bootstrapping a Tkinter runtime.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DEFAULT_RANDOM_STATE = 42


@dataclasses.dataclass
class TrainingResult:
    """Container describing a single training run."""

    model_name: str
    estimator: object
    metrics: Dict[str, object]
    confusion_matrix: np.ndarray
    classes: List[str]
    feature_importances: Sequence[float]
    train_rows: int
    test_rows: int
    cv_scores: Optional[np.ndarray]
    cv_warning: str
    artifacts: Dict[str, object]
    features: Sequence[str]
    target: str
    config: Dict[str, object]
    training_time: float


def _classification_metrics(
    y_true: pd.Series, y_pred: np.ndarray, class_labels: Optional[Sequence[str]] = None
) -> Tuple[Dict[str, object], np.ndarray, List[str]]:
    """Compute the standard set of metrics returned by the UI."""

    accuracy = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    report_text = classification_report(y_true, y_pred, digits=3, zero_division=0)
    report_dict_raw = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    report_dict: Dict[str, object] = {}
    for key, value in report_dict_raw.items():
        if isinstance(value, Mapping):
            report_dict[key] = {inner_key: float(inner_val) for inner_key, inner_val in value.items()}
        else:
            report_dict[key] = float(value)

    if class_labels is None:
        class_labels = sorted(pd.unique(pd.Series(list(y_true) + list(y_pred))))
    matrix = confusion_matrix(y_true, y_pred, labels=class_labels)

    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "report_text": report_text,
        "report_dict": report_dict,
    }
    return metrics, matrix, list(class_labels)


def _perform_cross_validation(
    estimator: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int,
    n_jobs: int,
) -> Tuple[Optional[np.ndarray], str]:
    if cv_folds < 2 or len(y_train) < cv_folds:
        return None, ""
    try:
        scores = cross_val_score(
            estimator,
            X_train,
            y_train,
            cv=cv_folds,
            scoring="f1_macro",
            n_jobs=n_jobs,
        )
        return scores, ""
    except ValueError as exc:  # Happens when the folds are incompatible with the data
        return None, str(exc)


def _train_random_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    params: Mapping[str, object],
    cv_folds: int,
    n_jobs: int,
) -> TrainingResult:
    model = RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=None if params["max_depth"] is None else int(params["max_depth"]),
        max_features=str(params["max_features"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        n_jobs=n_jobs,
        random_state=DEFAULT_RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics, matrix, classes = _classification_metrics(y_test, predictions, list(model.classes_))
    cv_scores, cv_warning = _perform_cross_validation(
        RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=None if params["max_depth"] is None else int(params["max_depth"]),
            max_features=str(params["max_features"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            n_jobs=n_jobs,
            random_state=DEFAULT_RANDOM_STATE,
        ),
        X_train,
        y_train,
        cv_folds,
        n_jobs,
    )
    return TrainingResult(
        model_name="Random Forest",
        estimator=model,
        metrics=metrics,
        confusion_matrix=matrix,
        classes=classes,
        feature_importances=model.feature_importances_.tolist(),
        train_rows=len(X_train),
        test_rows=len(X_test),
        cv_scores=cv_scores,
        cv_warning=cv_warning,
        artifacts={},
        features=list(X_train.columns),
        target=y_train.name or "target",
        config={"model_params": dict(params)},
        training_time=0.0,
    )


def _train_lda(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    params: Mapping[str, object],
    cv_folds: int,
    n_jobs: int,
) -> TrainingResult:
    lda = LinearDiscriminantAnalysis(
        solver=str(params["solver"]),
        shrinkage=params.get("shrinkage"),
    )
    pipeline = Pipeline([("scaler", StandardScaler()), ("lda", lda)])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics, matrix, classes = _classification_metrics(y_test, predictions)
    cv_scores, cv_warning = _perform_cross_validation(
        pipeline, X_train, y_train, cv_folds, n_jobs
    )
    return TrainingResult(
        model_name="LDA",
        estimator=pipeline,
        metrics=metrics,
        confusion_matrix=matrix,
        classes=classes,
        feature_importances=[],
        train_rows=len(X_train),
        test_rows=len(X_test),
        cv_scores=cv_scores,
        cv_warning=cv_warning,
        artifacts={},
        features=list(X_train.columns),
        target=y_train.name or "target",
        config={"model_params": dict(params)},
        training_time=0.0,
    )


def _train_svm(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    params: Mapping[str, object],
    cv_folds: int,
    n_jobs: int,
) -> TrainingResult:
    svm = SVC(
        kernel=str(params["kernel"]),
        C=float(params["C"]),
        gamma=str(params["gamma"]),
        degree=int(params["degree"]),
        probability=False,
    )
    pipeline = Pipeline([("scaler", StandardScaler()), ("svm", svm)])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics, matrix, classes = _classification_metrics(y_test, predictions)
    cv_scores, cv_warning = _perform_cross_validation(
        pipeline, X_train, y_train, cv_folds, n_jobs
    )
    return TrainingResult(
        model_name="SVM",
        estimator=pipeline,
        metrics=metrics,
        confusion_matrix=matrix,
        classes=classes,
        feature_importances=[],
        train_rows=len(X_train),
        test_rows=len(X_test),
        cv_scores=cv_scores,
        cv_warning=cv_warning,
        artifacts={},
        features=list(X_train.columns),
        target=y_train.name or "target",
        config={"model_params": dict(params)},
        training_time=0.0,
    )


def _train_logistic_regression(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    params: Mapping[str, object],
    cv_folds: int,
    n_jobs: int,
) -> TrainingResult:
    lr_kwargs: Dict[str, object] = {
        "solver": str(params["solver"]),
        "penalty": str(params["penalty"]),
        "C": float(params["C"]),
        "max_iter": int(params["max_iter"]),
        "class_weight": params.get("class_weight"),
        "n_jobs": n_jobs,
        "random_state": DEFAULT_RANDOM_STATE,
    }
    if params.get("penalty") == "elasticnet":
        lr_kwargs["l1_ratio"] = float(params.get("l1_ratio", 0.5))
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(**lr_kwargs)),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics, matrix, classes = _classification_metrics(y_test, predictions)
    cv_scores, cv_warning = _perform_cross_validation(
        pipeline, X_train, y_train, cv_folds, n_jobs
    )
    return TrainingResult(
        model_name="Logistic Regression",
        estimator=pipeline,
        metrics=metrics,
        confusion_matrix=matrix,
        classes=classes,
        feature_importances=[],
        train_rows=len(X_train),
        test_rows=len(X_test),
        cv_scores=cv_scores,
        cv_warning=cv_warning,
        artifacts={},
        features=list(X_train.columns),
        target=y_train.name or "target",
        config={"model_params": dict(params)},
        training_time=0.0,
    )


def _train_naive_bayes(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    params: Mapping[str, object],
    cv_folds: int,
    n_jobs: int,
) -> TrainingResult:
    model = GaussianNB(var_smoothing=float(params["var_smoothing"]))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics, matrix, classes = _classification_metrics(y_test, predictions)
    return TrainingResult(
        model_name="Naive Bayes",
        estimator=model,
        metrics=metrics,
        confusion_matrix=matrix,
        classes=classes,
        feature_importances=[],
        train_rows=len(X_train),
        test_rows=len(X_test),
        cv_scores=None,
        cv_warning="",
        artifacts={},
        features=list(X_train.columns),
        target=y_train.name or "target",
        config={"model_params": dict(params)},
        training_time=0.0,
    )


TRAINERS = {
    "random_forest": _train_random_forest,
    "lda": _train_lda,
    "svm": _train_svm,
    "logistic_regression": _train_logistic_regression,
    "naive_bayes": _train_naive_bayes,
}


DEFAULT_PARAMS: Dict[str, Dict[str, object]] = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": None,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
    },
    "lda": {"solver": "svd", "shrinkage": None},
    "svm": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "degree": 3},
    "logistic_regression": {
        "solver": "lbfgs",
        "penalty": "l2",
        "C": 1.0,
        "max_iter": 200,
        "l1_ratio": 0.5,
        "class_weight": None,
    },
    "naive_bayes": {"var_smoothing": 1e-9},
}


def available_models() -> List[str]:
    return sorted(TRAINERS.keys())


def normalise_model_name(model_name: str) -> str:
    key = model_name.strip().lower().replace(" ", "_")
    aliases = {
        "rf": "random_forest",
        "randomforest": "random_forest",
        "linear_discriminant_analysis": "lda",
        "svc": "svm",
        "support_vector_machine": "svm",
        "logreg": "logistic_regression",
        "gaussian_nb": "naive_bayes",
    }
    return aliases.get(key, key)


def resolve_model_params(model_key: str, overrides: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
    params = dict(DEFAULT_PARAMS[model_key])
    if overrides:
        params.update(overrides)
    return params


def train_and_evaluate(
    data: pd.DataFrame,
    target: str,
    features: Sequence[str],
    model_name: str,
    params: Optional[Mapping[str, object]] = None,
    *,
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
    cv_folds: int = 0,
    n_jobs: int = 1,
) -> TrainingResult:
    key = normalise_model_name(model_name)
    if key not in TRAINERS:
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {', '.join(available_models())}")

    params_resolved = resolve_model_params(key, params)
    X = data.loc[:, list(features)]
    y = data.loc[:, target]

    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    start = time.time()
    trainer = TRAINERS[key]
    result = trainer(X_train, X_test, y_train, y_test, params_resolved, cv_folds, n_jobs)
    result.training_time = time.time() - start
    result.features = list(features)
    result.target = target
    result.config = {
        "model_params": dict(params_resolved),
        "test_size": test_size,
        "cv_folds": cv_folds,
        "n_jobs": n_jobs,
        "random_state": random_state,
    }
    return result


def prepare_metadata(result: TrainingResult) -> Dict[str, object]:
    return {
        "model_name": result.model_name,
        "metrics": {
            "accuracy": float(result.metrics.get("accuracy", 0.0)),
            "f1_macro": float(result.metrics.get("f1_macro", 0.0)),
            "f1_weighted": float(result.metrics.get("f1_weighted", 0.0)),
        },
        "features": list(result.features),
        "target": result.target,
        "train_rows": result.train_rows,
        "test_rows": result.test_rows,
        "classes": list(map(str, result.classes)),
        "cv_scores": result.cv_scores.tolist() if isinstance(result.cv_scores, np.ndarray) else None,
        "cv_warning": result.cv_warning,
        "config": result.config,
    }


def validate_features(data_columns: Iterable[str], requested: Sequence[str]) -> List[str]:
    columns = list(data_columns)
    missing = [col for col in requested if col not in columns]
    if missing:
        raise ValueError(f"Columns not found in data: {', '.join(missing)}")
    return list(requested)

