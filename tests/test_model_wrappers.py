from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model_wrappers import (
    BalancePayload,
    apply_smote,
    class_weight_sample_array,
    compute_class_weight_dict,
    focal_sample_weight_array,
    train_lightgbm,
    train_logistic_regression,
    train_naive_bayes,
    train_xgboost,
)

RANDOM_STATE = 42


def _make_dataset():
    X, y = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        n_classes=2,
        weights=[0.85, 0.15],
        random_state=RANDOM_STATE,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="target")
    return train_test_split(
        X_df,
        y_series,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_series,
    )


def test_focal_weights_emphasize_minority():
    y = pd.Series([0] * 90 + [1] * 10)
    class_weights = compute_class_weight_dict(y)
    focal_weights = focal_sample_weight_array(y, class_weights, gamma=2.0)
    majority_weight = float(np.mean(focal_weights[y.to_numpy() == 0]))
    minority_weight = float(np.mean(focal_weights[y.to_numpy() == 1]))
    assert minority_weight > majority_weight


def test_logistic_regression_smote_training():
    X_train, X_test, y_train, y_test = _make_dataset()
    pytest.importorskip("imblearn", reason="SMOTE balancing requires imbalanced-learn")
    X_balanced, y_balanced = apply_smote(X_train, y_train, RANDOM_STATE)
    balance = BalancePayload(X_train=X_balanced, y_train=y_balanced, strategy="SMOTE")
    params = {
        "solver": "lbfgs",
        "penalty": "l2",
        "C": 1.0,
        "max_iter": 200,
        "l1_ratio": 0.5,
        "random_state": RANDOM_STATE,
    }
    result = train_logistic_regression(
        balance.X_train,
        X_test,
        balance.y_train,
        params,
        balance,
        calibration="None",
        cv_folds=3,
        n_jobs=1,
    )
    assert len(result.predictions) == len(X_test)
    assert result.probabilities is not None


def test_naive_bayes_focal_calibrated():
    X_train, X_test, y_train, y_test = _make_dataset()
    class_weights = compute_class_weight_dict(y_train)
    focal_weights = focal_sample_weight_array(y_train, class_weights, gamma=2.0)
    balance = BalancePayload(
        X_train=X_train,
        y_train=y_train,
        strategy="Focal Loss",
        sample_weight=focal_weights,
    )
    params = {"var_smoothing": 1e-9}
    result = train_naive_bayes(
        balance.X_train,
        X_test,
        balance.y_train,
        params,
        balance,
        calibration="Platt (sigmoid)",
        cv_folds=3,
        n_jobs=1,
    )
    assert isinstance(result.model, CalibratedClassifierCV)
    assert len(result.predictions) == len(X_test)


def test_xgboost_class_weights_training():
    X_train, X_test, y_train, y_test = _make_dataset()
    class_weights = compute_class_weight_dict(y_train)
    sample_weight = class_weight_sample_array(y_train, class_weights)
    balance = BalancePayload(
        X_train=X_train,
        y_train=y_train,
        strategy="Class Weights",
        class_weights=class_weights,
        sample_weight=sample_weight,
    )
    params = {
        "n_estimators": 40,
        "learning_rate": 0.1,
        "max_depth": 3,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": RANDOM_STATE,
    }
    result = train_xgboost(
        balance.X_train,
        X_test,
        balance.y_train,
        params,
        balance,
        calibration="None",
        n_jobs=1,
    )
    assert len(result.predictions) == len(X_test)


def test_lightgbm_isotonic_calibration():
    X_train, X_test, y_train, y_test = _make_dataset()
    class_weights = compute_class_weight_dict(y_train)
    sample_weight = class_weight_sample_array(y_train, class_weights)
    balance = BalancePayload(
        X_train=X_train,
        y_train=y_train,
        strategy="Class Weights",
        class_weights=class_weights,
        sample_weight=sample_weight,
    )
    params = {
        "n_estimators": 60,
        "learning_rate": 0.1,
        "max_depth": -1,
        "num_leaves": 31,
        "subsample": 0.9,
        "random_state": RANDOM_STATE,
    }
    result = train_lightgbm(
        balance.X_train,
        X_test,
        balance.y_train,
        params,
        balance,
        calibration="Isotonic",
        n_jobs=1,
    )
    assert isinstance(result.model, CalibratedClassifierCV)
    assert len(result.predictions) == len(X_test)
