import numpy as np
import pandas as pd

from app import FlowDataApp


class DummyVar:
    def __init__(self, value: str = "") -> None:
        self._value = value

    def get(self) -> str:
        return self._value

    def set(self, value: str) -> None:
        self._value = value


def test_classification_metrics_produces_curves_and_per_class():
    app = FlowDataApp.__new__(FlowDataApp)
    y_true = pd.Series([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_proba = np.array(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.75, 0.25],
            [0.4, 0.6],
            [0.1, 0.9],
        ]
    )
    metrics, conf_matrix, classes, evaluation = app._classification_metrics(
        y_true, y_pred, [0, 1], y_proba
    )

    assert conf_matrix.shape == (2, 2)
    assert list(classes) == [0, 1]
    assert metrics["per_class"] and len(metrics["per_class"]) == 2
    assert isinstance(metrics.get("roc_auc_macro"), float)
    assert isinstance(metrics.get("pr_auc_macro"), float)
    assert evaluation["roc_curves"] and evaluation["pr_curves"]
    for curve in evaluation["roc_curves"].values():
        assert curve["auc"] is not None


def test_predict_with_thresholds_applies_minimums():
    app = FlowDataApp.__new__(FlowDataApp)
    probabilities = np.array([[0.7, 0.3], [0.3, 0.7]])
    classes = ["A", "B"]
    thresholds = {"A": 0.2, "B": 0.8}
    baseline = np.array(["A", "B"])

    adjusted = app._predict_with_thresholds(probabilities, classes, thresholds, baseline)
    assert adjusted.tolist() == ["A", "A"]


def test_apply_threshold_changes_updates_metrics():
    app = FlowDataApp.__new__(FlowDataApp)
    y_true = pd.Series([0, 1, 0, 1])
    probabilities = np.array(
        [
            [0.7, 0.3],
            [0.4, 0.6],
            [0.65, 0.35],
            [0.2, 0.8],
        ]
    )
    baseline_pred = np.argmax(probabilities, axis=1)

    metrics, conf_matrix, classes, evaluation = app._classification_metrics(
        y_true, baseline_pred, [0, 1], probabilities
    )

    app.training_results = {
        "metrics": metrics.copy(),
        "confusion_matrix": conf_matrix,
        "classes": classes,
        "evaluation_details": evaluation,
        "thresholds": evaluation.get("default_thresholds", {}).copy(),
    }

    app.threshold_state = {
        "thresholds": {0: 0.0, 1: 0.9},
        "default_thresholds": evaluation.get("default_thresholds", {}).copy(),
        "classes": classes,
        "display_to_class": {"0": 0, "1": 1},
        "class_to_display": {0: "0", 1: "1"},
        "y_true": y_true.to_numpy(),
        "probabilities": probabilities,
        "baseline_predictions": baseline_pred,
        "roc_curves": evaluation.get("roc_curves", {}),
        "pr_curves": evaluation.get("pr_curves", {}),
    }

    app.threshold_class_var = DummyVar("1")
    app.threshold_value_var = DummyVar()
    app.threshold_summary_var = DummyVar()
    app.threshold_status_var = DummyVar()
    app.threshold_slider = None
    app.per_class_tree = None
    app.curves_canvas = None
    app.roc_ax = None
    app.pr_ax = None
    app.curves_fig = None

    app._apply_threshold_changes()

    updated_metrics = app.training_results["metrics"]
    evaluation_details = app.training_results["evaluation_details"]
    assert updated_metrics["thresholds"][1] == 0.9
    assert updated_metrics["accuracy"] < 1.0
    assert app.training_results["thresholds"][1] == 0.9
    assert app.threshold_state["baseline_predictions"].tolist() == baseline_pred.tolist()
    assert evaluation_details["baseline_predictions"].tolist() == baseline_pred.tolist()
    assert evaluation_details["adjusted_predictions"].tolist() == [0, 0, 0, 0]
    assert app.threshold_state["last_predictions"].tolist() == [0, 0, 0, 0]
