"""
Headless smoke test for the FlowDataApp training pipeline.

This script bypasses the GUI (which requires a display) and instead exercises the
same data preparation workflow plus a couple of representative model families
used in the app (Random Forest + SVM). It relies on the demo CSVs generated
under sample_data/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from app import RANDOM_STATE


def _load_sample_data() -> tuple[pd.DataFrame, list[str], pd.Series]:
    sample_dir = Path("sample_data")
    csv_paths = sorted(sample_dir.glob("demo_flow_*.csv"))
    assert csv_paths, "Sample CSVs not found. Run sample_data/generate_sample_data.py first."

    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        frames.append(df.assign(__source_file=path.name))

    data = pd.concat(frames, ignore_index=True)
    assert "CellType" in data.columns, "Expected CellType column in sample data."

    feature_columns = [col for col in data.columns if col not in {"CellType", "__source_file"}]
    assert feature_columns, "No feature columns detected."

    return data[feature_columns], feature_columns, data["CellType"]


def _train_random_forest(X_train, X_test, y_train, y_test) -> tuple[float, float]:
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=1,
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1_macro = f1_score(y_test, predictions, average="macro")

    importances = model.feature_importances_
    assert np.any(importances > 0), "Feature importances all zero."

    return accuracy, f1_macro


def _train_svm(X_train, X_test, y_train, y_test) -> tuple[float, float]:
    svm_clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale")),
        ]
    )
    svm_clf.fit(X_train, y_train)
    predictions = svm_clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1_macro = f1_score(y_test, predictions, average="macro")
    return accuracy, f1_macro


def run_headless_smoke_test() -> None:
    X, features, y = _load_sample_data()
    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    rf_metrics = _train_random_forest(X_train, X_test, y_train, y_test)
    svm_metrics = _train_svm(X_train, X_test, y_train, y_test)

    print(
        "Headless smoke test passed.\n"
        f"- RandomForest  Accuracy: {rf_metrics[0]:.3f}, Macro F1: {rf_metrics[1]:.3f}\n"
        f"- SVM           Accuracy: {svm_metrics[0]:.3f}, Macro F1: {svm_metrics[1]:.3f}\n"
        f"Features used: {len(features)}"
    )


if __name__ == "__main__":
    run_headless_smoke_test()
