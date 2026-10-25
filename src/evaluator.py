"""Utilities for training classifiers and measuring their performance."""

import time
from collections import OrderedDict
from typing import Any

from sklearn.metrics import accuracy_score


def evaluate_model(
    name: str,
    model: Any,
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
) -> OrderedDict:
    """Train and evaluate one classifier while measuring each operation.

    Args:
        name: Display name assigned to the classifier.
        model: Scikit-learn compatible classifier instance.
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        y_train: Training target values.
        y_test: Test target values.

    Returns:
        OrderedDict: Classifier name, accuracy, training time, and prediction
            time in a stable insertion order.
    """
    training_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - training_start

    prediction_start = time.perf_counter()
    predictions = model.predict(X_test)
    prediction_time = time.perf_counter() - prediction_start

    return OrderedDict(
        [
            ("name", name),
            ("accuracy", accuracy_score(y_test, predictions)),
            ("training_time", training_time),
            ("prediction_time", prediction_time),
        ]
    )
