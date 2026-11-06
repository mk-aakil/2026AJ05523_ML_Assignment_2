import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)


def test_multiclass_metrics():

    y_true = np.array([
        "A", "A", "B", "B", "C", "C"
    ])

    y_pred = np.array([
        "A", "A", "B", "B", "C", "C"
    ])

    probabilities = np.array([
        [0.9, 0.05, 0.05],
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.05, 0.9, 0.05],
        [0.1, 0.1, 0.8],
        [0.05, 0.05, 0.9]
    ])

    labels = ["A", "B", "C"]

    assert accuracy_score(y_true, y_pred) == 1.0

    assert precision_score(
        y_true, y_pred, average="macro"
    ) == 1.0

    assert recall_score(
        y_true, y_pred, average="macro"
    ) == 1.0

    assert f1_score(
        y_true, y_pred, average="macro"
    ) == 1.0

    assert matthews_corrcoef(
        y_true, y_pred
    ) == 1.0

    assert np.isclose(
        roc_auc_score(
            y_true,
            probabilities,
            labels=labels,
            multi_class="ovr",
            average="macro"
        ),
        1.0
    )