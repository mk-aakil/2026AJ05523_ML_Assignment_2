import numpy as np
import pandas as pd
import pytest

from app.components.evaluation import (
    validate_dataset,
    evaluate_pipeline,
    make_prediction_dataframe,
)


FEATURES = ["Area", "Perimeter"]
CLASSES = ["A", "B", "C"]


class DummyModel:
    classes_ = np.array(CLASSES)

    def predict(self, X):
        return np.array(
            ["A", "B", "C"][:len(X)]
        )

    def predict_proba(self, X):
        return np.array([
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ])[:len(X)]


@pytest.fixture
def labeled_df():
    return pd.DataFrame({
        "Area": [100, 200, 300],
        "Perimeter": [40, 60, 80],
        "Class": ["A", "B", "C"]
    })


def test_labeled_csv(labeled_df):
    X, y, has_labels = validate_dataset(
        labeled_df,
        FEATURES,
        "Class",
        CLASSES
    )

    assert X.shape == (3, 2)
    assert has_labels is True
    assert len(y) == 3


def test_unlabeled_csv(labeled_df):
    df = labeled_df.drop(columns=["Class"])

    X, y, has_labels = validate_dataset(
        df,
        FEATURES,
        "Class",
        CLASSES
    )

    assert X.shape == (3, 2)
    assert y is None
    assert has_labels is False


def test_missing_feature_rejected(labeled_df):
    df = labeled_df.drop(columns=["Area"])

    with pytest.raises(ValueError):
        validate_dataset(
            df,
            FEATURES,
            "Class",
            CLASSES
        )


def test_evaluation_with_labels(labeled_df):
    model = DummyModel()

    X = labeled_df[FEATURES]
    y = labeled_df["Class"]

    result = evaluate_pipeline(
        model,
        X,
        y,
        CLASSES
    )

    assert result["metrics"]["Accuracy"] == 1.0
    assert result["metrics"]["F1"] == 1.0
    assert result["metrics"]["MCC"] == 1.0
    assert result["metrics"]["AUC"] == 1.0


def test_prediction_only_mode(labeled_df):
    model = DummyModel()

    X = labeled_df[FEATURES]

    result = evaluate_pipeline(
        model,
        X,
        None,
        CLASSES
    )

    assert result["metrics"] is None
    assert len(result["predictions"]) == 3


def test_prediction_download_columns(labeled_df):
    model = DummyModel()

    result = evaluate_pipeline(
        model,
        labeled_df[FEATURES],
        labeled_df["Class"],
        CLASSES
    )

    output = make_prediction_dataframe(
        labeled_df,
        result,
        CLASSES,
        "Class"
    )

    assert "Predicted_Class" in output.columns
    assert "Correct" in output.columns
    assert "Probability_A" in output.columns
    assert len(output) == 3


def test_missing_class_auc_unavailable():
    df = pd.DataFrame({
        "Area": [100, 200],
        "Perimeter": [40, 60],
        "Class": ["A", "B"]
    })

    model = DummyModel()

    result = evaluate_pipeline(
        model,
        df[FEATURES],
        df["Class"],
        CLASSES
    )

    assert np.isnan(result["metrics"]["AUC"])
    assert result["roc_data"] is None