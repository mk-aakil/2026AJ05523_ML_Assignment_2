from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILES = [
    "logistic_regression.joblib",
    "decision_tree.joblib",
    "knn.joblib",
    "naive_bayes.joblib",
    "svm.joblib"
]


def test_saved_models_exist():
    for filename in MODEL_FILES:
        path = PROJECT_ROOT / "models" / filename
        assert path.is_file(), f"Missing: {filename}"


def test_saved_models_predict_probabilities():

    train_path = (
        PROJECT_ROOT / "data" / "processed" / "train_data.csv"
    )

    sample = pd.read_csv(train_path)[FEATURES].head(10)

    for filename in MODEL_FILES:

        model = joblib.load(
            PROJECT_ROOT / "models" / filename
        )

        predictions = model.predict(sample)
        probabilities = model.predict_proba(sample)

        assert len(predictions) == 10
        assert probabilities.shape == (10, 7)

        np.testing.assert_allclose(
            probabilities.sum(axis=1),
            np.ones(10),
            atol=1e-6
        )