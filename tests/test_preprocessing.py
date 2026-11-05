import numpy as np
import pandas as pd

from src.preprocessing import build_preprocessor, FEATURES


def test_preprocessor_output_shape():

    sample = pd.DataFrame(
        np.ones((10, len(FEATURES))),
        columns=FEATURES
    )

    preprocessor = build_preprocessor(scale=True)

    transformed = preprocessor.fit_transform(sample)

    assert transformed.shape == (10, 16)


def test_preprocessor_handles_missing_values():

    sample = pd.DataFrame(
        np.arange(160, dtype=float).reshape(10, 16),
        columns=FEATURES
    )

    sample.loc[0, "Area"] = np.nan

    preprocessor = build_preprocessor(scale=True)

    transformed = preprocessor.fit_transform(sample)

    assert np.isfinite(transformed).all()