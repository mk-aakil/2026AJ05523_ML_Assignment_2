from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter",
    "Extent", "Solidity", "roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4"
]


def build_preprocessor(scale=True):
    """
    Build an unfitted numerical preprocessing pipeline.

    Parameters
    ----------
    scale : bool
        Whether to apply StandardScaler.

    Returns
    -------
    ColumnTransformer
        An unfitted preprocessing transformer.
    """

    steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale:
        steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor