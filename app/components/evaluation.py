import re
from time import perf_counter

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


METRICS = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC",
]


def validate_dataset(df, features, target, classes):
    """
    Validate a labeled or unlabeled prediction CSV.

    Returns:
        X: numeric feature DataFrame
        y: actual labels or None
        has_labels: boolean
    """
    if df.empty:
        raise ValueError("The uploaded CSV is empty.")

    if df.columns.duplicated().any():
        raise ValueError("The CSV contains duplicate column names.")

    missing_features = [
        feature for feature in features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_features)
        )

    X = df.loc[:, features].copy()

    # All Dry Bean predictors must be numeric.
    for column in features:
        X[column] = pd.to_numeric(
            X[column],
            errors="raise"
        )

    # Median imputation in each saved pipeline
    # handles missing numerical values.
    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError(
            "Feature columns contain infinite values."
        )

    completely_missing = X.columns[
        X.isna().all()
    ].tolist()

    if completely_missing:
        raise ValueError(
            "Entirely missing feature columns: "
            + ", ".join(completely_missing)
        )

    has_labels = target in df.columns

    if not has_labels:
        return X, None, False

    y = df[target].copy()

    if y.isna().any():
        raise ValueError(
            "The target column contains missing labels. "
            "Remove the target column for prediction-only mode."
        )

    unknown_classes = sorted(
        set(y.unique()) - set(classes)
    )

    if unknown_classes:
        raise ValueError(
            f"Unknown target classes: {unknown_classes}"
        )

    return X, y, True


def evaluate_pipeline(
    model,
    X,
    y=None,
    class_labels=None
):
    """
    Run inference on an already-fitted pipeline.

    When labels are available, calculate six metrics.
    When labels are absent, return predictions only.
    """
    class_labels = list(class_labels)

    start = perf_counter()

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    elapsed_ms = (
        perf_counter() - start
    ) * 1000

    # Ensure probability columns match metadata order.
    model_classes = list(model.classes_)

    if set(model_classes) != set(class_labels):
        raise ValueError(
            "Saved model classes do not match metadata."
        )

    indices = [
        model_classes.index(label)
        for label in class_labels
    ]

    probabilities = probabilities[:, indices]

    if probabilities.shape != (
        len(X), len(class_labels)
    ):
        raise ValueError(
            "Unexpected probability matrix shape."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Model returned non-finite probabilities."
        )

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(len(X)),
        atol=1e-5
    )

    result = {
        "predictions": predictions,
        "probabilities": probabilities,
        "prediction_time_ms": elapsed_ms,
        "metrics": None,
        "confusion_matrix": None,
        "classification_report": None,
        "roc_data": None,
    }

    if y is None:
        return result

    # AUC requires both positive and negative examples
    # for every class in the one-vs-rest evaluation.
    class_counts = pd.Series(y).value_counts()

    auc_available = all(
        0 < class_counts.get(label, 0) < len(y)
        for label in class_labels
    )

    auc_score = np.nan

    if auc_available:
        auc_score = roc_auc_score(
            y,
            probabilities,
            labels=class_labels,
            multi_class="ovr",
            average="macro"
        )

    metrics = {
        "Accuracy": accuracy_score(
            y, predictions
        ),
        "AUC": auc_score,
        "Precision": precision_score(
            y,
            predictions,
            labels=class_labels,
            average="macro",
            zero_division=0
        ),
        "Recall": recall_score(
            y,
            predictions,
            labels=class_labels,
            average="macro",
            zero_division=0
        ),
        "F1": f1_score(
            y,
            predictions,
            labels=class_labels,
            average="macro",
            zero_division=0
        ),
        "MCC": matthews_corrcoef(
            y, predictions
        ),
    }

    result["metrics"] = metrics

    result["confusion_matrix"] = confusion_matrix(
        y,
        predictions,
        labels=class_labels
    )

    result["classification_report"] = (
        classification_report(
            y,
            predictions,
            labels=class_labels,
            output_dict=True,
            zero_division=0
        )
    )

    result["roc_data"] = (
        {
            "labels": np.asarray(y),
            "probabilities": probabilities,
        }
        if auc_available
        else None
    )

    return result


def make_prediction_dataframe(
    original_df,
    result,
    classes,
    target
):
    """
    Preserve original input columns and append predictions.
    """
    output = original_df.copy()

    # Avoid silently overwriting existing user columns.
    generated = [
        "Predicted_Class",
        "Correct",
    ] + [
        f"Probability_{label}"
        for label in classes
    ]

    conflicts = [
        name for name in generated
        if name in output.columns
    ]

    if conflicts:
        raise ValueError(
            "Input CSV contains reserved output columns: "
            + ", ".join(conflicts)
        )

    output["Predicted_Class"] = result["predictions"]

    if target in original_df.columns:
        output["Correct"] = (
            original_df[target].to_numpy()
            == result["predictions"]
        )

    for index, label in enumerate(classes):
        output[f"Probability_{label}"] = (
            result["probabilities"][:, index]
        )

    return output


def safe_model_filename(model_name):
    """Convert display names into safe CSV filenames."""
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        model_name.lower()
    ).strip("_")