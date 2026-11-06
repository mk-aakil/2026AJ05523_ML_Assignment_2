import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc


def leaderboard_chart(leaderboard):
    """Interactive grouped comparison of six metrics."""
    metric_columns = [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC",
    ]

    chart_df = leaderboard.melt(
        id_vars="Model",
        value_vars=metric_columns,
        var_name="Metric",
        value_name="Score"
    )

    fig = px.bar(
        chart_df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="Six-Metric Model Comparison",
        hover_data={"Score": ":.4f"},
        template="plotly_white"
    )

    fig.update_layout(
        height=460,
        legend_title_text="Classifier",
        margin=dict(l=20, r=20, t=65, b=20)
    )

    fig.update_yaxes(
        title="Score",
        range=[
            min(
                0,
                float(chart_df["Score"].min())
                if chart_df["Score"].notna().any()
                else 0
            ) - 0.05,
            1.05
        ]
    )

    return fig


def confusion_matrix_figure(
    matrix,
    classes,
    model_name,
    normalize=False
):
    """Draw raw or row-normalized confusion matrix."""
    matrix = np.asarray(matrix)

    if normalize:
        row_totals = matrix.sum(
            axis=1,
            keepdims=True
        )

        matrix = np.divide(
            matrix.astype(float),
            row_totals,
            out=np.zeros_like(
                matrix,
                dtype=float
            ),
            where=row_totals != 0
        )

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        vmin=0,
        vmax=1 if normalize else None,
        ax=ax
    )

    ax.set_title(
        f"{model_name} — Confusion Matrix"
    )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right"
    )

    fig.tight_layout()
    return fig


def roc_figure(
    y_true,
    probabilities,
    classes,
    model_name
):
    """Plot seven one-vs-rest ROC curves."""
    fig, ax = plt.subplots(figsize=(8, 6))

    y_true = np.asarray(y_true)

    for index, class_name in enumerate(classes):
        binary_truth = (
            y_true == class_name
        ).astype(int)

        # Every class must have positive and negative
        # observations for a valid ROC curve.
        if np.unique(binary_truth).size < 2:
            continue

        fpr, tpr, _ = roc_curve(
            binary_truth,
            probabilities[:, index]
        )

        class_auc = auc(fpr, tpr)

        ax.plot(
            fpr,
            tpr,
            label=f"{class_name}: {class_auc:.3f}"
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Chance"
    )

    ax.set_title(
        f"{model_name} — One-vs-Rest ROC"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    return fig