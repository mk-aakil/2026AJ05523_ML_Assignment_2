
from pathlib import Path
from io import BytesIO
from time import perf_counter
from zipfile import ZipFile
import json

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from app.components.evaluation import (
    METRICS,
    validate_dataset,
    evaluate_pipeline,
    make_prediction_dataframe,
    safe_model_filename,
)

from app.components.charts import (
    leaderboard_chart,
    confusion_matrix_figure,
    roc_figure,
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Model Benchmark Arena | Dry Bean",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
DEFAULT_CSV = ROOT / "data" / "samples" / "default_test.csv"


# ============================================================
# 2. DASHBOARD STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    [data-testid="stMetric"] {
        background: rgba(120, 120, 120, 0.07);
        border: 1px solid rgba(120, 120, 120, 0.15);
        border-radius: 12px;
        padding: 15px;
    }

    div.stButton > button {
        border-radius: 10px;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. HEADER - DISPLAY BEFORE PROCESSING
# ============================================================

st.title("🫘 Model Benchmark Arena")

st.caption(
    "Dry Bean Classification | "
    "BITS M.Tech AIML — Machine Learning Assignment 2"
)

st.write(
    "Compare five trained machine learning classifiers "
    "using six evaluation metrics, confusion matrices, "
    "ROC curves, prediction timing and downloadable results."
)


# ============================================================
# 4. CACHED LOADERS
# ============================================================

@st.cache_data(show_spinner=False)
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_default_data():
    return pd.read_csv(DEFAULT_CSV)


@st.cache_resource(show_spinner=False)
def load_model(model_name, filename):
    # Saved pipeline is loaded; no retraining.
    return joblib.load(MODEL_DIR / filename)


# ============================================================
# 5. LOAD METADATA
# ============================================================

if not METADATA_PATH.is_file():
    st.error(f"Missing metadata: {METADATA_PATH}")
    st.stop()

try:
    metadata = load_metadata()

    FEATURES = metadata["features"]
    TARGET = metadata["target"]
    CLASSES = metadata["classes"]
    MODEL_FILES = metadata["model_files"]

except Exception as exc:
    st.error(f"Unable to load model metadata: {exc}")
    st.stop()

if not MODEL_FILES:
    st.error("No trained models are configured.")
    st.stop()


# ============================================================
# 6. SIDEBAR
# ============================================================

st.sidebar.title("🫘 Benchmark Controls")

st.sidebar.caption(
    "Choose a classifier and optionally upload a dataset."
)

model_options = ["All Models", *MODEL_FILES.keys()]

selected_model = st.sidebar.selectbox(
    "Select classifier",
    options=model_options,
    index=0,
)

st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV (optional)",
    type=["csv"],
    help=(
        "CSV must contain all 16 numerical features. "
        "The Class column is optional."
    ),
)

st.sidebar.divider()

with st.sidebar.expander("Required CSV columns"):
    st.write("**Input features:**")
    st.code("\n".join(FEATURES))
    st.write(f"**Optional target:** {TARGET}")


# ============================================================
# 7. AUTOMATIC DATASET LOADING
# ============================================================

st.subheader("Dataset")

if uploaded_file is None:

    source_name = "Default held-out test dataset"

    if not DEFAULT_CSV.is_file():
        st.error(
            f"Default test dataset not found: {DEFAULT_CSV}"
        )
        st.stop()

    try:
        input_df = load_default_data().copy()
    except Exception as exc:
        st.error(f"Unable to load default CSV: {exc}")
        st.stop()

    st.success(
        "Default held-out test dataset loaded automatically."
    )

else:

    source_name = uploaded_file.name

    try:
        input_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Unable to read uploaded CSV: {exc}")
        st.stop()

    st.success(f"Uploaded dataset loaded: {source_name}")


# ============================================================
# 8. VALIDATE DATASET
# ============================================================

try:
    X, y, has_labels = validate_dataset(
        df=input_df,
        features=FEATURES,
        target=TARGET,
        classes=CLASSES,
    )
except (ValueError, TypeError) as exc:
    st.error(f"Dataset validation failed: {exc}")
    st.stop()

selected_names = (
    list(MODEL_FILES.keys())
    if selected_model == "All Models"
    else [selected_model]
)

st.sidebar.caption(f"Source: {source_name}")
st.sidebar.caption(f"Rows: {len(input_df):,}")
st.sidebar.caption(
    "Mode: Evaluation" if has_labels else "Mode: Prediction"
)


# ============================================================
# 9. DATASET SUMMARY
# ============================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric("Models Selected", len(selected_names))
c2.metric("Dataset Records", f"{len(input_df):,}")
c3.metric("Input Features", len(FEATURES))
c4.metric("Bean Classes", len(CLASSES))

st.caption(f"Data source: {source_name}")

if has_labels:
    st.info(
        "Labeled dataset detected. Six evaluation metrics "
        "will be calculated using the actual Class labels."
    )
else:
    st.info(
        "Unlabeled dataset detected. Predictions and "
        "probabilities will be generated without "
        "label-dependent evaluation metrics."
    )

with st.expander("Preview Input Dataset"):
    st.dataframe(
        input_df.head(20),
        width="stretch",
        hide_index=True,
    )


# ============================================================
# 10. LIVE MODEL EVALUATION
# ============================================================

st.divider()
st.subheader("Model Evaluation")

st.caption(
    "Running inference with previously trained "
    "and saved machine learning pipelines."
)

results = {}
timing_records = []
total_models = len(selected_names)

with st.status(
    "Preparing model evaluation...",
    expanded=True,
) as evaluation_status:

    progress_bar = st.progress(
        0,
        text="Starting evaluation...",
    )

    for index, model_name in enumerate(selected_names):

        st.write(
            f"**Model {index + 1}/{total_models}: "
            f"{model_name}**"
        )

        progress_bar.progress(
            int(index / total_models * 100),
            text=f"Evaluating {model_name}...",
        )

        filename = MODEL_FILES[model_name]
        model_path = MODEL_DIR / filename

        if not model_path.is_file():
            st.error(f"Missing saved model: {model_path}")
            st.stop()

        try:
            load_start = perf_counter()

            model = load_model(model_name, filename)

            load_time = perf_counter() - load_start

            evaluation_start = perf_counter()

            result = evaluate_pipeline(
                model=model,
                X=X,
                y=y,
                class_labels=CLASSES,
            )

            evaluation_time = (
                perf_counter() - evaluation_start
            )

            results[model_name] = result

            timing_records.append({
                "Model": model_name,
                "Model Load Time (s)": load_time,
                "Evaluation Time (s)": evaluation_time,
                "Prediction Time (ms)": result[
                    "prediction_time_ms"
                ],
            })

            st.write(
                f"Completed {model_name} "
                f"in {evaluation_time:.3f} seconds."
            )

        except Exception as exc:
            st.error(
                f"Evaluation failed for {model_name}: {exc}"
            )
            st.exception(exc)
            st.stop()

        progress_bar.progress(
            int((index + 1) / total_models * 100),
            text=(
                f"Completed {index + 1} "
                f"of {total_models} models"
            ),
        )

    evaluation_status.update(
        label=f"Completed {total_models} model(s)",
        state="complete",
        expanded=False,
    )

st.success(
    f"Successfully evaluated {len(results)} model(s)."
)


# ============================================================
# 11. PREDICTION TIMING - ALWAYS VISIBLE
# ============================================================

st.subheader("⏱️ Inference Timing")

timing_df = pd.DataFrame(timing_records)

timing_df["Milliseconds per Row"] = (
    timing_df["Prediction Time (ms)"] / len(input_df)
)

st.dataframe(
    timing_df.style.format({
        "Model Load Time (s)": "{:.4f}",
        "Evaluation Time (s)": "{:.4f}",
        "Prediction Time (ms)": "{:.3f}",
        "Milliseconds per Row": "{:.5f}",
    }),
    width="stretch",
    hide_index=True,
)

fastest_row = timing_df.loc[
    timing_df["Prediction Time (ms)"].idxmin()
]

st.caption(
    f"Fastest measured inference: "
    f"{fastest_row['Model']} "
    f"({fastest_row['Prediction Time (ms)']:.3f} ms). "
    "Prediction timing excludes model loading."
)


# ============================================================
# 12. BUILD LEADERBOARD
# ============================================================

leaderboard_records = []

for model_name, result in results.items():

    if result["metrics"] is None:
        continue

    leaderboard_records.append({
        "Model": model_name,
        **result["metrics"],
        "Prediction Time (ms)": result[
            "prediction_time_ms"
        ],
    })

leaderboard = pd.DataFrame(leaderboard_records)

if not leaderboard.empty:
    leaderboard = leaderboard.sort_values(
        by=["F1", "MCC"],
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# 13. DASHBOARD TABS
# ============================================================

st.divider()

tab_leaderboard, tab_confusion, tab_roc, tab_download = (
    st.tabs([
        "🏆 Leaderboard",
        "🧩 Confusion Matrices",
        "📈 ROC Analysis",
        "📥 Predictions & Downloads",
    ])
)


# ============================================================
# 14. LEADERBOARD
# ============================================================

with tab_leaderboard:

    st.subheader("Model Performance Comparison")

    if leaderboard.empty:

        st.info(
            "Evaluation metrics are unavailable because "
            "the uploaded dataset does not contain "
            "actual Class labels."
        )

        st.write(
            "Predictions and timing information "
            "are still available."
        )

    else:

        best_model = leaderboard.iloc[0]

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Best Model",
            best_model["Model"],
        )

        m2.metric(
            "Best Macro F1",
            f"{best_model['F1']:.4f}",
        )

        m3.metric(
            "Best MCC",
            f"{best_model['MCC']:.4f}",
        )

        st.markdown("#### Six-Metric Comparison")

        st.dataframe(
            leaderboard[
                ["Model", *METRICS]
            ].style.format(
                {
                    metric: "{:.4f}"
                    for metric in METRICS
                },
                na_rep="N/A",
            ),
            width="stretch",
            hide_index=True,
        )

        st.markdown("#### Interactive Comparison")

        fig = leaderboard_chart(leaderboard)

        st.plotly_chart(
            fig,
            width="stretch",
        )

        st.download_button(
            label="Download Leaderboard CSV",
            data=leaderboard.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="model_leaderboard.csv",
            mime="text/csv",
        )

        if leaderboard["AUC"].isna().any():
            st.warning(
                "ROC-AUC is unavailable for one or more "
                "models because the dataset does not "
                "contain every class."
            )


# ============================================================
# 15. CONFUSION MATRICES
# ============================================================

with tab_confusion:

    st.subheader(
        "Confusion Matrices and Classification Reports"
    )

    if not has_labels:

        st.info(
            "Actual Class labels are required "
            "to generate confusion matrices."
        )

    else:

        normalize = st.toggle(
            "Normalize confusion matrices",
            value=False,
        )

        for model_name, result in results.items():

            with st.expander(
                f"{model_name} — Diagnostics",
                expanded=(
                    selected_model != "All Models"
                ),
            ):

                left, right = st.columns([1.1, 1])

                with left:

                    fig = confusion_matrix_figure(
                        matrix=result["confusion_matrix"],
                        classes=CLASSES,
                        model_name=model_name,
                        normalize=normalize,
                    )

                    st.pyplot(fig)
                    plt.close(fig)

                with right:

                    st.markdown(
                        "##### Classification Report"
                    )

                    report_df = pd.DataFrame(
                        result["classification_report"]
                    ).T

                    st.dataframe(
                        report_df,
                        width="stretch",
                    )

                    st.download_button(
                        label="Download Classification Report",
                        data=report_df.to_csv().encode(
                            "utf-8"
                        ),
                        file_name=(
                            safe_model_filename(model_name)
                            + "_classification_report.csv"
                        ),
                        mime="text/csv",
                        key=(
                            "report_"
                            + safe_model_filename(model_name)
                        ),
                    )


# ============================================================
# 16. ROC ANALYSIS
# ============================================================

with tab_roc:

    st.subheader("Multiclass ROC Analysis")

    if not has_labels:

        st.info(
            "ROC analysis requires actual Class labels."
        )

    else:

        st.caption(
            "One-vs-rest ROC analysis for the seven "
            "Dry Bean classes."
        )

        for model_name, result in results.items():

            with st.expander(
                f"{model_name} — ROC Curves",
                expanded=(
                    selected_model != "All Models"
                ),
            ):

                roc_data = result["roc_data"]

                if roc_data is None:

                    st.warning(
                        "ROC curves are unavailable because "
                        "the dataset does not contain all "
                        "required classes."
                    )
                    continue

                fig = roc_figure(
                    y_true=roc_data["labels"],
                    probabilities=roc_data["probabilities"],
                    classes=CLASSES,
                    model_name=model_name,
                )

                st.pyplot(fig)
                plt.close(fig)

                st.metric(
                    "Macro ROC-AUC",
                    f"{result['metrics']['AUC']:.4f}",
                )


# ============================================================
# 17. PREDICTIONS AND DOWNLOADS
# ============================================================

with tab_download:

    st.subheader("Prediction Explorer")

    prediction_tables = {}

    for model_name, result in results.items():

        prediction_tables[model_name] = (
            make_prediction_dataframe(
                original_df=input_df,
                result=result,
                classes=CLASSES,
                target=TARGET,
            )
        )

    if selected_model == "All Models":

        st.markdown("#### Combined Predictions")

        combined_df = input_df.copy()

        for model_name, result in results.items():

            column_name = (
                "Prediction_"
                + safe_model_filename(model_name)
            )

            combined_df[column_name] = result[
                "predictions"
            ]

        st.dataframe(
            combined_df.head(100),
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "Preview shows up to 100 records. "
            "Downloads contain all records."
        )

        st.download_button(
            label="Download Combined Predictions",
            data=combined_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="all_models_predictions.csv",
            mime="text/csv",
        )

    else:

        st.dataframe(
            prediction_tables[
                selected_model
            ].head(100),
            width="stretch",
            hide_index=True,
        )

    st.markdown("#### Individual Model Downloads")

    for model_name, prediction_df in (
        prediction_tables.items()
    ):

        st.download_button(
            label=f"Download {model_name} Predictions",
            data=prediction_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name=(
                safe_model_filename(model_name)
                + "_predictions.csv"
            ),
            mime="text/csv",
            key=(
                "prediction_"
                + safe_model_filename(model_name)
            ),
        )

    if selected_model == "All Models":

        zip_buffer = BytesIO()

        with ZipFile(
            zip_buffer,
            mode="w",
        ) as archive:

            for model_name, prediction_df in (
                prediction_tables.items()
            ):

                archive.writestr(
                    safe_model_filename(model_name)
                    + "_predictions.csv",
                    prediction_df.to_csv(
                        index=False
                    ),
                )

        st.download_button(
            label="Download All Predictions (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="dry_bean_predictions.zip",
            mime="application/zip",
        )


# ============================================================
# 18. FOOTER
# ============================================================

st.divider()

st.caption(
    "Dry Bean Classification | "
    "Five Saved ML Pipelines | "
    "Training-Free Inference | "
    "BITS M.Tech AIML — Assignment 2"
)
