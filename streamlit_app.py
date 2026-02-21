from pathlib import Path
import logging
import json
import hmac

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import REPORTS_DIR, TARGET_COLUMN
from src.data.load_data import load_raw_metabric_data
from src.data.preprocess import clean_metabric_df, get_feature_columns
from src.modeling.predict import predict_from_dataframe
from src.viz.plots import (
    plot_confusion_matrix_heatmap,
    plot_feature_importance,
    plot_metrics_bar,
    plot_pca_scatter,
)

# Logging
LOG_DIR = Path("artifacts/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "app.log"),
        ],
    )

st.set_page_config(
    page_title="Molecular Subtype Classifier",
    page_icon="🧬",
    layout="wide",
)


# Auth for Part C security requirement - uses Streamlit secrets for the password
def require_login() -> None:

    if st.session_state.get("authenticated", False):
        return

    st.sidebar.subheader("Login")
    pw = st.sidebar.text_input("Password", type="password")

    correct = st.secrets.get("APP_PASSWORD", None)
    if correct is None:
        st.sidebar.error("APP_PASSWORD is not set in Streamlit secrets.")
        st.stop()

    if pw:
        if hmac.compare_digest(pw, str(correct)):
            st.session_state["authenticated"] = True
            logger.info("User authenticated successfully.")
            st.sidebar.success("Logged in")
            st.rerun()
        else:
            logger.warning("Failed login attempt.")
            st.sidebar.error("Incorrect password")

    st.warning("Please log in to use the dashboard.")
    st.stop()


def _metrics_path():
    return REPORTS_DIR / "metrics.json"


def _feature_importance_path(model_name: str = "random_forest"):
    return REPORTS_DIR / f"feature_importance_{model_name}.csv"


@st.cache_data
def load_dataset_summary():
    df = load_raw_metabric_data()
    shape = df.shape
    subtype_counts = (
        df[TARGET_COLUMN]
        .dropna()
        .astype(str)
        .value_counts()
        .rename_axis("subtype")
        .reset_index(name="count")
    )
    return shape, subtype_counts


@st.cache_data
def load_clean_dataset():
    df = load_raw_metabric_data()
    return clean_metabric_df(df, target_column=TARGET_COLUMN)


@st.cache_data
def load_metrics():
    path = _metrics_path()
    with open(path, "r") as f:
        return json.load(f)


@st.cache_data
def load_confusion_matrix(model_name: str):
    metrics = load_metrics()
    cm_data = metrics[model_name].get("confusion_matrix_labeled")
    if not cm_data:
        matrix = metrics[model_name]["confusion_matrix"]
        labels = [str(i) for i in range(len(matrix))]
        return pd.DataFrame(matrix, index=labels, columns=labels)

    labels = cm_data["labels"]
    matrix = cm_data["matrix"]
    return pd.DataFrame(matrix, index=labels, columns=labels)


@st.cache_data
def load_feature_importance_csv(model_name: str = "random_forest"):
    return pd.read_csv(_feature_importance_path(model_name))


# Descriptive PCA visualization on the cleaned dataset (not part of training)
@st.cache_data
def build_pca_df():

    df = load_raw_metabric_data()
    df = clean_metabric_df(df, target_column=TARGET_COLUMN)

    feature_cols = get_feature_columns(df, target_column=TARGET_COLUMN)
    X_df = df[feature_cols].copy()

    # Coerce, sanitize, and bound values to keep PCA numerically stable.
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    medians = X_df.median(numeric_only=True).fillna(0.0)
    X_df = X_df.fillna(medians).fillna(0.0).astype("float64")
    X_df = X_df.clip(lower=-1e6, upper=1e6)

    X_scaled = StandardScaler().fit_transform(X_df.values)
    X_2d = PCA(n_components=2, svd_solver="full").fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        {
            "PC1": X_2d[:, 0],
            "PC2": X_2d[:, 1],
            "subtype": df[TARGET_COLUMN].astype(str).values,
        }
    )
    return pca_df


def _format_float(x):
    if x is None:
        return None
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def main():
    require_login()

    st.sidebar.divider()
    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        logger.info("User logged out.")
        st.rerun()

    st.sidebar.subheader("Monitoring")
    st.sidebar.write(f"metrics.json present: {_metrics_path().exists()}")
    st.sidebar.write(
        f"RF importance present: {_feature_importance_path('random_forest').exists()}"
    )
    st.sidebar.write(
        f"LR importance present: {_feature_importance_path('logistic_regression').exists()}"
    )

    st.title("Breast Cancer Molecular Subtype Classification")
    st.caption(
        "METABRIC PAM50 subtype classification (Logistic Regression + Random Forest)"
    )

    # Project Overview
    st.markdown(
        """
This dashboard evaluates supervised machine learning models for classifying breast cancer tumors
into **PAM50 molecular subtypes** using **gene-expression + clinical features** from the **METABRIC** dataset.

It includes:
- dataset summary + subtype distribution,
- model metrics and confusion matrix evaluation,
- descriptive PCA visualization,
- explainability via feature importance,
- and a decision-support prediction workflow (CSV upload).
"""
    )

    # Sidebar
    st.sidebar.header("Options")

    model_for_cm = st.sidebar.selectbox(
        "Confusion matrix model",
        options=["random_forest", "logistic_regression"],
        index=0,
        help="Controls which model's confusion matrix is displayed in the Evaluation Visuals section.",
    )

    feature_model = st.sidebar.selectbox(
        "Feature importance model",
        options=["random_forest", "logistic_regression"],
        index=0,
        help="Controls which feature-importance CSV and plot are shown in Explainability & Visualization.",
    )

    top_n = st.sidebar.slider(
        "Top features to display",
        10,
        50,
        20,
        step=5,
        help="Controls the number of features displayed in the feature-importance table and plot.",
    )

    logger.info(
        "User options: confusion_matrix_model=%s feature_model=%s top_n=%s",
        model_for_cm,
        feature_model,
        top_n,
    )

    # Dataset
    st.header("Dataset Summary")
    try:
        shape, subtype_counts = load_dataset_summary()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Rows", shape[0])
            st.metric("Columns (raw)", shape[1])
        with c2:
            st.subheader("Subtype distribution (raw)")
            # show 1-based display index
            st.dataframe(
                subtype_counts.reset_index(drop=True)
                .rename_axis(" ")
                .reset_index()
                .rename(columns={" ": "#"}),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Note: The 'NC' subtype (n=6) is excluded during preprocessing due to extremely low sample size."
            )
    except Exception as e:
        logger.exception("Could not load raw dataset summary.")
        st.warning(f"Could not load raw dataset summary. Error: {e}")

    st.divider()

    # Reports availability check
    if not _metrics_path().exists():
        st.error(
            "No evaluation artifacts found in artifacts/reports/.\n\n"
            "Run this from the project root to generate them:\n"
            "```bash\n"
            "python3 -m src.main all --explain\n"
            "```\n"
        )
        st.stop()

    # Model metrics table
    st.header("Model Metrics")
    st.caption("Model evaluation using test set.")

    results = load_metrics()

    rows = []
    for model_name, m in results.items():
        rows.append(
            {
                "model": model_name,
                "accuracy": m.get("accuracy"),
                "macro_f1": m.get("macro_f1"),
                "weighted_f1": m.get("weighted_f1"),
            }
        )
    metrics_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)

    # Best model highlights
    if not metrics_df.empty:
        best = metrics_df.iloc[0].to_dict()
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Model", str(best.get("model")))
        col2.metric("Accuracy", _format_float(best.get("accuracy")))
        col3.metric("Macro F1", _format_float(best.get("macro_f1")))

    # 1-based display index for table
    metrics_display = metrics_df.reset_index(drop=True).copy()
    metrics_display.insert(0, "#", range(1, len(metrics_display) + 1))
    st.dataframe(metrics_display, use_container_width=True, hide_index=True)

    st.divider()

    # Interactive queries
    st.header("Interactive Queries")
    st.caption("Filter and inspect dataset slices, model metrics, and feature scores.")

    tab1, tab2, tab3 = st.tabs(["Dataset Query", "Metrics Query", "Feature Query"])

    with tab1:
        try:
            df_clean = load_clean_dataset()
            subtypes = sorted(df_clean[TARGET_COLUMN].astype(str).unique().tolist())
            selected_subtypes = st.multiselect(
                "Subtypes",
                options=subtypes,
                default=subtypes,
                help="Filter rows by one or more molecular subtypes.",
            )

            numeric_cols = get_feature_columns(df_clean, target_column=TARGET_COLUMN)
            query_col = st.selectbox(
                "Numeric feature",
                options=numeric_cols,
                index=0,
                help="Choose one numeric field to query by value range.",
            )

            series = pd.to_numeric(df_clean[query_col], errors="coerce")
            series = series.replace([np.inf, -np.inf], np.nan).dropna()

            if series.empty:
                st.warning("Selected feature has no finite numeric values to query.")
            else:
                min_val = float(series.min())
                max_val = float(series.max())
                value_min, value_max = st.slider(
                    "Value range",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                )

                qdf = df_clean[
                    df_clean[TARGET_COLUMN].astype(str).isin(selected_subtypes)
                ]
                qseries = pd.to_numeric(qdf[query_col], errors="coerce")
                qdf = qdf[qseries.between(value_min, value_max, inclusive="both")]

                st.metric("Matching rows", int(len(qdf)))
                display_cols = [TARGET_COLUMN, query_col]
                st.dataframe(
                    qdf[display_cols].reset_index(drop=True).head(250),
                    use_container_width=True,
                    hide_index=True,
                )
        except Exception as e:
            logger.exception("Dataset query failed.")
            st.warning(f"Dataset query unavailable. Error: {e}")

    with tab2:
        metric_col = st.selectbox(
            "Metric",
            options=["accuracy", "macro_f1", "weighted_f1"],
            index=0,
        )
        min_metric = st.slider(
            "Minimum score",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.01,
        )
        qmetrics = metrics_df[metrics_df[metric_col] >= min_metric].copy()
        qmetrics = qmetrics.sort_values(metric_col, ascending=False)
        st.metric("Matching models", int(len(qmetrics)))
        st.dataframe(
            qmetrics.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

    with tab3:
        try:
            fi_df_query = load_feature_importance_csv(feature_model)
            score_col = (
                "importance" if "importance" in fi_df_query.columns else "mean_abs_coef"
            )
            feature_term = st.text_input(
                "Feature name contains",
                value="",
                help="Case-insensitive substring filter for feature names.",
            ).strip()
            min_score = st.number_input(
                "Minimum score",
                min_value=0.0,
                value=0.0,
                step=0.0001,
                format="%.4f",
            )

            qfi = fi_df_query[fi_df_query[score_col] >= min_score].copy()
            if feature_term:
                qfi = qfi[
                    qfi["feature"].str.contains(feature_term, case=False, na=False)
                ]
            qfi = qfi.sort_values(score_col, ascending=False)

            st.metric("Matching features", int(len(qfi)))
            st.dataframe(
                qfi.reset_index(drop=True).head(250),
                use_container_width=True,
                hide_index=True,
            )
        except Exception as e:
            logger.exception("Feature query failed.")
            st.warning(f"Feature query unavailable. Error: {e}")

    st.divider()

    # Metrics chart + Confusion Matrix heatmap
    st.header("Evaluation Visuals")
    try:
        metrics_fig = plot_metrics_bar(metrics_df)
        cm_df = load_confusion_matrix(model_for_cm)
        cm_fig = plot_confusion_matrix_heatmap(cm_df)

        col1, col2 = st.columns([1, 1.25])

        with col1:
            st.subheader("Metrics comparison")
            st.pyplot(metrics_fig, use_container_width=True)

        with col2:
            st.subheader(f"Confusion Matrix Heatmap ({model_for_cm})")
            st.pyplot(cm_fig, use_container_width=True)

        st.caption("Confusion matrix: rows = true labels, columns = predicted labels.")

        with st.expander("View confusion matrix table"):
            # 1-based display index
            cm_display = cm_df.copy()
            cm_display.insert(0, "true_label", cm_display.index)
            cm_display = cm_display.reset_index(drop=True)
            cm_display.insert(0, "#", range(1, len(cm_display) + 1))
            st.dataframe(cm_display, use_container_width=True, hide_index=True)

    except Exception as e:
        logger.exception("Could not render evaluation visuals.")
        st.warning(f"Could not render evaluation visuals. Error: {e}")

    st.divider()

    # PCA + Feature Importance
    st.header("Explainability & Visualization")
    st.caption(
        "Left: descriptive PCA projection (not used for training). "
        "Right: feature importance for the selected model."
    )

    try:
        pca_df = build_pca_df()
        pca_fig = plot_pca_scatter(pca_df, label_col="subtype")

        st.caption(
            "Note: PCA is used for visualization only and is not part of the training pipeline."
        )

        fi_df = load_feature_importance_csv(feature_model)
        fi_fig = plot_feature_importance(fi_df, top_n=top_n)

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.subheader("PCA (2D) by subtype")
            st.pyplot(pca_fig, use_container_width=True)

        with col2:
            st.subheader(f"Top {top_n} features ({feature_model})")
            st.pyplot(fi_fig, use_container_width=True)

            if feature_model == "random_forest":
                st.caption(
                    "Feature importance reflects mean decrease in impurity (Random Forest)."
                )
            else:
                st.caption(
                    "Coefficients reflect feature weight magnitude (Logistic Regression, mean abs)."
                )

        with st.expander("View feature importance table"):
            fi_head = fi_df.head(top_n).reset_index(drop=True).copy()
            fi_head.insert(0, "#", range(1, len(fi_head) + 1))
            st.dataframe(fi_head, use_container_width=True, hide_index=True)

    except FileNotFoundError:
        st.warning(
            "Feature importance CSV not found.\n\n"
            "Generate it with:\n"
            "```bash\n"
            "python3 -m src.main eval --explain\n"
            "```\n"
        )
    except Exception as e:
        logger.exception("Could not render PCA / feature importance section.")
        st.warning(f"Could not render PCA / feature importance section. Error: {e}")

    st.divider()

    # Predictions (Decision Support)
    st.header("Make a Prediction (Decision Support)")
    st.caption("Upload a CSV with the same feature columns used during training.")

    uploaded = st.file_uploader(
        "Upload a CSV of samples (rows = samples)", type=["csv"]
    )
    model_for_pred = st.selectbox(
        "Prediction model",
        ["random_forest", "logistic_regression"],
        index=0,
    )

    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
            logger.info(
                "Prediction CSV uploaded: filename=%s rows=%s cols=%s",
                getattr(uploaded, "name", "unknown"),
                len(df_in),
                len(df_in.columns),
            )
            st.write("Preview of uploaded data:")
            preview = df_in.head().reset_index(drop=True).copy()
            preview.insert(0, "#", range(1, len(preview) + 1))
            st.dataframe(preview, use_container_width=True, hide_index=True)

            if st.button("Run prediction"):
                logger.info(
                    "Running prediction: model=%s rows=%s", model_for_pred, len(df_in)
                )
                preds = predict_from_dataframe(
                    df_in, model_name=model_for_pred, return_proba=True
                )

                st.subheader("Predictions")
                preds_display = preds.reset_index(drop=True).copy()
                preds_display.insert(0, "#", range(1, len(preds_display) + 1))
                st.dataframe(preds_display, use_container_width=True, hide_index=True)

                # Show probability bars if probability columns exist
                proba_cols = [
                    c for c in preds.columns if c.lower().startswith("proba_")
                ]
                if proba_cols:
                    st.subheader("Prediction Probabilities (first 10 rows)")
                    st.caption(
                        "These probabilities can help interpret model confidence."
                    )
                    st.bar_chart(
                        preds[proba_cols].head(10),
                        use_container_width=True,
                    )

                st.download_button(
                    "Download predictions as CSV",
                    data=preds.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )

        except Exception as e:
            logger.exception("Prediction failed.")
            st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
