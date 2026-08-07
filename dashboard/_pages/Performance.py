"""Model Performance page - train, compare, and evaluate models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st

from utils.charts import (
    confusion_matrix_chart,
    feature_importance_chart,
    model_comparison_chart,
    precision_recall_chart,
    roc_curve_chart,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "model"


def render() -> None:
    st.markdown('<p class="section-header">Model Performance</p>', unsafe_allow_html=True)

    st.subheader("Train Models")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Train Random Forest", use_container_width=True):
            with st.spinner("Training Random Forest..."):
                from utils.helper import api_request
                result = api_request("POST", "/api/v1/model/train", {"model_type": "random_forest"})
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.session_state["model_metrics"] = result.get("metrics", {})
                    st.success("Random Forest trained!")

    with col2:
        if st.button("Train XGBoost", use_container_width=True):
            with st.spinner("Training XGBoost..."):
                from utils.helper import api_request
                result = api_request("POST", "/api/v1/model/train", {"model_type": "xgboost"})
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.session_state["model_metrics"] = result.get("metrics", {})
                    st.success("XGBoost trained!")

    with col3:
        if st.button("Train Both & Save Best", type="primary", use_container_width=True):
            with st.spinner("Training both models..."):
                from utils.helper import api_request
                result = api_request("POST", "/api/v1/model/train", {"model_type": "both"})
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.session_state["model_metrics"] = result.get("metrics", {})
                    st.success(f"Best model saved: {result.get('best_model', 'Unknown')}")

    st.divider()

    # Load saved metrics
    metadata_path = MODEL_DIR / "model_metadata.json"
    metrics = st.session_state.get("model_metrics", {})

    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            saved = json.load(f)
        if not metrics:
            metrics = saved.get("metrics", {})
        st.info(f"Current best model: **{saved.get('best_model', 'N/A')}**")

    if not metrics:
        st.warning("No model metrics available. Train a model first using the buttons above or run `python training/train_model.py`.")
        return

    # Comparison chart
    st.plotly_chart(model_comparison_chart(metrics), use_container_width=True)

    # Metrics table
    st.subheader("Detailed Metrics")
    for model_name, model_metrics in metrics.items():
        with st.expander(model_name, expanded=True):
            mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
            mcol1.metric("Accuracy", f"{model_metrics.get('accuracy', 0):.3f}")
            mcol2.metric("Precision", f"{model_metrics.get('precision', 0):.3f}")
            mcol3.metric("Recall", f"{model_metrics.get('recall', 0):.3f}")
            mcol4.metric("F1 Score", f"{model_metrics.get('f1', 0):.3f}")
            mcol5.metric("ROC AUC", f"{model_metrics.get('roc_auc', 0):.3f}")

            cm = model_metrics.get("confusion_matrix")
            if cm:
                st.write("Confusion Matrix:")
                st.dataframe(cm)

    # Evaluation charts from saved test predictions
    test_pred_path = MODEL_DIR / "test_predictions.npz"
    if test_pred_path.exists():
        data = np.load(test_pred_path)
        y_true, y_pred, y_prob = data["y_true"], data["y_pred"], data["y_prob"]

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(roc_curve_chart(y_true, y_prob), use_container_width=True)
        with col_b:
            st.plotly_chart(precision_recall_chart(y_true, y_prob), use_container_width=True)

        st.plotly_chart(confusion_matrix_chart(y_true, y_pred), use_container_width=True)

    # Feature importance
    best_metrics = metrics.get(list(metrics.keys())[0], {})
    fi = best_metrics.get("feature_importances", {})
    if fi:
        st.plotly_chart(
            feature_importance_chart(list(fi.values()), list(fi.keys())),
            use_container_width=True,
        )
