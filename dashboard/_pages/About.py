"""About page."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.markdown('<p class="section-header">About</p>', unsafe_allow_html=True)

    st.markdown("""
    ## Employee Attrition Prediction Dashboard

    A production-ready HR Analytics platform for predicting employee attrition
    using machine learning.

    ### Technology Stack

    | Layer | Technologies |
    |-------|-------------|
    | **Frontend** | Streamlit, Plotly, Custom CSS |
    | **Backend** | FastAPI, Pydantic, Uvicorn |
    | **ML** | Random Forest, XGBoost, Scikit-learn |
    | **Database** | PostgreSQL, SQLAlchemy |

    ### Features

    - Interactive HR dashboard with KPI cards and charts
    - Real-time attrition prediction with risk scoring
    - CSV upload with automatic validation and cleaning
    - Model training, comparison, and performance metrics
    - Prediction history with Excel export
    - Light/Dark theme toggle
    - Dashboard export to HTML/PDF

    ### Models

    - **Random Forest Classifier** — ensemble tree-based model
    - **XGBoost Classifier** — gradient boosted trees
    - Best model selected by ROC AUC score

    ### Project Structure

    ```
    Employee_Attrition_Prediction/
    ├── dataset/          # CSV data files
    ├── model/            # Trained models & artifacts
    ├── training/         # Model training scripts
    ├── api/              # FastAPI backend
    ├── dashboard/        # Streamlit frontend
    ├── database/         # SQLAlchemy models
    └── utils/            # Shared utilities
    ```

    ### Authors

    Built as a modern HR analytics application with a focus on actionable insights, clear reporting, and easy local deployment.

    **Version:** 1.0.0
    """)
