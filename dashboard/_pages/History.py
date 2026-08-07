"""Prediction History page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.helper import api_request, dataframe_to_excel


def render() -> None:
    st.markdown('<p class="section-header">Prediction History</p>', unsafe_allow_html=True)

    with st.spinner("Loading prediction history..."):
        result = api_request("GET", "/api/v1/predictions/history?limit=200")

    if "error" in result:
        st.error(f"Could not load history: {result['error']}")
        st.info("Start the API server: `uvicorn api.main:app --reload`")
        return

    records = result.get("data", [])
    if not records:
        st.info("No predictions recorded yet. Make a prediction on the Prediction page.")
        return

    df = pd.DataFrame(records)
    st.caption(f"Total predictions: {result.get('total', len(records))}")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        dept_filter = st.selectbox(
            "Filter by Department",
            ["All"] + sorted(df["department"].dropna().unique().tolist()),
        )
    with col2:
        pred_filter = st.selectbox("Filter by Result", ["All", "Yes", "No"])

    filtered = df.copy()
    if dept_filter != "All":
        filtered = filtered[filtered["department"] == dept_filter]
    if pred_filter != "All":
        filtered = filtered[filtered["prediction_result"] == pred_filter]

    st.dataframe(filtered, use_container_width=True, height=400)

    # Download as Excel
    excel_bytes = dataframe_to_excel(filtered)
    st.download_button(
        "Download Predictions as Excel",
        data=excel_bytes,
        file_name="prediction_history.xlsx",
        mime="application/vnd.openxmlformats-officedocument-spreadsheetml.sheet",
    )

    # Summary
    st.subheader("Summary")
    yes_count = (filtered["prediction_result"] == "Yes").sum()
    st.metric("High-Risk Predictions", int(yes_count))
