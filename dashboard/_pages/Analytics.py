"""Analytics page with interactive filters and advanced charts."""

from __future__ import annotations

import streamlit as st

from utils.charts import (
    experience_analysis_chart,
    monthly_trend_chart,
    promotion_analysis_chart,
    salary_group_analysis,
    top_departments_attrition,
    travel_frequency_chart,
)
from utils.helper import load_default_dataset


def render() -> None:
    st.markdown('<p class="section-header">Analytics</p>', unsafe_allow_html=True)

    df = load_default_dataset()
    if df.empty:
        st.warning("No data available for analytics.")
        return

    st.subheader("Interactive Filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        if "Department" in df.columns:
            dept_options = ["All"] + sorted(df["Department"].unique().tolist())
            selected_dept = st.multiselect("Filter by Department", dept_options, default=["All"])
        else:
            selected_dept = ["All"]

    with col2:
        if "Gender" in df.columns:
            gender_options = ["All"] + sorted(df["Gender"].unique().tolist())
            selected_gender = st.selectbox("Filter by Gender", gender_options)
        else:
            selected_gender = "All"

    with col3:
        if "Age" in df.columns:
            age_range = st.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))
        else:
            age_range = (18, 65)

    # Apply filters
    filtered = df.copy()
    if "All" not in selected_dept and "Department" in filtered.columns:
        filtered = filtered[filtered["Department"].isin(selected_dept)]
    if selected_gender != "All" and "Gender" in filtered.columns:
        filtered = filtered[filtered["Gender"] == selected_gender]
    if "Age" in filtered.columns:
        filtered = filtered[(filtered["Age"] >= age_range[0]) & (filtered["Age"] <= age_range[1])]

    st.caption(f"Analyzing {len(filtered)} employees")

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(top_departments_attrition(filtered), use_container_width=True)
    with col_b:
        st.plotly_chart(salary_group_analysis(filtered), use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(experience_analysis_chart(filtered), use_container_width=True)
    with col_d:
        st.plotly_chart(promotion_analysis_chart(filtered), use_container_width=True)

    col_e, col_f = st.columns(2)
    with col_e:
        st.plotly_chart(travel_frequency_chart(filtered), use_container_width=True)
    with col_f:
        st.plotly_chart(monthly_trend_chart(filtered), use_container_width=True)

    # Summary stats
    st.subheader("Summary Statistics")
    if "Attrition" in filtered.columns:
        attr_rate = (filtered["Attrition"].astype(str).str.lower().isin(["yes", "1"])).mean() * 100
        st.metric("Filtered Attrition Rate", f"{attr_rate:.1f}%")
