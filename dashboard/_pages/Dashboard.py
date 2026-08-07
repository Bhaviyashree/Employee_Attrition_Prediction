"""Dashboard home page with KPIs and charts."""

from __future__ import annotations

import streamlit as st

from utils.charts import (
    age_distribution_chart,
    business_travel_chart,
    correlation_heatmap,
    department_attrition_chart,
    education_level_chart,
    gender_distribution_chart,
    job_satisfaction_chart,
    monthly_income_box_chart,
    overtime_attrition_chart,
    salary_distribution_chart,
)
from utils.helper import load_default_dataset, render_kpi_card
from utils.preprocessing import get_kpi_metrics


def render() -> None:
    st.markdown('<p class="section-header">Dashboard Overview</p>', unsafe_allow_html=True)

    with st.spinner("Loading dashboard data..."):
        df = load_default_dataset()

    if df.empty:
        st.warning("No dataset loaded. Upload a CSV from the Upload Dataset page.")
        return

    kpis = get_kpi_metrics(df)

    st.subheader("Executive summary")
    st.markdown(
        f"""
        - **Total employees:** {kpis['total_employees']:,}
        - **Current employees:** {kpis['current_employees']:,}
        - **Attrition rate:** {kpis['attrition_percentage']}%
        - **Top risk department:** {kpis.get('highest_attrition_department', 'N/A')}
        - **Top risk role:** {kpis.get('highest_attrition_job_role', 'N/A')}
        """
    )
    st.caption("Use these insights to guide retention and workforce planning decisions.")

    # KPI Cards Row 1
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_kpi_card("Total Employees", f"{kpis['total_employees']:,}"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_kpi_card("Current Employees", f"{kpis['current_employees']:,}", "kpi-card kpi-card-alt"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_kpi_card("Attrition Count", f"{kpis['attrition_count']:,}", "kpi-card kpi-card-warn"), unsafe_allow_html=True)
    with c4:
        st.markdown(render_kpi_card("Attrition %", f"{kpis['attrition_percentage']}%", "kpi-card kpi-card-warn"), unsafe_allow_html=True)

    # KPI Cards Row 2
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.markdown(render_kpi_card("Avg Salary", f"${kpis['average_salary']:,.0f}", "kpi-card kpi-card-info"), unsafe_allow_html=True)
    with c6:
        st.markdown(render_kpi_card("Avg Experience", f"{kpis['average_experience']} yrs", "kpi-card kpi-card-info"), unsafe_allow_html=True)
    with c7:
        st.markdown(render_kpi_card("Avg Job Satisfaction", f"{kpis['average_job_satisfaction']}/4", "kpi-card"), unsafe_allow_html=True)
    with c8:
        st.markdown(render_kpi_card("Average Age", f"{kpis['average_age']} yrs", "kpi-card"), unsafe_allow_html=True)

    # High-impact HR insights
    c9, c10, c11, c12 = st.columns(4)
    with c9:
        st.markdown(render_kpi_card("High Attrition Dept", kpis.get("highest_attrition_department", "N/A"), "kpi-card kpi-card-warn"), unsafe_allow_html=True)
    with c10:
        st.markdown(render_kpi_card("High Attrition Role", kpis.get("highest_attrition_job_role", "N/A"), "kpi-card kpi-card-warn"), unsafe_allow_html=True)
    with c11:
        st.markdown(render_kpi_card("Overtime Attrition", f"{kpis.get('overtime_attrition_rate', 0.0):.1f}%", "kpi-card kpi-card-alt"), unsafe_allow_html=True)
    with c12:
        st.markdown(render_kpi_card("Low Salary Attrition", f"{kpis.get('low_salary_attrition_rate', 0.0):.1f}%", "kpi-card kpi-card-alt"), unsafe_allow_html=True)

    st.divider()

    # Charts Row 1
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(department_attrition_chart(df), use_container_width=True)
    with col2:
        st.plotly_chart(gender_distribution_chart(df), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(age_distribution_chart(df), use_container_width=True)
    with col4:
        st.plotly_chart(salary_distribution_chart(df), use_container_width=True)

    # Charts Row 2
    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(job_satisfaction_chart(df), use_container_width=True)
    with col6:
        st.plotly_chart(overtime_attrition_chart(df), use_container_width=True)

    col7, col8 = st.columns(2)
    with col7:
        st.plotly_chart(education_level_chart(df), use_container_width=True)
    with col8:
        st.plotly_chart(business_travel_chart(df), use_container_width=True)

    st.plotly_chart(monthly_income_box_chart(df), use_container_width=True)
    st.plotly_chart(correlation_heatmap(df), use_container_width=True)

    # Export PDF
    st.divider()
    from utils.helper import export_summary_to_pdf_html

    pdf_html = export_summary_to_pdf_html(kpis)
    st.download_button(
        "Export Dashboard Summary",
        data=pdf_html,
        file_name="hr_dashboard_report.html",
        mime="text/html",
    )
    st.caption("Download a ready-to-share summary report for stakeholder review.")
