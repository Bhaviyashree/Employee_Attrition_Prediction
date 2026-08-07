"""Employee Records page with search, filter, and pagination."""

from __future__ import annotations

import streamlit as st

from utils.helper import filter_dataframe, load_default_dataset, paginate_dataframe, total_pages


def render() -> None:
    st.markdown('<p class="section-header">Employee Records</p>', unsafe_allow_html=True)

    with st.spinner("Loading employee records..."):
        df = load_default_dataset()

    if df.empty:
        st.warning("No employee data available.")
        return

    # Filters
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search = st.text_input("Search employees", placeholder="Search by any field...")
    with col2:
        departments = ["All"] + sorted(df["Department"].unique().tolist()) if "Department" in df.columns else ["All"]
        dept_filter = st.selectbox("Department", departments)
    with col3:
        attrition_opts = ["All", "Yes", "No"] if "Attrition" in df.columns else ["All"]
        attrition_filter = st.selectbox("Attrition", attrition_opts)

    filtered = filter_dataframe(df, search, dept_filter, attrition_filter)
    st.caption(f"Showing {len(filtered)} of {len(df)} employees")

    # Pagination
    page_size = st.selectbox("Rows per page", [10, 20, 50, 100], index=1)
    total_p = total_pages(len(filtered), page_size)
    page = st.number_input("Page", min_value=1, max_value=total_p, value=1)

    display_cols = [
        c for c in [
            "EmployeeNumber", "Age", "Gender", "Department", "JobRole",
            "MonthlyIncome", "YearsAtCompany", "JobSatisfaction", "OverTime", "Attrition",
        ]
        if c in filtered.columns
    ]

    page_df = paginate_dataframe(filtered[display_cols], int(page), page_size)
    st.dataframe(page_df, use_container_width=True, height=400)

    # Sync to database
    if st.button("Sync to Database via API"):
        from utils.helper import api_request

        result = api_request("POST", "/api/v1/employees/sync-from-dataset")
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(result.get("message", "Sync complete"))

    # Download
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download Filtered Records", csv, "employees.csv", "text/csv")
