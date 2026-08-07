"""Upload Dataset page - CSV upload, validation, cleaning."""

from __future__ import annotations

import streamlit as st

from utils.preprocessing import clean_dataset, validate_dataset


def render() -> None:
    st.markdown('<p class="section-header">Upload Dataset</p>', unsafe_allow_html=True)
    st.info("Upload a CSV file with employee attrition data. The system will validate and clean it automatically.")

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        with st.spinner("Processing uploaded file..."):
            try:
                import pandas as pd

                df = pd.read_csv(uploaded)
                report = validate_dataset(df)

                st.subheader("Dataset Preview")
                st.dataframe(df.head(20), use_container_width=True)

                st.subheader("Validation Report")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", report["rows"])
                with col2:
                    st.metric("Duplicates", report["duplicates"])
                with col3:
                    missing_total = sum(report["missing_values"].values())
                    st.metric("Missing Values", missing_total)

                if report["missing_columns"]:
                    st.error(f"Missing columns: {', '.join(report['missing_columns'])}")
                else:
                    st.success("All required columns present.")

                if report["missing_values"]:
                    st.warning("Missing values by column:")
                    st.json(report["missing_values"])

                if report["warnings"]:
                    for w in report["warnings"]:
                        st.warning(w)

                st.divider()
                st.subheader("Data Cleaning")

                remove_missing = st.checkbox("Remove rows with missing values", value=True)
                remove_duplicates = st.checkbox("Remove duplicate rows", value=True)

                if st.button("Apply Cleaning", type="primary"):
                    cleaned = clean_dataset(df, remove_missing, remove_duplicates)
                    st.session_state["dataset_df"] = cleaned
                    st.success(f"Dataset cleaned: {len(cleaned)} rows remaining.")
                    st.dataframe(cleaned.head(20), use_container_width=True)

                    csv_data = cleaned.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Clean Dataset",
                        data=csv_data,
                        file_name="cleaned_employee_attrition.csv",
                        mime="text/csv",
                    )

                if st.button("Use This Dataset", type="secondary"):
                    cleaned = clean_dataset(df, True, True)
                    st.session_state["dataset_df"] = cleaned
                    st.success("Dataset loaded into dashboard session.")

            except Exception as exc:
                st.error(f"Error processing file: {exc}")
