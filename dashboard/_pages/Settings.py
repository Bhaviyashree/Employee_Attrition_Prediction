"""Settings page."""

from __future__ import annotations

import streamlit as st

from utils.helper import API_BASE_URL


def render() -> None:
    st.markdown('<p class="section-header">Settings</p>', unsafe_allow_html=True)

    st.subheader("API Configuration")
    api_url = st.text_input("API Base URL", value=st.session_state.get("api_url", API_BASE_URL))
    if st.button("Save API URL"):
        st.session_state["api_url"] = api_url.rstrip("/")
        st.success("API URL updated.")

    st.subheader("Theme")
    theme = st.radio("Select Theme", ["dark", "light"], index=0 if st.session_state.get("theme") == "dark" else 1, horizontal=True)
    st.session_state["theme"] = theme

    st.subheader("Database")
    st.code("DATABASE_URL=postgresql://postgres:postgres@localhost:5432/employee_attrition", language="bash")
    st.caption("Set this environment variable for PostgreSQL. Falls back to SQLite if unavailable.")

    st.subheader("Data Management")
    if st.button("Reset to Default Dataset"):
        st.session_state["dataset_df"] = None
        from utils.helper import load_default_dataset
        load_default_dataset()
        st.success("Default dataset reloaded.")

    if st.button("Clear Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Session cleared. Refresh the page.")
