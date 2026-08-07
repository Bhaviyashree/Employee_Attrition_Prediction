"""
Employee Attrition Prediction Dashboard - Main Entry Point.
Run: streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Add project root and dashboard dir to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DASHBOARD_ROOT))

from utils.helper import get_custom_css, init_session_state, load_default_dataset

# Page configuration
st.set_page_config(
    page_title="HR Attrition Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()

# Apply theme CSS
theme = st.session_state.get("theme", "dark")
st.markdown(get_custom_css(theme), unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/combo-chart.png",
        width=80,
    )
    st.title("HR Analytics")
    st.caption("Employee attrition insights and prediction dashboard")
    st.divider()

    menu_items = {
        "Dashboard": "Dashboard",
        "Upload Dataset": "Dataset",
        "Employee Records": "Employees",
        "Analytics": "Analytics",
        "Prediction": "Prediction",
        "Model Performance": "Performance",
        "Prediction History": "History",
        "Settings": "Settings",
        "About": "About",
    }

    current_label = st.radio(
        "Navigate",
        list(menu_items.keys()),
        index=list(menu_items.values()).index(st.session_state.get("current_page", "Dashboard")),
        key="sidebar_navigation",
    )
    st.session_state["current_page"] = menu_items[current_label]

    st.divider()
    dark_mode = st.toggle(
        "Dark Mode",
        value=st.session_state.get("theme", "dark") == "dark",
        key="dark_mode_toggle",
    )
    st.session_state["theme"] = "dark" if dark_mode else "light"

# Pre-load dataset
load_default_dataset()

# Route to selected page
current = st.session_state.get("current_page", "Dashboard")

page_modules = {
    "Dashboard": "_pages.Dashboard",
    "Dataset": "_pages.Dataset",
    "Employees": "_pages.Employees",
    "Analytics": "_pages.Analytics",
    "Prediction": "_pages.Prediction",
    "Performance": "_pages.Performance",
    "History": "_pages.History",
    "Settings": "_pages.Settings",
    "About": "_pages.About",
}

import importlib

module_name = page_modules.get(current, "_pages.Dashboard")
page_module = importlib.import_module(module_name)
page_module.render()
