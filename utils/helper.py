"""
Helper utilities: CSS themes, PDF export, Excel download, pagination, API client.
"""

from __future__ import annotations

import io
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_BASE_URL = os.getenv("ATTRITION_API_URL", "http://localhost:8000")


def get_custom_css(theme: str = "dark") -> str:
    """Return injected Streamlit CSS for HR dashboard styling."""
    is_dark = theme == "dark"
    bg = "#0F172A" if is_dark else "#F8FAFC"
    card = "#1E293B" if is_dark else "#FFFFFF"
    text = "#F1F5F9" if is_dark else "#0F172A"
    border = "#334155" if is_dark else "#E2E8F0"

    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background: linear-gradient(135deg, {bg} 0%, {"#1a1f3c" if is_dark else "#EEF2FF"} 100%);
    }}

    [data-testid="stSidebar"] {{
        background: {card};
        border-right: 1px solid {border};
    }}

    .kpi-card {{
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #06B6D4 100%);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        color: white;
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.25);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 0.5rem;
    }}

    .kpi-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 14px 30px rgba(99, 102, 241, 0.35);
    }}

    .kpi-card-alt {{
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
    }}

    .kpi-card-warn {{
        background: linear-gradient(135deg, #F59E0B 0%, #EF4444 100%);
    }}

    .kpi-card-info {{
        background: linear-gradient(135deg, #06B6D4 0%, #3B82F6 100%);
    }}

    .kpi-title {{
        font-size: 0.85rem;
        opacity: 0.9;
        font-weight: 500;
        margin-bottom: 0.25rem;
    }}

    .kpi-value {{
        font-size: 1.75rem;
        font-weight: 700;
    }}

    .risk-low {{
        background: #10B981;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
    }}

    .risk-medium {{
        background: #F59E0B;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
    }}

    .risk-high {{
        background: #EF4444;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
    }}

    .section-header {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {text};
        margin: 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #6366F1;
    }}

    div[data-testid="stMetric"] {{
        background: {card};
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid {border};
    }}

    .stButton > button {{
        background: linear-gradient(90deg, #6366F1, #8B5CF6);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: opacity 0.2s;
    }}

    .stButton > button:hover {{
        opacity: 0.9;
        color: white;
    }}
    </style>
    """


def render_kpi_card(title: str, value: str | float, css_class: str = "kpi-card") -> str:
    """HTML snippet for gradient KPI card."""
    return f"""
    <div class="{css_class}">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """


def paginate_dataframe(df: pd.DataFrame, page: int, page_size: int = 20) -> pd.DataFrame:
    """Return a slice of dataframe for current page."""
    start = (page - 1) * page_size
    end = start + page_size
    return df.iloc[start:end]


def total_pages(total_rows: int, page_size: int = 20) -> int:
    """Calculate total pagination pages."""
    return max(1, (total_rows + page_size - 1) // page_size)


def filter_dataframe(
    df: pd.DataFrame,
    search: str = "",
    department: str | None = None,
    attrition: str | None = None,
) -> pd.DataFrame:
    """Filter employee records by search and dropdown filters."""
    filtered = df.copy()

    if search:
        mask = filtered.astype(str).apply(
            lambda row: row.str.contains(search, case=False, na=False).any(),
            axis=1,
        )
        filtered = filtered[mask]

    if department and department != "All" and "Department" in filtered.columns:
        filtered = filtered[filtered["Department"] == department]

    if attrition and attrition != "All" and "Attrition" in filtered.columns:
        filtered = filtered[filtered["Attrition"] == attrition]

    return filtered


def dataframe_to_excel(df: pd.DataFrame) -> bytes:
    """Convert dataframe to Excel bytes for download."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Predictions")
    return buffer.getvalue()


def export_summary_to_pdf_html(kpis: dict[str, Any], title: str = "HR Dashboard Report") -> str:
    """Generate HTML report suitable for browser print-to-PDF."""
    kpi_rows = "".join(
        f"<tr><td>{k.replace('_', ' ').title()}</td><td>{v}</td></tr>"
        for k, v in kpis.items()
    )
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 40px; }}
            h1 {{ color: #6366F1; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background: #6366F1; color: white; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <h2>Key Performance Indicators</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            {kpi_rows}
        </table>
    </body>
    </html>
    """


def init_session_state() -> None:
    """Initialize Streamlit session state defaults."""
    defaults = {
        "theme": "dark",
        "dataset_df": None,
        "current_page": "Dashboard",
        "api_url": API_BASE_URL,
        "model_metrics": {},
        "last_prediction": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def api_request(
    method: str,
    endpoint: str,
    json_data: dict | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Make HTTP request to FastAPI backend."""
    url = f"{st.session_state.get('api_url', API_BASE_URL)}{endpoint}"
    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=timeout)
        elif method.upper() == "POST":
            response = requests.post(url, json=json_data, timeout=timeout)
        else:
            raise ValueError(f"Unsupported method: {method}")

        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        logger.warning("API unavailable at %s", url)
        return {"error": "API server is not running. Start with: uvicorn api.main:app --reload"}
    except requests.exceptions.HTTPError as exc:
        logger.exception("API HTTP error")
        try:
            detail = exc.response.json()
        except Exception:
            detail = {"detail": str(exc)}
        return {"error": detail.get("detail", str(exc))}
    except Exception as exc:
        logger.exception("API request failed")
        return {"error": str(exc)}


def load_default_dataset() -> pd.DataFrame:
    """Load default CSV into session if not already loaded."""
    if st.session_state.get("dataset_df") is not None:
        return st.session_state["dataset_df"]

    from utils.preprocessing import DEFAULT_DATASET_PATH, clean_dataset, load_dataset

    try:
        df = clean_dataset(load_dataset(DEFAULT_DATASET_PATH))
        st.session_state["dataset_df"] = df
        return df
    except Exception as exc:
        logger.exception("Failed to load default dataset")
        st.error(f"Could not load dataset: {exc}")
        return pd.DataFrame()


def get_risk_badge_html(risk_level: str) -> str:
    """Return styled HTML badge for risk level."""
    css_map = {
        "Low Risk": "risk-low",
        "Medium Risk": "risk-medium",
        "High Risk": "risk-high",
    }
    css = css_map.get(risk_level, "risk-low")
    return f'<span class="{css}">{risk_level}</span>'
