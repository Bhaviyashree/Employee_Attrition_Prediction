"""
Plotly chart builders for the HR Analytics Dashboard.
All charts use a consistent dark-theme color palette.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

# Professional HR dashboard palette
COLORS = {
    "primary": "#6366F1",
    "secondary": "#8B5CF6",
    "accent": "#06B6D4",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "background": "#0F172A",
    "card": "#1E293B",
    "text": "#F1F5F9",
    "muted": "#94A3B8",
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"], family="Inter, sans-serif"),
        colorway=[
            COLORS["primary"],
            COLORS["secondary"],
            COLORS["accent"],
            COLORS["success"],
            COLORS["warning"],
            COLORS["danger"],
        ],
        margin=dict(l=40, r=40, t=50, b=40),
    )
)


def _apply_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply consistent dark theme styling."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=COLORS["text"])),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#334155", zerolinecolor="#334155")
    fig.update_yaxes(gridcolor="#334155", zerolinecolor="#334155")
    return fig


def department_attrition_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart of attrition rate by department."""
    if "Department" not in df.columns or "Attrition" not in df.columns:
        return go.Figure()

    agg = (
        df.groupby("Department")["Attrition"]
        .apply(lambda x: (x.astype(str).str.lower().isin(["yes", "1"])).mean() * 100)
        .reset_index(name="AttritionRate")
        .sort_values("AttritionRate", ascending=False)
    )
    fig = px.bar(
        agg,
        x="Department",
        y="AttritionRate",
        color="AttritionRate",
        color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
        labels={"AttritionRate": "Attrition Rate (%)"},
    )
    return _apply_theme(fig, "Department-wise Attrition Rate")


def gender_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Pie chart of gender distribution."""
    if "Gender" not in df.columns:
        return go.Figure()
    counts = df["Gender"].value_counts().reset_index()
    counts.columns = ["Gender", "Count"]
    fig = px.pie(
        counts,
        names="Gender",
        values="Count",
        hole=0.45,
        color_discrete_sequence=[COLORS["primary"], COLORS["accent"], COLORS["secondary"]],
    )
    return _apply_theme(fig, "Gender Distribution")


def age_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Histogram of employee age."""
    if "Age" not in df.columns:
        return go.Figure()
    fig = px.histogram(
        df,
        x="Age",
        nbins=20,
        color_discrete_sequence=[COLORS["primary"]],
        opacity=0.85,
    )
    return _apply_theme(fig, "Age Distribution")


def salary_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Histogram of monthly income."""
    if "MonthlyIncome" not in df.columns:
        return go.Figure()
    fig = px.histogram(
        df,
        x="MonthlyIncome",
        nbins=30,
        color_discrete_sequence=[COLORS["secondary"]],
        opacity=0.85,
    )
    return _apply_theme(fig, "Salary Distribution")


def job_satisfaction_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart of job satisfaction levels."""
    if "JobSatisfaction" not in df.columns:
        return go.Figure()
    counts = df["JobSatisfaction"].value_counts().sort_index().reset_index()
    counts.columns = ["Level", "Count"]
    fig = px.bar(
        counts,
        x="Level",
        y="Count",
        color="Count",
        color_continuous_scale=[[0, COLORS["danger"]], [1, COLORS["success"]]],
    )
    return _apply_theme(fig, "Job Satisfaction Distribution")


def overtime_attrition_chart(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: overtime vs attrition."""
    if "OverTime" not in df.columns or "Attrition" not in df.columns:
        return go.Figure()
    cross = pd.crosstab(df["OverTime"], df["Attrition"])
    fig = go.Figure()
    for col in cross.columns:
        fig.add_trace(
            go.Bar(
                name=str(col),
                x=cross.index.astype(str),
                y=cross[col],
            )
        )
    fig.update_layout(barmode="group")
    return _apply_theme(fig, "Overtime vs Attrition")


def education_level_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart of education level distribution."""
    if "Education" not in df.columns:
        return go.Figure()
    counts = df["Education"].value_counts().sort_index().reset_index()
    counts.columns = ["Education", "Count"]
    fig = px.bar(
        counts,
        x="Education",
        y="Count",
        color_discrete_sequence=[COLORS["accent"]],
    )
    return _apply_theme(fig, "Education Level Distribution")


def business_travel_chart(df: pd.DataFrame) -> go.Figure:
    """Donut chart of business travel frequency."""
    if "BusinessTravel" not in df.columns:
        return go.Figure()
    counts = df["BusinessTravel"].value_counts().reset_index()
    counts.columns = ["Travel", "Count"]
    fig = px.pie(
        counts,
        names="Travel",
        values="Count",
        hole=0.5,
        color_discrete_sequence=[COLORS["primary"], COLORS["warning"], COLORS["success"]],
    )
    return _apply_theme(fig, "Business Travel Frequency")


def monthly_income_box_chart(df: pd.DataFrame) -> go.Figure:
    """Box plot of monthly income by department."""
    if "MonthlyIncome" not in df.columns or "Department" not in df.columns:
        return go.Figure()
    fig = px.box(
        df,
        x="Department",
        y="MonthlyIncome",
        color="Department",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    return _apply_theme(fig, "Monthly Income by Department")


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Correlation heatmap for numeric columns."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return go.Figure()
    corr = numeric.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
    )
    return _apply_theme(fig, "Feature Correlation Heatmap")


def feature_importance_chart(
    importances: list[float],
    feature_names: list[str],
    top_n: int = 15,
) -> go.Figure:
    """Horizontal bar chart of model feature importances."""
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:top_n]
    names = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker=dict(color=values, colorscale="Viridis"),
        )
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _apply_theme(fig, "Feature Importance")


def roc_curve_chart(y_true: np.ndarray, y_prob: np.ndarray) -> go.Figure:
    """ROC curve visualization."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC (AUC = {roc_auc:.3f})",
            line=dict(color=COLORS["primary"], width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(dash="dash", color=COLORS["muted"]),
        )
    )
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return _apply_theme(fig, "ROC Curve")


def confusion_matrix_chart(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale=[[0, COLORS["card"]], [1, COLORS["primary"]]],
        labels=dict(x="Predicted", y="Actual"),
        x=["No Attrition", "Attrition"],
        y=["No Attrition", "Attrition"],
    )
    return _apply_theme(fig, "Confusion Matrix")


def precision_recall_chart(y_true: np.ndarray, y_prob: np.ndarray) -> go.Figure:
    """Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"PR (AUC = {pr_auc:.3f})",
            line=dict(color=COLORS["accent"], width=3),
        )
    )
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    return _apply_theme(fig, "Precision-Recall Curve")


def top_departments_attrition(df: pd.DataFrame, top_n: int = 8) -> go.Figure:
    """Top departments by attrition count."""
    if "Department" not in df.columns or "Attrition" not in df.columns:
        return go.Figure()
    mask = df["Attrition"].astype(str).str.lower().isin(["yes", "1"])
    top = df.loc[mask].groupby("Department").size().sort_values(ascending=False).head(top_n)
    fig = px.bar(
        x=top.index,
        y=top.values,
        labels={"x": "Department", "y": "Attrition Count"},
        color=top.values,
        color_continuous_scale="Reds",
    )
    return _apply_theme(fig, "Top Departments with Attrition")


def salary_group_analysis(df: pd.DataFrame) -> go.Figure:
    """Attrition by salary quartile groups."""
    if "MonthlyIncome" not in df.columns:
        return go.Figure()
    temp = df.copy()
    temp["SalaryGroup"] = pd.qcut(temp["MonthlyIncome"], 4, labels=["Low", "Medium", "High", "Very High"])
    if "Attrition" not in temp.columns:
        return go.Figure()
    agg = (
        temp.groupby("SalaryGroup")["Attrition"]
        .apply(lambda x: (x.astype(str).str.lower().isin(["yes", "1"])).mean() * 100)
        .reset_index(name="Rate")
    )
    fig = px.bar(agg, x="SalaryGroup", y="Rate", color="Rate", color_continuous_scale="Blues")
    return _apply_theme(fig, "Attrition by Salary Group")


def experience_analysis_chart(df: pd.DataFrame) -> go.Figure:
    """Scatter: experience vs income colored by attrition."""
    if not {"TotalWorkingYears", "MonthlyIncome", "Attrition"}.issubset(df.columns):
        return go.Figure()
    fig = px.scatter(
        df,
        x="TotalWorkingYears",
        y="MonthlyIncome",
        color="Attrition",
        opacity=0.6,
        color_discrete_map={"Yes": COLORS["danger"], "No": COLORS["success"]},
    )
    return _apply_theme(fig, "Experience vs Income Analysis")


def promotion_analysis_chart(df: pd.DataFrame) -> go.Figure:
    """Attrition rate by years since last promotion buckets."""
    if "YearsSinceLastPromotion" not in df.columns:
        return go.Figure()
    temp = df.copy()
    temp["PromoBucket"] = pd.cut(
        temp["YearsSinceLastPromotion"],
        bins=[-1, 1, 3, 5, 10, 100],
        labels=["0-1", "2-3", "4-5", "6-10", "10+"],
    )
    agg = (
        temp.groupby("PromoBucket")["Attrition"]
        .apply(lambda x: (x.astype(str).str.lower().isin(["yes", "1"])).mean() * 100)
        .reset_index(name="Rate")
    )
    fig = px.line(agg, x="PromoBucket", y="Rate", markers=True)
    fig.update_traces(line_color=COLORS["warning"])
    return _apply_theme(fig, "Promotion Gap vs Attrition")


def travel_frequency_chart(df: pd.DataFrame) -> go.Figure:
    """Attrition by business travel type."""
    if "BusinessTravel" not in df.columns:
        return go.Figure()
    agg = (
        df.groupby("BusinessTravel")["Attrition"]
        .apply(lambda x: (x.astype(str).str.lower().isin(["yes", "1"])).mean() * 100)
        .reset_index(name="Rate")
    )
    fig = px.bar(agg, x="BusinessTravel", y="Rate", color="Rate", color_continuous_scale="Purples")
    return _apply_theme(fig, "Travel Frequency vs Attrition")


def monthly_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Simulated monthly attrition trend based on employee number grouping."""
    if "EmployeeNumber" not in df.columns or "Attrition" not in df.columns:
        return go.Figure()
    temp = df.copy()
    temp["MonthGroup"] = (temp.index % 12) + 1
    trend = (
        temp.groupby("MonthGroup")["Attrition"]
        .apply(lambda x: (x.astype(str).str.lower().isin(["yes", "1"])).mean() * 100)
        .reset_index(name="AttritionRate")
    )
    fig = px.area(
        trend,
        x="MonthGroup",
        y="AttritionRate",
        labels={"MonthGroup": "Month", "AttritionRate": "Attrition Rate (%)"},
    )
    fig.update_traces(line_color=COLORS["primary"], fillcolor="rgba(99,102,241,0.3)")
    return _apply_theme(fig, "Monthly Attrition Trend")


def model_comparison_chart(metrics: dict[str, dict[str, float]]) -> go.Figure:
    """Compare RF vs XGBoost metrics side by side."""
    models = list(metrics.keys())
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig = go.Figure()
    for model in models:
        fig.add_trace(
            go.Bar(
                name=model,
                x=[m.upper() for m in metric_names],
                y=[metrics[model].get(m, 0) for m in metric_names],
            )
        )
    fig.update_layout(barmode="group")
    return _apply_theme(fig, "Model Performance Comparison")
