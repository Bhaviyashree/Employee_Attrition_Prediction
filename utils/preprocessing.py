"""
Data preprocessing, validation, and cleaning utilities.
Handles CSV ingestion, feature engineering, and model-ready transformations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

# Canonical column definitions aligned with IBM HR Analytics dataset
REQUIRED_COLUMNS = [
    "Age",
    "Attrition",
    "BusinessTravel",
    "DailyRate",
    "Department",
    "DistanceFromHome",
    "Education",
    "EducationField",
    "EmployeeCount",
    "EmployeeNumber",
    "EnvironmentSatisfaction",
    "Gender",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobRole",
    "JobSatisfaction",
    "MaritalStatus",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "Over18",
    "OverTime",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StandardHours",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

# Features used for prediction form and model training
PREDICTION_FEATURES = [
    "Age",
    "Gender",
    "Department",
    "JobRole",
    "Education",
    "MaritalStatus",
    "MonthlyIncome",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "DistanceFromHome",
    "BusinessTravel",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "WorkLifeBalance",
    "TrainingTimesLastYear",
    "PerformanceRating",
    "OverTime",
]

CATEGORICAL_COLUMNS = [
    "Attrition",
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "Over18",
    "OverTime",
]

NUMERIC_COLUMNS = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EmployeeCount",
    "EmployeeNumber",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StandardHours",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "dataset" / "employee_attrition.csv"
MODEL_DIR = PROJECT_ROOT / "model"


class DataValidationError(Exception):
    """Raised when uploaded dataset fails validation."""


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load employee attrition CSV from disk."""
    file_path = Path(path) if path else DEFAULT_DATASET_PATH
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info("Loaded dataset with %d rows from %s", len(df), file_path)
        return df
    except Exception as exc:
        logger.exception("Failed to load dataset")
        raise DataValidationError(f"Unable to read CSV: {exc}") from exc


def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """
    Validate dataset structure and return a diagnostic report.
    Allows partial datasets for upload preview but flags missing columns.
    """
    report: dict[str, Any] = {
        "valid": True,
        "rows": len(df),
        "columns": list(df.columns),
        "missing_columns": [],
        "missing_values": {},
        "duplicates": int(df.duplicated().sum()),
        "warnings": [],
    }

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        report["valid"] = False
        report["missing_columns"] = missing_cols
        report["warnings"].append(
            f"Missing required columns: {', '.join(missing_cols)}"
        )

    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            report["missing_values"][col] = null_count

    if report["duplicates"] > 0:
        report["warnings"].append(f"Found {report['duplicates']} duplicate rows")

    return report


def clean_dataset(
    df: pd.DataFrame,
    remove_missing: bool = True,
    remove_duplicates: bool = True,
) -> pd.DataFrame:
    """Clean dataset by handling missing values and duplicates."""
    cleaned = df.copy()

    if remove_duplicates:
        cleaned = cleaned.drop_duplicates()

    if remove_missing:
        cleaned = cleaned.dropna()

    # Normalize Attrition column values
    if "Attrition" in cleaned.columns:
        cleaned["Attrition"] = (
            cleaned["Attrition"]
            .astype(str)
            .str.strip()
            .str.replace("1", "Yes")
            .str.replace("0", "No")
        )
        cleaned.loc[
            cleaned["Attrition"].str.lower().isin(["yes", "y", "true"]),
            "Attrition",
        ] = "Yes"
        cleaned.loc[
            cleaned["Attrition"].str.lower().isin(["no", "n", "false"]),
            "Attrition",
        ] = "No"

    # Coerce numeric columns
    for col in NUMERIC_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    logger.info("Cleaned dataset: %d rows remaining", len(cleaned))
    return cleaned


def encode_categorical_features(
    df: pd.DataFrame,
    label_encoders: dict[str, LabelEncoder] | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Encode categorical columns using LabelEncoder per column."""
    encoded_df = df.copy()
    encoders = label_encoders or {}

    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in encoded_df.columns]
    for col in cat_cols:
        if fit or col not in encoders:
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            encoded_df[col] = encoded_df[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    return encoded_df, encoders


def prepare_features(
    df: pd.DataFrame,
    target_col: str = "Attrition",
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Prepare feature matrix X and optional target y."""
    feature_cols = feature_cols or [c for c in df.columns if c != target_col]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()

    y = None
    if target_col in df.columns:
        y = df[target_col].copy()
        if y.dtype == object or str(y.dtype) == "string":
            y = y.map({"Yes": 1, "No": 0, "yes": 1, "no": 0}).fillna(y)
            y = pd.to_numeric(y, errors="coerce")

    return X, y


def prepare_prediction_input(
    input_data: dict[str, Any],
    label_encoders: dict[str, LabelEncoder],
    feature_names: list[str],
) -> np.ndarray:
    """Transform a single prediction form submission into model input."""
    row: dict[str, Any] = {}

    for feature in feature_names:
        value = input_data.get(feature)
        if feature in label_encoders:
            le = label_encoders[feature]
            str_val = str(value)
            if str_val in le.classes_:
                row[feature] = le.transform([str_val])[0]
            else:
                # Fallback to most frequent class encoding
                row[feature] = le.transform([le.classes_[0]])[0]
        else:
            row[feature] = float(value) if value is not None else 0.0

    return np.array([[row[f] for f in feature_names]])


def save_preprocessing_artifacts(
    scaler: StandardScaler,
    label_encoders: dict[str, LabelEncoder],
    feature_names: list[str],
    model_dir: Path | None = None,
) -> None:
    """Persist scaler, encoders, and feature names."""
    out_dir = model_dir or MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out_dir / "scaler.pkl")
    joblib.dump(label_encoders, out_dir / "label_encoder.pkl")
    joblib.dump(feature_names, out_dir / "feature_names.pkl")


def load_preprocessing_artifacts(
    model_dir: Path | None = None,
) -> tuple[StandardScaler, dict[str, LabelEncoder], list[str]]:
    """Load persisted preprocessing artifacts."""
    in_dir = model_dir or MODEL_DIR
    scaler = joblib.load(in_dir / "scaler.pkl")
    label_encoders = joblib.load(in_dir / "label_encoder.pkl")
    feature_names = joblib.load(in_dir / "feature_names.pkl")
    return scaler, label_encoders, feature_names


def _attrition_rate(series: pd.Series) -> float:
    return (series.astype(str).str.lower().isin(["yes", "1", "true"]).mean() * 100) if not series.empty else 0.0


def get_kpi_metrics(df: pd.DataFrame) -> dict[str, object]:
    """Compute dashboard KPI metrics from employee dataframe."""
    total = len(df)
    attrition_col = df["Attrition"] if "Attrition" in df.columns else pd.Series(["No"] * total)

    attrition_yes = attrition_col.astype(str).str.lower().isin(["yes", "1", "true"]).sum()
    current = total - attrition_yes
    attrition_pct = (attrition_yes / total * 100) if total else 0.0

    department_high = "N/A"
    if {"Department", "Attrition"}.issubset(df.columns):
        dept_rates = (
            df.groupby("Department")["Attrition"]
            .apply(lambda x: _attrition_rate(x))
            .sort_values(ascending=False)
        )
        if not dept_rates.empty:
            department_high = f"{dept_rates.index[0]} ({dept_rates.iloc[0]:.1f}%)"

    job_role_high = "N/A"
    if {"JobRole", "Attrition"}.issubset(df.columns):
        role_rates = (
            df.groupby("JobRole")["Attrition"]
            .apply(lambda x: _attrition_rate(x))
            .sort_values(ascending=False)
        )
        if not role_rates.empty:
            job_role_high = f"{role_rates.index[0]} ({role_rates.iloc[0]:.1f}%)"

    overtime_rate = 0.0
    if {"OverTime", "Attrition"}.issubset(df.columns):
        overtime_rate = _attrition_rate(df.loc[df["OverTime"].astype(str).str.lower() == "yes", "Attrition"])

    low_salary_attrition = 0.0
    if "MonthlyIncome" in df.columns and "Attrition" in df.columns:
        temp = df.copy()
        try:
            temp["SalaryGroup"] = pd.qcut(temp["MonthlyIncome"], 4, labels=["Low", "Medium", "High", "Very High"], duplicates="drop")
            low_salary_attrition = _attrition_rate(temp.loc[temp["SalaryGroup"] == "Low", "Attrition"])
        except ValueError:
            low_salary_attrition = 0.0

    return {
        "total_employees": total,
        "current_employees": int(current),
        "attrition_count": int(attrition_yes),
        "attrition_percentage": round(float(attrition_pct), 2),
        "average_salary": round(float(df["MonthlyIncome"].mean()), 2) if "MonthlyIncome" in df.columns else 0.0,
        "average_experience": round(float(df["TotalWorkingYears"].mean()), 2) if "TotalWorkingYears" in df.columns else 0.0,
        "average_job_satisfaction": round(float(df["JobSatisfaction"].mean()), 2) if "JobSatisfaction" in df.columns else 0.0,
        "average_age": round(float(df["Age"].mean()), 2) if "Age" in df.columns else 0.0,
        "highest_attrition_department": department_high,
        "highest_attrition_job_role": job_role_high,
        "overtime_attrition_rate": round(float(overtime_rate), 2),
        "low_salary_attrition_rate": round(float(low_salary_attrition), 2),
    }


def get_risk_level(probability: float) -> str:
    """Map attrition probability to risk tier."""
    if probability < 0.35:
        return "Low Risk"
    if probability < 0.65:
        return "Medium Risk"
    return "High Risk"


def get_recommendation(prediction: str, probability: float, risk_level: str) -> str:
    """Generate HR recommendation based on prediction outcome."""
    if prediction == "Yes" or probability >= 0.5:
        if risk_level == "High Risk":
            return (
                "High attrition risk detected. Recommend immediate retention review: "
                "schedule 1-on-1 with manager, assess compensation alignment, "
                "review workload and career development opportunities."
            )
        return (
            "Moderate attrition risk. Consider proactive engagement: "
            "recognition programs, flexible work options, and skills development planning."
        )
    return (
        "Employee shows low attrition risk. Maintain current engagement strategies "
        "and continue monitoring satisfaction metrics quarterly."
    )
