"""
FastAPI route handlers for predictions, employees, analytics, and model training.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.database import get_db  # noqa: E402
from database.models import Employee, PredictionHistory  # noqa: E402
from utils.preprocessing import (  # noqa: E402
    MODEL_DIR,
    clean_dataset,
    encode_categorical_features,
    get_kpi_metrics,
    get_recommendation,
    get_risk_level,
    load_dataset,
    prepare_prediction_input,
    save_preprocessing_artifacts,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class PredictionInput(BaseModel):
    """Employee attributes for attrition prediction."""

    age: int = Field(..., ge=18, le=70, alias="Age")
    gender: str = Field(..., alias="Gender")
    department: str = Field(..., alias="Department")
    job_role: str = Field(..., alias="JobRole")
    education: int = Field(..., ge=1, le=5, alias="Education")
    marital_status: str = Field(..., alias="MaritalStatus")
    monthly_income: float = Field(..., ge=0, alias="MonthlyIncome")
    years_at_company: int = Field(..., ge=0, alias="YearsAtCompany")
    years_in_current_role: int = Field(..., ge=0, alias="YearsInCurrentRole")
    distance_from_home: int = Field(..., ge=0, alias="DistanceFromHome")
    business_travel: str = Field(..., alias="BusinessTravel")
    job_satisfaction: int = Field(..., ge=1, le=4, alias="JobSatisfaction")
    environment_satisfaction: int = Field(..., ge=1, le=4, alias="EnvironmentSatisfaction")
    work_life_balance: int = Field(..., ge=1, le=4, alias="WorkLifeBalance")
    training_times_last_year: int = Field(..., ge=0, alias="TrainingTimesLastYear")
    performance_rating: int = Field(..., ge=1, le=4, alias="PerformanceRating")
    overtime: str = Field(..., alias="OverTime")

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    confidence_score: float
    risk_level: str
    recommendation: str


class TrainRequest(BaseModel):
    model_type: str = Field(default="both", description="random_forest, xgboost, or both")


class EmployeeCreate(BaseModel):
    employee_number: Optional[int] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    department: Optional[str] = None
    job_role: Optional[str] = None
    education: Optional[int] = None
    marital_status: Optional[str] = None
    monthly_income: Optional[float] = None
    years_at_company: Optional[int] = None
    years_in_current_role: Optional[int] = None
    distance_from_home: Optional[int] = None
    business_travel: Optional[str] = None
    job_satisfaction: Optional[int] = None
    environment_satisfaction: Optional[int] = None
    work_life_balance: Optional[int] = None
    training_times_last_year: Optional[int] = None
    performance_rating: Optional[int] = None
    overtime: Optional[str] = None
    attrition: Optional[str] = None


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


def _load_model_artifacts() -> tuple[Any, Any, dict, list]:
    """Load trained model and preprocessing artifacts."""
    model_path = MODEL_DIR / "attrition_model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"
    encoder_path = MODEL_DIR / "label_encoder.pkl"
    features_path = MODEL_DIR / "feature_names.pkl"

    if not all(p.exists() for p in [model_path, scaler_path, encoder_path, features_path]):
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Run: python training/train_model.py",
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoders = joblib.load(encoder_path)
    feature_names = joblib.load(features_path)
    return model, scaler, encoders, feature_names


def _build_full_input(raw: dict[str, Any]) -> dict[str, Any]:
    """Enrich minimal form input with defaults for all model features."""
    defaults = {
        "DailyRate": 800,
        "EducationField": "Life Sciences",
        "EmployeeCount": 1,
        "EmployeeNumber": 9999,
        "HourlyRate": 65,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "MonthlyRate": 12000,
        "NumCompaniesWorked": 2,
        "Over18": "Y",
        "PercentSalaryHike": 15,
        "RelationshipSatisfaction": 3,
        "StandardHours": 80,
        "StockOptionLevel": 1,
        "TotalWorkingYears": raw.get("YearsAtCompany", 3) + 2,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 2,
    }
    merged = {**defaults, **raw}
    return merged


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/kpis")
async def get_kpis():
    """Return dashboard KPI metrics from default dataset."""
    try:
        df = clean_dataset(load_dataset())
        return get_kpi_metrics(df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/employees")
async def list_employees(
    skip: int = 0,
    limit: int = 100,
    search: Optional[str] = None,
    department: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """List employees from database with optional filters."""
    query = db.query(Employee)
    if department:
        query = query.filter(Employee.department == department)
    if search:
        query = query.filter(
            (Employee.department.ilike(f"%{search}%"))
            | (Employee.job_role.ilike(f"%{search}%"))
            | (Employee.gender.ilike(f"%{search}%"))
        )
    total = query.count()
    employees = query.offset(skip).limit(limit).all()
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "data": [
            {
                "id": e.id,
                "employee_number": e.employee_number,
                "age": e.age,
                "gender": e.gender,
                "department": e.department,
                "job_role": e.job_role,
                "monthly_income": e.monthly_income,
                "attrition": e.attrition,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in employees
        ],
    }


@router.post("/employees")
async def create_employee(employee: EmployeeCreate, db: Session = Depends(get_db)):
    """Create a new employee record."""
    db_employee = Employee(**employee.model_dump())
    db.add(db_employee)
    db.commit()
    db.refresh(db_employee)
    return {"id": db_employee.id, "message": "Employee created successfully"}


@router.post("/employees/sync-from-dataset")
async def sync_employees_from_dataset(db: Session = Depends(get_db)):
    """Sync employee records from CSV dataset into database."""
    try:
        df = clean_dataset(load_dataset())
        count = 0
        for _, row in df.iterrows():
            existing = (
                db.query(Employee)
                .filter(Employee.employee_number == row.get("EmployeeNumber"))
                .first()
            )
            if existing:
                continue
            emp = Employee(
                employee_number=int(row.get("EmployeeNumber", 0)),
                age=int(row.get("Age", 0)),
                gender=str(row.get("Gender", "")),
                department=str(row.get("Department", "")),
                job_role=str(row.get("JobRole", "")),
                education=int(row.get("Education", 1)),
                marital_status=str(row.get("MaritalStatus", "")),
                monthly_income=float(row.get("MonthlyIncome", 0)),
                years_at_company=int(row.get("YearsAtCompany", 0)),
                years_in_current_role=int(row.get("YearsInCurrentRole", 0)),
                distance_from_home=int(row.get("DistanceFromHome", 0)),
                business_travel=str(row.get("BusinessTravel", "")),
                job_satisfaction=int(row.get("JobSatisfaction", 3)),
                environment_satisfaction=int(row.get("EnvironmentSatisfaction", 3)),
                work_life_balance=int(row.get("WorkLifeBalance", 3)),
                training_times_last_year=int(row.get("TrainingTimesLastYear", 0)),
                performance_rating=int(row.get("PerformanceRating", 3)),
                overtime=str(row.get("OverTime", "No")),
                attrition=str(row.get("Attrition", "No")),
            )
            db.add(emp)
            count += 1
        db.commit()
        return {"synced": count, "message": f"Synced {count} employees"}
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/predict", response_model=PredictionResponse)
async def predict_attrition(payload: PredictionInput, db: Session = Depends(get_db)):
    """Predict employee attrition from input features."""
    try:
        model, scaler, encoders, feature_names = _load_model_artifacts()

        raw_input = {
            "Age": payload.age,
            "Gender": payload.gender,
            "Department": payload.department,
            "JobRole": payload.job_role,
            "Education": payload.education,
            "MaritalStatus": payload.marital_status,
            "MonthlyIncome": payload.monthly_income,
            "YearsAtCompany": payload.years_at_company,
            "YearsInCurrentRole": payload.years_in_current_role,
            "DistanceFromHome": payload.distance_from_home,
            "BusinessTravel": payload.business_travel,
            "JobSatisfaction": payload.job_satisfaction,
            "EnvironmentSatisfaction": payload.environment_satisfaction,
            "WorkLifeBalance": payload.work_life_balance,
            "TrainingTimesLastYear": payload.training_times_last_year,
            "PerformanceRating": payload.performance_rating,
            "OverTime": payload.overtime,
        }
        full_input = _build_full_input(raw_input)
        full_input["Attrition"] = "No"

        df_row = pd.DataFrame([full_input])
        encoded, _ = encode_categorical_features(df_row, encoders, fit=False)

        for col in feature_names:
            if col not in encoded.columns:
                encoded[col] = 0

        X = encoded[feature_names]
        X_scaled = scaler.transform(X)

        prob = float(model.predict_proba(X_scaled)[0][1])
        pred_label = "Yes" if prob >= 0.5 else "No"
        confidence = float(max(prob, 1 - prob))
        risk = get_risk_level(prob)
        recommendation = get_recommendation(pred_label, prob, risk)

        # Save to prediction history
        history = PredictionHistory(
            age=payload.age,
            gender=payload.gender,
            department=payload.department,
            job_role=payload.job_role,
            monthly_income=payload.monthly_income,
            prediction_result=pred_label,
            probability=round(prob, 4),
            risk_level=risk,
            recommendation=recommendation,
            input_payload=json.dumps(raw_input),
        )
        db.add(history)
        db.commit()

        return PredictionResponse(
            prediction=pred_label,
            probability=round(prob, 4),
            confidence_score=round(confidence, 4),
            risk_level=risk,
            recommendation=recommendation,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/predictions/history")
async def get_prediction_history(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """Retrieve prediction history records."""
    total = db.query(PredictionHistory).count()
    records = (
        db.query(PredictionHistory)
        .order_by(PredictionHistory.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return {
        "total": total,
        "data": [
            {
                "id": r.id,
                "department": r.department,
                "job_role": r.job_role,
                "prediction_result": r.prediction_result,
                "probability": r.probability,
                "risk_level": r.risk_level,
                "recommendation": r.recommendation,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ],
    }


@router.post("/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload and validate a CSV dataset."""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        from utils.preprocessing import validate_dataset

        report = validate_dataset(df)
        cleaned = clean_dataset(df)
        return {
            "filename": file.filename,
            "validation": report,
            "preview": cleaned.head(10).to_dict(orient="records"),
            "cleaned_rows": len(cleaned),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/model/metrics")
async def get_model_metrics():
    """Return saved model performance metrics."""
    metadata_path = MODEL_DIR / "model_metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Model not trained yet")
    with open(metadata_path, encoding="utf-8") as f:
        return json.load(f)


@router.get("/model/feature-importance")
async def get_feature_importance():
    """Return feature importances from trained model."""
    try:
        model, _, _, feature_names = _load_model_artifacts()
        if not hasattr(model, "feature_importances_"):
            raise HTTPException(status_code=404, detail="Model has no feature importances")
        return {
            "features": feature_names,
            "importances": [float(x) for x in model.feature_importances_],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/model/train")
async def train_model_endpoint(request: TrainRequest):
    """Train Random Forest and/or XGBoost and save best model."""
    try:
        df = clean_dataset(load_dataset())
        encoded_df, label_encoders = encode_categorical_features(df, fit=True)

        target = encoded_df["Attrition"]
        if target.max() > 1:
            target = df["Attrition"].map({"Yes": 1, "No": 0})

        feature_cols = [c for c in encoded_df.columns if c != "Attrition"]
        X = encoded_df[feature_cols]
        y = target.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        models_to_train = {}
        if request.model_type in ("random_forest", "both"):
            models_to_train["Random Forest"] = RandomForestClassifier(
                n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
            )
        if request.model_type in ("xgboost", "both"):
            models_to_train["XGBoost"] = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42,
                eval_metric="logloss", use_label_encoder=False,
            )

        results = {}
        best_model = None
        best_name = ""
        best_auc = 0.0

        for name, model in models_to_train.items():
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            y_prob = model.predict_proba(X_test_s)[:, 1]
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, y_prob)),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            }
            if hasattr(model, "feature_importances_"):
                metrics["feature_importances"] = dict(
                    zip(feature_cols, [float(x) for x in model.feature_importances_])
                )
            results[name] = metrics
            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_model = model
                best_name = name

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODEL_DIR / "attrition_model.pkl")
        save_preprocessing_artifacts(scaler, label_encoders, feature_cols)

        metadata = {"best_model": best_name, "metrics": results, "trained_at": datetime.utcnow().isoformat()}
        with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return metadata
    except Exception as exc:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
