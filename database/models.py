"""
SQLAlchemy ORM models for employees and prediction history.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Employee(Base):
    """Stored employee record from dataset or manual entry."""

    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    employee_number = Column(Integer, unique=True, nullable=True, index=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(20), nullable=True)
    department = Column(String(100), nullable=True)
    job_role = Column(String(100), nullable=True)
    education = Column(Integer, nullable=True)
    marital_status = Column(String(50), nullable=True)
    monthly_income = Column(Float, nullable=True)
    years_at_company = Column(Integer, nullable=True)
    years_in_current_role = Column(Integer, nullable=True)
    distance_from_home = Column(Integer, nullable=True)
    business_travel = Column(String(50), nullable=True)
    job_satisfaction = Column(Integer, nullable=True)
    environment_satisfaction = Column(Integer, nullable=True)
    work_life_balance = Column(Integer, nullable=True)
    training_times_last_year = Column(Integer, nullable=True)
    performance_rating = Column(Integer, nullable=True)
    overtime = Column(String(10), nullable=True)
    attrition = Column(String(10), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionHistory(Base):
    """Historical attrition predictions with metadata."""

    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    employee_name = Column(String(200), nullable=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(20), nullable=True)
    department = Column(String(100), nullable=True)
    job_role = Column(String(100), nullable=True)
    monthly_income = Column(Float, nullable=True)
    prediction_result = Column(String(10), nullable=False)
    probability = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=True)
    recommendation = Column(Text, nullable=True)
    input_payload = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
