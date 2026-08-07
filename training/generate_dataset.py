"""
Generate synthetic employee attrition dataset (IBM HR Analytics schema).
Run once to create dataset/employee_attrition.csv
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "dataset" / "employee_attrition.csv"

DEPARTMENTS = ["Sales", "Research & Development", "Human Resources"]
JOB_ROLES = [
    "Sales Executive",
    "Research Scientist",
    "Laboratory Technician",
    "Manufacturing Director",
    "Healthcare Representative",
    "Manager",
    "Sales Representative",
    "Research Director",
    "Human Resources",
]
GENDERS = ["Male", "Female"]
MARITAL = ["Single", "Married", "Divorced"]
TRAVEL = ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
EDUCATION_FIELDS = [
    "Life Sciences",
    "Medical",
    "Marketing",
    "Technical Degree",
    "Human Resources",
    "Other",
]
OVERTIME = ["Yes", "No"]


def _attrition_probability(row: dict) -> float:
    """Simulate realistic attrition probability based on features."""
    prob = 0.12
    if row["OverTime"] == "Yes":
        prob += 0.18
    if row["JobSatisfaction"] <= 2:
        prob += 0.15
    if row["WorkLifeBalance"] <= 2:
        prob += 0.12
    if row["EnvironmentSatisfaction"] <= 2:
        prob += 0.10
    if row["YearsAtCompany"] < 2:
        prob += 0.10
    if row["MonthlyIncome"] < 3000:
        prob += 0.08
    if row["Age"] < 30:
        prob += 0.06
    if row["DistanceFromHome"] > 15:
        prob += 0.05
    if row["MaritalStatus"] == "Single":
        prob += 0.04
    if row["BusinessTravel"] == "Travel_Frequently":
        prob += 0.05
    return min(prob, 0.85)


def generate_dataset(n_rows: int = 1470) -> pd.DataFrame:
    """Generate n_rows of synthetic employee data."""
    rows = []
    for i in range(1, n_rows + 1):
        age = int(np.random.randint(18, 61))
        dept = random.choice(DEPARTMENTS)
        job_role = random.choice(JOB_ROLES)
        monthly_income = int(np.random.lognormal(8.5, 0.5))
        monthly_income = max(1009, min(monthly_income, 20000))
        years_at_company = int(np.random.exponential(4)) + 1
        years_at_company = min(years_at_company, 40)
        years_in_role = min(int(np.random.exponential(2)) + 1, years_at_company)
        total_working = years_at_company + int(np.random.randint(0, 15))

        row = {
            "Age": age,
            "BusinessTravel": random.choice(TRAVEL),
            "DailyRate": int(np.random.randint(100, 1500)),
            "Department": dept,
            "DistanceFromHome": int(np.random.randint(1, 30)),
            "Education": int(np.random.randint(1, 6)),
            "EducationField": random.choice(EDUCATION_FIELDS),
            "EmployeeCount": 1,
            "EmployeeNumber": i,
            "EnvironmentSatisfaction": int(np.random.randint(1, 5)),
            "Gender": random.choice(GENDERS),
            "HourlyRate": int(np.random.randint(30, 100)),
            "JobInvolvement": int(np.random.randint(1, 5)),
            "JobLevel": int(np.random.randint(1, 6)),
            "JobRole": job_role,
            "JobSatisfaction": int(np.random.randint(1, 5)),
            "MaritalStatus": random.choice(MARITAL),
            "MonthlyIncome": monthly_income,
            "MonthlyRate": int(np.random.randint(2000, 27000)),
            "NumCompaniesWorked": int(np.random.randint(0, 10)),
            "Over18": "Y",
            "OverTime": random.choice(OVERTIME),
            "PercentSalaryHike": int(np.random.randint(11, 26)),
            "PerformanceRating": int(np.random.choice([3, 3, 3, 4, 4])),
            "RelationshipSatisfaction": int(np.random.randint(1, 5)),
            "StandardHours": 80,
            "StockOptionLevel": int(np.random.randint(0, 4)),
            "TotalWorkingYears": total_working,
            "TrainingTimesLastYear": int(np.random.randint(0, 7)),
            "WorkLifeBalance": int(np.random.randint(1, 5)),
            "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_role,
            "YearsSinceLastPromotion": min(int(np.random.exponential(2)), years_at_company),
            "YearsWithCurrManager": min(int(np.random.exponential(3)), years_at_company),
        }
        row["Attrition"] = "Yes" if random.random() < _attrition_probability(row) else "No"
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df)} rows -> {OUTPUT_PATH}")
    print(f"Attrition rate: {(df['Attrition'] == 'Yes').mean() * 100:.1f}%")
