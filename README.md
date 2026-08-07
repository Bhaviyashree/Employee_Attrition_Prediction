# Employee Attrition Prediction Dashboard

> **Changelog:** High-impact updates added (created now — 2026-08-07)

A production-ready HR Analytics platform for predicting employee attrition using machine learning. Built with **Streamlit**, **FastAPI**, **PostgreSQL**, **Random Forest**, and **XGBoost**.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-red.svg)

---

## Features

- **Interactive HR Dashboard** — KPI cards, Plotly charts, dark theme
- **Attrition Prediction** — Form-based prediction with risk scoring and recommendations
- **Analytics** — Department, salary, experience, promotion, and travel analysis
- **Dataset Management** — CSV upload, validation, cleaning, and download
- **Model Training** — Train and compare Random Forest vs XGBoost
- **Prediction History** — Stored in PostgreSQL with Excel export
- **REST API** — Full FastAPI backend with OpenAPI docs

---

## Project Structure

```
Employee_Attrition_Prediction/
├── dataset/
│   └── employee_attrition.csv      # Default HR dataset
├── model/
│   ├── attrition_model.pkl         # Best trained model
│   ├── scaler.pkl                  # Feature scaler
│   ├── label_encoder.pkl           # Categorical encoders
│   └── feature_names.pkl           # Feature column names
├── training/
│   ├── generate_dataset.py         # Generate synthetic dataset
│   └── train_model.py              # Train RF & XGBoost models
├── api/
│   ├── main.py                     # FastAPI app entry point
│   └── routes.py                   # API route handlers
├── dashboard/
│   ├── app.py                      # Streamlit main app
│   └── _pages/
│       ├── Dashboard.py            # Home KPIs & charts
│       ├── Dataset.py              # Upload & clean CSV
│       ├── Employees.py            # Employee records
│       ├── Analytics.py            # Advanced analytics
│       ├── Prediction.py           # Attrition prediction
│       ├── Performance.py          # Model training & metrics
│       ├── History.py              # Prediction history
│       ├── Settings.py             # App settings
│       └── About.py                # About page
├── database/
│   ├── database.py                 # SQLAlchemy connection
│   └── models.py                   # ORM models
├── utils/
│   ├── preprocessing.py            # Data cleaning & encoding
│   ├── charts.py                   # Plotly chart builders
│   └── helper.py                   # CSS, API client, exports
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python 3.10 or higher
- PostgreSQL 14+ (optional — falls back to SQLite)
- pip

---

## Installation

### 1. Clone and navigate to the project

```bash
cd Employee_Attrition_Prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the dataset

```bash
python training/generate_dataset.py
```

### 5. Train the ML models

```bash
python training/train_model.py
```

This trains Random Forest and XGBoost, compares metrics, and saves the best model to `model/`.

---

## Database Setup (PostgreSQL)

### Option A: PostgreSQL (Recommended)

```bash
# Create database
createdb employee_attrition

# Set environment variable
# Windows PowerShell
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/employee_attrition"

# Linux/macOS
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/employee_attrition"
```

Tables are created automatically on first API startup.

### Option B: SQLite (Default Fallback)

If PostgreSQL is unavailable, the app automatically uses `employee_attrition.db` in the project root.

---

## Running the Application

Open **two terminals** from the project root:

### Terminal 1 — Start the FastAPI Backend

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Terminal 2 — Start the Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard: [http://localhost:8501](http://localhost:8501)

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/v1/kpis` | Dashboard KPI metrics |
| GET | `/api/v1/employees` | List employees |
| POST | `/api/v1/employees` | Create employee |
| POST | `/api/v1/employees/sync-from-dataset` | Sync CSV to DB |
| POST | `/api/v1/predict` | Predict attrition |
| GET | `/api/v1/predictions/history` | Prediction history |
| POST | `/api/v1/dataset/upload` | Upload & validate CSV |
| GET | `/api/v1/model/metrics` | Model performance metrics |
| GET | `/api/v1/model/feature-importance` | Feature importances |
| POST | `/api/v1/model/train` | Train models |

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| Dashboard | KPI cards, department/gender/age/salary charts |
| Upload Dataset | CSV upload, validation, cleaning |
| Employee Records | Search, filter, paginate employees |
| Analytics | Interactive filters, trend analysis |
| Prediction | Form input, risk level, recommendations |
| Model Performance | Train RF/XGBoost, compare metrics |
| Prediction History | View and export predictions |
| Settings | API URL, theme, data management |
| About | Project information |

> Note: Streamlit page modules are stored in `dashboard/_pages/`, so the default Streamlit multipage explorer is intentionally hidden and the app launches directly at `http://localhost:8501`.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/employee_attrition` | PostgreSQL connection string |
| `ATTRITION_API_URL` | `http://localhost:8000` | API URL for Streamlit dashboard |

---

## Model Details

- **Algorithms:** Random Forest Classifier, XGBoost Classifier
- **Selection Criteria:** Best ROC AUC score on hold-out test set
- **Features:** 34 employee attributes (age, department, income, satisfaction, etc.)
- **Preprocessing:** Label encoding for categoricals, StandardScaler for numerics

---

## License

MIT License — free for personal and commercial use.

----
PS C:\YOURFILEPATHE\Employee_Attrition_Prediction> & c:/Users/Admin/Music/Employee_Attritio
(venv) PS C:\Users\Admin\Music\Employee_Attrition_Prediction> streamlit run dashboard/app.py

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.31.179:8501

----