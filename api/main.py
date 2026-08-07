"""
FastAPI application entry point for Employee Attrition Prediction API.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.routes import router  # noqa: E402
from database.database import init_db  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialize database tables."""
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as exc:
        logger.error("Database initialization failed: %s", exc)
    yield


app = FastAPI(
    title="Employee Attrition Prediction API",
    description="REST API for HR attrition analytics, predictions, and model management",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["Attrition"])


@app.get("/")
async def root():
    """Health check root endpoint."""
    return {
        "service": "Employee Attrition Prediction API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check for load balancers."""
    return {"status": "healthy"}
