"""
PostgreSQL database connection and session management via SQLAlchemy.
Falls back to SQLite for local development when PostgreSQL is unavailable.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from database.models import Base

logger = logging.getLogger(__name__)

# PostgreSQL connection (override via environment variable)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/employee_attrition",
)

# SQLite fallback for development without PostgreSQL
SQLITE_FALLBACK_URL = "sqlite:///./employee_attrition.db"


def _create_engine_with_fallback():
    """Try PostgreSQL first, fall back to SQLite."""
    try:
        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
        with engine.connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        logger.info("Connected to PostgreSQL database")
        return engine
    except Exception as exc:
        logger.warning(
            "PostgreSQL unavailable (%s). Using SQLite fallback.", exc
        )
        return create_engine(
            SQLITE_FALLBACK_URL,
            connect_args={"check_same_thread": False},
        )


engine = _create_engine_with_fallback()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized")


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yield database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for database sessions outside FastAPI."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
