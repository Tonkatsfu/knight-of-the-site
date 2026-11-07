"""SQLAlchemy 2.0 persistence for Tier 2: scan telemetry and the model registry.

The same ORM runs on SQLite (local development and tests) and PostgreSQL (deployment); only the
`database_url` changes. Feature vectors are stored as portable JSON columns.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ScanEvent(Base):
    """One URL escalated by a Tier 1 client (uncertain or user-reported) and scored by Tier 2."""

    __tablename__ = "scan_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    url: Mapped[str] = mapped_column(Text)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    user_reported: Mapped[bool] = mapped_column(Boolean, default=False)

    # What the edge client sent (kept for telemetry; Tier 2 scores its own recomputed lexical features)
    tier1_features: Mapped[dict[str, Any]] = mapped_column(JSON)
    tier1_probability: Mapped[float] = mapped_column(Float)
    tier1_model_version: Mapped[str] = mapped_column(String(32))

    # Heavy features after imputation, plus per-feature provenance: ok/timeout/error/blocked/unavailable
    heavy_features: Mapped[dict[str, Any]] = mapped_column(JSON)
    heavy_feature_status: Mapped[dict[str, Any]] = mapped_column(JSON)
    extraction_ms: Mapped[float] = mapped_column(Float)

    tier2_probability: Mapped[float] = mapped_column(Float)
    verdict: Mapped[str] = mapped_column(String(16))
    tier2_model_version: Mapped[str] = mapped_column(String(32))

    # Ground truth for continual retraining (Phase 4): 1 = phishing, 0 = legitimate
    label: Mapped[int | None] = mapped_column(Integer, nullable=True)
    label_source: Mapped[str | None] = mapped_column(String(32), nullable=True)


class ModelVersion(Base):
    """A model artifact on disk that Tier 2 serves (tier1 ONNX) or scores with (tier2 joblib)."""

    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("tier", "version", name="uq_model_tier_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tier: Mapped[str] = mapped_column(String(8))  # "tier1" | "tier2"
    version: Mapped[str] = mapped_column(String(16))  # "vN"
    filename: Mapped[str] = mapped_column(String(128))
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(Integer)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)


def make_engine(database_url: str) -> Engine:
    """Create an engine; SQLite connections may be used from FastAPI's worker threads."""
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, connect_args=connect_args, pool_pre_ping=True)


def init_db(engine: Engine) -> sessionmaker:
    """Create missing tables and return a session factory bound to `engine`."""
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
