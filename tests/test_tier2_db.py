"""Tests for Tier 2 settings and the SQLAlchemy persistence layer (`tier2_cloud/config.py`, `db.py`)."""

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from shared.feature_schema import TIER1_FEATURES, TIER2_HEAVY_FEATURES
from tier2_cloud.config import DEFAULT_EXTRACTION_BUDGET_S, Settings
from tier2_cloud.db import ModelVersion, ScanEvent, init_db, make_engine


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(f"sqlite:///{(tmp_path / 't2.db').as_posix()}")
    yield init_db(engine)
    engine.dispose()


def test_settings_from_env_overrides_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("TIER2_DATABASE_URL", "sqlite:///x.db")
    monkeypatch.setenv("TIER2_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("TIER2_EXTRACTION_BUDGET_S", "1.25")
    settings = Settings.from_env()
    assert settings.database_url == "sqlite:///x.db"
    assert settings.models_dir == tmp_path
    assert settings.extraction_budget_s == 1.25


def test_default_budget_fits_inside_tier1_client_timeout():
    from tier1_edge.inference_client import DEFAULT_TIMEOUT_S

    assert Settings().extraction_budget_s == DEFAULT_EXTRACTION_BUDGET_S < DEFAULT_TIMEOUT_S


def test_settings_reject_non_positive_budget():
    with pytest.raises(ValueError):
        Settings(extraction_budget_s=0)


def test_scan_event_round_trips_json_features(session_factory):
    tier1 = {name: 1 for name in TIER1_FEATURES}
    heavy = {name: -1 for name in TIER2_HEAVY_FEATURES}
    status = {name: "ok" for name in TIER2_HEAVY_FEATURES}
    with session_factory() as session:
        session.add(ScanEvent(
            url="http://example.com/", tier1_features=tier1, tier1_probability=0.5,
            tier1_model_version="v1", heavy_features=heavy, heavy_feature_status=status,
            extraction_ms=12.5, tier2_probability=0.9, verdict="phishing", tier2_model_version="v1",
        ))
        session.commit()

    with session_factory() as session:
        event = session.scalars(select(ScanEvent)).one()
        assert event.tier1_features == tier1
        assert event.heavy_features == heavy
        assert event.received_at is not None
        assert event.user_reported is False
        assert event.label is None


def test_model_version_tier_and_version_are_unique(session_factory):
    def row():
        return ModelVersion(tier="tier1", version="v1", filename="tier1_v1.onnx", sha256="0" * 64, size_bytes=1)

    with session_factory() as session:
        session.add(row())
        session.commit()
        session.add(row())
        with pytest.raises(IntegrityError):
            session.commit()
