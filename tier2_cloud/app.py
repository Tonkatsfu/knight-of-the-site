"""Tier 2 Cloud FastAPI application (`tier2_cloud/app.py`).

Implements the HTTP contract consumed by `tier1_edge.inference_client.Tier1EdgeClient`:
    POST /api/v1/ingest                   -> score escalated URL with 30 features + persist ScanEvent
    GET  /api/v1/model/latest             -> active Tier 1 ONNX version metadata & SHA-256
    GET  /api/v1/model/download/{version} -> raw .onnx bytes for OTA hot-swap
    GET  /api/v1/health                   -> liveness + active model versions
    GET  /api/v1/stats                    -> telemetry summary (counts, verdict split, latency)
    GET  /api/v1/events                   -> recent escalated scan events

Run locally:
    python -m uvicorn tier2_cloud.app:app --reload --port 8000
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select

from shared.feature_schema import BINARY_PHISHING_LABEL, TIER1_FEATURES, validate_tier1_features
from shared.lexical_extractor import extract_lexical_features
from tier2_cloud.config import Settings
from tier2_cloud.db import ScanEvent, init_db, make_engine
from tier2_cloud.heavy_extractor import HeavyFeatureExtractor
from tier2_cloud.model_registry import (
    SAFE_VERSION_RE,
    bootstrap_models,
    get_active_model,
    resolve_tier1_onnx_path,
)
from tier2_cloud.scoring import Tier2Scorer


class IngestRequest(BaseModel):
    """Escalation payload sent by `Tier1EdgeClient.evaluate_url` / `flush_offline_queue`."""

    url: str = Field(..., min_length=1)
    tier1_features: dict[str, int]
    tier1_probability: float = Field(..., ge=0.0, le=1.0)
    tier1_model_version: str = Field(default="v1", min_length=1, max_length=32)
    user_reported: bool = False

    @field_validator("tier1_features")
    @classmethod
    def _check_tier1_features(cls, value: dict[str, int]) -> dict[str, int]:
        validate_tier1_features(value, strict_domain=True)
        return {name: int(value[name]) for name in TIER1_FEATURES}


def create_app(
    settings: Settings | None = None,
    *,
    extractor: HeavyFeatureExtractor | None = None,
    scorer: Tier2Scorer | None = None,
) -> FastAPI:
    """Create and wire a Tier 2 FastAPI instance.

    Accepts optional `settings`, `extractor`, and `scorer` overrides so unit and integration tests
    can run against an isolated SQLite database and deterministic offline probes.
    """
    cfg = settings or Settings.from_env()
    engine = make_engine(cfg.database_url)
    session_factory = init_db(engine)
    active_t1, active_t2 = bootstrap_models(session_factory, cfg.models_dir)

    heavy_extractor = extractor or HeavyFeatureExtractor(
        budget_s=cfg.extraction_budget_s,
        blocklist_path=cfg.blocklist_path,
        tranco_path=cfg.tranco_path,
    )
    tier2_scorer = scorer or Tier2Scorer.from_joblib(
        cfg.models_dir / active_t2.filename,
        version=active_t2.version,
    )

    api = FastAPI(
        title="Knight of the Site — Tier 2 Cloud Server",
        version="1.0.0",
    )
    api.state.settings = cfg
    api.state.engine = engine
    api.state.session_factory = session_factory
    api.state.extractor = heavy_extractor
    api.state.scorer = tier2_scorer

    @api.get("/api/v1/health")
    def health() -> dict[str, Any]:
        latest_t1 = get_active_model(session_factory, "tier1")
        return {
            "status": "ok",
            "tier1_latest_version": latest_t1.version,
            "tier2_active_version": tier2_scorer.model_version,
        }

    @api.post("/api/v1/ingest")
    def ingest(req: IngestRequest) -> dict[str, Any]:
        try:
            lexical_features = extract_lexical_features(req.url)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        heavy = heavy_extractor.extract(req.url)
        scored = tier2_scorer.score(lexical_features, heavy.features)

        event = ScanEvent(
            url=req.url,
            user_reported=req.user_reported,
            tier1_features=req.tier1_features,
            tier1_probability=req.tier1_probability,
            tier1_model_version=req.tier1_model_version,
            heavy_features=heavy.features,
            heavy_feature_status=heavy.status,
            extraction_ms=heavy.extraction_ms,
            tier2_probability=scored.probability,
            verdict=scored.verdict,
            tier2_model_version=scored.model_version,
            label=BINARY_PHISHING_LABEL if req.user_reported else None,
            label_source="user_report" if req.user_reported else None,
        )
        with session_factory() as session:
            session.add(event)
            session.commit()
            session.refresh(event)

        return {
            "verdict": scored.verdict,
            "probability": scored.probability,
            "model_version": scored.model_version,
            "scan_id": event.id,
            "extraction_ms": heavy.extraction_ms,
        }

    @api.get("/api/v1/model/latest")
    def latest_model() -> dict[str, Any]:
        latest = get_active_model(session_factory, "tier1")
        return {
            "version": latest.version,
            "download_url": f"/api/v1/model/download/{latest.version}",
            "sha256": latest.sha256,
            "size_bytes": latest.size_bytes,
        }

    @api.get("/api/v1/model/download/{version}")
    def download_model(version: str) -> Response:
        if not SAFE_VERSION_RE.match(version):
            raise HTTPException(status_code=400, detail=f"Invalid model version: {version!r}")
        try:
            onnx_path = resolve_tier1_onnx_path(session_factory, cfg.models_dir, version)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(
            content=onnx_path.read_bytes(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{onnx_path.name}"'},
        )

    @api.get("/api/v1/stats")
    def stats() -> dict[str, Any]:
        with session_factory() as session:
            total = int(session.scalar(select(func.count(ScanEvent.id))) or 0)
            reported = int(
                session.scalar(select(func.count(ScanEvent.id)).where(ScanEvent.user_reported.is_(True)))
                or 0
            )
            phishing_count = int(
                session.scalar(select(func.count(ScanEvent.id)).where(ScanEvent.verdict == "phishing"))
                or 0
            )
            legit_count = int(
                session.scalar(select(func.count(ScanEvent.id)).where(ScanEvent.verdict == "legitimate"))
                or 0
            )
            avg_ms = float(session.scalar(select(func.avg(ScanEvent.extraction_ms))) or 0.0)
        latest_t1 = get_active_model(session_factory, "tier1")
        return {
            "total_scans": total,
            "user_reported_scans": reported,
            "phishing_verdicts": phishing_count,
            "legitimate_verdicts": legit_count,
            "avg_extraction_ms": avg_ms,
            "tier1_latest_version": latest_t1.version,
            "tier2_active_version": tier2_scorer.model_version,
        }

    @api.get("/api/v1/events")
    def list_events(limit: int = Query(default=25, ge=1, le=200)) -> list[dict[str, Any]]:
        with session_factory() as session:
            rows = session.scalars(
                select(ScanEvent).order_by(ScanEvent.id.desc()).limit(limit)
            ).all()
            return [
                {
                    "id": r.id,
                    "url": r.url,
                    "received_at": r.received_at.isoformat() if r.received_at else None,
                    "user_reported": r.user_reported,
                    "tier1_probability": r.tier1_probability,
                    "tier1_model_version": r.tier1_model_version,
                    "tier2_probability": r.tier2_probability,
                    "verdict": r.verdict,
                    "tier2_model_version": r.tier2_model_version,
                    "extraction_ms": r.extraction_ms,
                    "label": r.label,
                    "label_source": r.label_source,
                }
                for r in rows
            ]

    return api


app = create_app()
