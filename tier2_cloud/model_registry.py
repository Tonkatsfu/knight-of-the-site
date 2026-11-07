"""Model registry service for Tier 2 (`tier2_cloud/model_registry.py`).

Tracks versioned `tier1_vN.onnx` and `tier2_vN.joblib` artifacts in the SQLAlchemy `model_versions`
table and serves ONNX binaries for Tier 1 Over-The-Air (OTA) model updates.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import sessionmaker

from shared.data_loader import load_stratified_splits
from shared.model_training import export_tier1_onnx, read_onnx_metadata, train_tier1_model
from tier2_cloud.db import ModelVersion
from tier2_cloud.scoring import ensure_tier2_model_file

SAFE_VERSION_RE = re.compile(r"^v(\d+)$")
_TIER_FILE_RE = re.compile(r"^(tier1|tier2)_(v\d+)\.(onnx|joblib)$")


def _version_num(version: str) -> int:
    match = SAFE_VERSION_RE.match(version)
    return int(match.group(1)) if match else -1


def compute_sha256(path: str | Path) -> str:
    """Compute the lowercase hex SHA-256 digest of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ensure_tier1_model_file(models_dir: str | Path, version: str = "v1") -> Path:
    """Return `models_dir/tier1_{version}.onnx`, training and exporting it first if missing."""
    target = Path(models_dir) / f"tier1_{version}.onnx"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    model = train_tier1_model(load_stratified_splits().train)
    return export_tier1_onnx(model, target, version=version)


def register_model_artifact(
    session_factory: sessionmaker,
    *,
    tier: str,
    version: str,
    path: str | Path,
    metrics: dict[str, Any] | None = None,
    activate: bool = True,
) -> ModelVersion:
    """Insert or update a `ModelVersion` record for `path` and optionally mark it active for `tier`."""
    if tier not in {"tier1", "tier2"}:
        raise ValueError(f"tier must be 'tier1' or 'tier2', got {tier!r}")
    if not SAFE_VERSION_RE.match(version):
        raise ValueError(f"Invalid model version string: {version!r}")

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {file_path}")

    sha = compute_sha256(file_path)
    size = file_path.stat().st_size

    with session_factory() as session:
        if activate:
            session.execute(
                update(ModelVersion).where(ModelVersion.tier == tier).values(is_active=False)
            )
        existing = session.scalars(
            select(ModelVersion).where(ModelVersion.tier == tier, ModelVersion.version == version)
        ).first()
        if existing is None:
            record = ModelVersion(
                tier=tier,
                version=version,
                filename=file_path.name,
                sha256=sha,
                size_bytes=size,
                metrics=metrics,
                is_active=activate,
            )
            session.add(record)
        else:
            existing.filename = file_path.name
            existing.sha256 = sha
            existing.size_bytes = size
            if metrics is not None:
                existing.metrics = metrics
            if activate:
                existing.is_active = True
            record = existing
        session.commit()
        session.refresh(record)
        return record


def bootstrap_models(session_factory: sessionmaker, models_dir: str | Path) -> tuple[ModelVersion, ModelVersion]:
    """Ensure at least v1 of Tier 1 and Tier 2 exist on disk, register all artifacts in `models_dir`,
    and activate the highest version number for each tier.
    """
    mdir = Path(models_dir)
    mdir.mkdir(parents=True, exist_ok=True)

    has_t1 = any(mdir.glob("tier1_v*.onnx"))
    has_t2 = any(mdir.glob("tier2_v*.joblib"))
    if not has_t1:
        ensure_tier1_model_file(mdir, "v1")
    if not has_t2:
        ensure_tier2_model_file(mdir, "v1")

    discovered: dict[str, list[tuple[int, str, Path]]] = {"tier1": [], "tier2": []}
    for candidate in sorted(mdir.iterdir()):
        match = _TIER_FILE_RE.match(candidate.name)
        if not match or not candidate.is_file():
            continue
        tier, ver, ext = match.groups()
        if (tier == "tier1" and ext != "onnx") or (tier == "tier2" and ext != "joblib"):
            continue
        if tier == "tier1":
            meta_ver = read_onnx_metadata(candidate).get("model_version")
            if meta_ver and meta_ver != ver:
                continue
        discovered[tier].append((_version_num(ver), ver, candidate))

    active_records: dict[str, ModelVersion] = {}
    for tier in ("tier1", "tier2"):
        items = sorted(discovered[tier], key=lambda item: item[0])
        for idx, (_, ver, path) in enumerate(items):
            is_last = idx == len(items) - 1
            rec = register_model_artifact(
                session_factory,
                tier=tier,
                version=ver,
                path=path,
                activate=is_last,
            )
            if is_last:
                active_records[tier] = rec

    return active_records["tier1"], active_records["tier2"]


def get_active_model(session_factory: sessionmaker, tier: str) -> ModelVersion:
    """Return the active `ModelVersion` row for `tier` ('tier1' or 'tier2')."""
    with session_factory() as session:
        row = session.scalars(
            select(ModelVersion)
            .where(ModelVersion.tier == tier, ModelVersion.is_active.is_(True))
            .order_by(ModelVersion.id.desc())
        ).first()
        if row is None:
            raise LookupError(f"No active model registered for {tier!r}.")
        return row


def resolve_tier1_onnx_path(
    session_factory: sessionmaker,
    models_dir: str | Path,
    version: str,
) -> Path:
    """Validate `version` (`^v\\d+$`) and return the verified `.onnx` file path on disk."""
    if not SAFE_VERSION_RE.match(version):
        raise ValueError(f"Invalid model version string: {version!r}")

    with session_factory() as session:
        row = session.scalars(
            select(ModelVersion).where(ModelVersion.tier == "tier1", ModelVersion.version == version)
        ).first()

    mdir = Path(models_dir).resolve()
    filename = row.filename if row is not None else f"tier1_{version}.onnx"
    candidate = (mdir / filename).resolve()
    if candidate.parent != mdir or not candidate.is_file():
        raise FileNotFoundError(f"Tier 1 ONNX artifact for version {version!r} not found.")
    return candidate
