"""Tier 2 30-feature cloud classifier loader and scorer (`tier2_cloud/scoring.py`)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import threading
from typing import Mapping

import joblib
from sklearn.base import ClassifierMixin

from shared.data_loader import load_stratified_splits
from shared.feature_schema import merge_tier1_and_heavy_features
from shared.model_training import train_tier2_model

PHISHING_THRESHOLD: float = 0.5
_VERSION_IN_STEM_RE = re.compile(r"_(v\d+)$")


@dataclass(frozen=True)
class ScoringResult:
    """Result of scoring a 30-feature vector with the active Tier 2 classifier."""

    verdict: str
    probability: float
    model_version: str


def _infer_version_from_path(path: Path, fallback: str = "v1") -> str:
    match = _VERSION_IN_STEM_RE.search(path.stem)
    return match.group(1) if match else fallback


def ensure_tier2_model_file(models_dir: str | Path, version: str = "v1") -> Path:
    """Return `models_dir/tier2_{version}.joblib`, training and saving it first if missing.

    `models/*.joblib` is gitignored per the course repository rules, so a fresh clone may only have
    `models/tier1_v1.onnx`. Auto-building `tier2_v1.joblib` on first startup makes Tier 2 runnable
    immediately after cloning.
    """
    target = Path(models_dir) / f"tier2_{version}.joblib"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    model = train_tier2_model(load_stratified_splits().train)
    joblib.dump(model, target, compress=3)
    return target


class Tier2Scorer:
    """Thread-safe wrapper around the active 30-feature scikit-learn classifier."""

    def __init__(
        self,
        model: ClassifierMixin,
        *,
        version: str = "v1",
        threshold: float = PHISHING_THRESHOLD,
    ) -> None:
        self._lock = threading.Lock()
        self._model = model
        self._version = version
        self.threshold = threshold

    @classmethod
    def from_joblib(
        cls,
        path: str | Path,
        *,
        version: str | None = None,
        threshold: float = PHISHING_THRESHOLD,
    ) -> "Tier2Scorer":
        p = Path(path)
        model = joblib.load(p)
        resolved_version = version or _infer_version_from_path(p)
        return cls(model, version=resolved_version, threshold=threshold)

    @property
    def model_version(self) -> str:
        with self._lock:
            return self._version

    def reload(self, model: ClassifierMixin, *, version: str) -> None:
        """Hot-swap the in-memory Tier 2 classifier (used by the Phase 4 retrainer)."""
        with self._lock:
            self._model = model
            self._version = version

    def score(
        self,
        tier1_features: Mapping[str, int],
        heavy_features: Mapping[str, int],
    ) -> ScoringResult:
        """Validate and merge the 9 + 21 features into the canonical 30-feature order, then score."""
        vec_30 = merge_tier1_and_heavy_features(tier1_features, heavy_features, strict_domain=True)
        with self._lock:
            model = self._model
            version = self._version
        proba = float(model.predict_proba(vec_30.reshape(1, -1))[0, 1])
        verdict = "phishing" if proba >= self.threshold else "legitimate"
        return ScoringResult(verdict=verdict, probability=proba, model_version=version)
