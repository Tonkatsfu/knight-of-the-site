"""Tier 1 edge inference client: local ONNX scoring, confidence-gated escalation, offline queue, OTA sync.

Flow for each URL:
    1. Extract the 9 lexical features locally (no network).
    2. Score them with the local ONNX model -> P(phishing).
    3. If P <= low_thresh or P >= high_thresh, return the local verdict (URL never leaves the device).
       Otherwise (or when the user explicitly reports the URL) escalate to Tier 2.
    4. If Tier 2 is unreachable, keep the provisional local verdict and queue the escalation on disk.

Tier 2 HTTP contract (implemented in Phase 3/4):
    POST /api/v1/ingest        body: {url, tier1_features, tier1_probability, tier1_model_version, user_reported}
                               reply: {"verdict": "phishing" | "legitimate", "probability": float}
    GET  /api/v1/model/latest  reply: {"version": "vN", "download_url": "/relative/path", "sha256": hex}
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
import numpy as np
import onnxruntime as ort

from shared.feature_schema import TIER1_FEATURES
from shared.lexical_extractor import extract_lexical_features

DEFAULT_TIER2_URL = "http://localhost:8000"
DEFAULT_QUEUE_PATH = Path(__file__).resolve().parent / "offline_queue.json"
DEFAULT_TIMEOUT_S = 3.0
PHISHING_DECISION_THRESHOLD = 0.5

_SAFE_VERSION_RE = re.compile(r"^v\d+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VALID_VERDICTS = {"phishing", "legitimate"}

Route = Literal["local", "escalate"]


class ModelSyncError(RuntimeError):
    """Raised when an OTA model update cannot be fetched, verified, or loaded. The current model stays active."""


def route_decision(p_phishing: float, *, low: float, high: float, user_reported: bool = False) -> Route:
    """Confidence gate: resolve locally when P(phishing) is outside the open interval (low, high)."""
    if user_reported:
        return "escalate"
    return "local" if p_phishing <= low or p_phishing >= high else "escalate"


@dataclass(frozen=True)
class _ActiveModel:
    session: ort.InferenceSession
    input_name: str
    version: str


def _build_model(source: str | bytes, fallback_version: str) -> _ActiveModel:
    """Create an ONNX session and verify it is a 9-feature Tier 1 model. Raises ValueError otherwise."""
    try:
        session = ort.InferenceSession(source, providers=["CPUExecutionProvider"])
    except Exception as exc:  # onnxruntime raises several internal exception types
        raise ValueError(f"Not a loadable ONNX model: {exc}") from exc

    model_input = session.get_inputs()[0]
    if model_input.shape[-1] != len(TIER1_FEATURES):
        raise ValueError(f"Expected {len(TIER1_FEATURES)} input features, model has shape {model_input.shape}.")
    metadata = session.get_modelmeta().custom_metadata_map
    feature_names = metadata.get("feature_names")
    if feature_names is not None and feature_names.split(",") != TIER1_FEATURES:
        raise ValueError("Model feature order does not match TIER1_FEATURES.")
    return _ActiveModel(session, model_input.name, metadata.get("model_version", fallback_version))


class Tier1EdgeClient:
    """Edge-side classifier with confidence-gated escalation to the Tier 2 cloud server."""

    def __init__(
        self,
        onnx_path: str | Path,
        tier2_url: str = DEFAULT_TIER2_URL,
        low_thresh: float = 0.35,
        high_thresh: float = 0.65,
        *,
        http_client: httpx.Client | None = None,
        queue_path: str | Path = DEFAULT_QUEUE_PATH,
        models_dir: str | Path | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        if not 0.0 <= low_thresh <= high_thresh <= 1.0:
            raise ValueError(f"Require 0 <= low_thresh <= high_thresh <= 1, got ({low_thresh}, {high_thresh}).")
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.tier2_url = tier2_url
        self.queue_path = Path(queue_path)
        self.models_dir = Path(models_dir) if models_dir else Path(onnx_path).parent
        self._owns_http = http_client is None
        self._http = http_client or httpx.Client(base_url=tier2_url, timeout=timeout_s)
        self._queue_lock = threading.Lock()
        self._sync_lock = threading.Lock()
        self._active = _build_model(str(onnx_path), fallback_version=Path(onnx_path).stem)

    # ------------------------------------------------------------------ inference

    @property
    def model_version(self) -> str:
        return self._active.version

    def evaluate_url(self, url: str, user_reported: bool = False) -> dict[str, Any]:
        """Classify a URL. Raises ValueError/TypeError for unparseable input."""
        t0 = time.perf_counter()
        features = extract_lexical_features(url)
        vector = np.asarray([[features[name] for name in TIER1_FEATURES]], dtype=np.float32)
        t1 = time.perf_counter()

        model = self._active  # single read so a concurrent OTA swap cannot mix two models
        _, proba = model.session.run(None, {model.input_name: vector})
        p_phishing = float(proba[0, 1])
        t2 = time.perf_counter()

        local_verdict = "phishing" if p_phishing >= PHISHING_DECISION_THRESHOLD else "legitimate"
        result: dict[str, Any] = {
            "url": url,
            "features": features,
            "p_phishing": p_phishing,
            "decision": "local",
            "verdict": local_verdict,
            "verdict_source": "tier1",
            "cloud_probability": None,
            "model_version": model.version,
            "extract_ms": (t1 - t0) * 1000,
            "inference_ms": (t2 - t1) * 1000,
            "network_ms": 0.0,
        }
        if route_decision(p_phishing, low=self.low_thresh, high=self.high_thresh,
                          user_reported=user_reported) == "local":
            return result

        payload = {
            "url": url,
            "tier1_features": features,
            "tier1_probability": p_phishing,
            "tier1_model_version": model.version,
            "user_reported": user_reported,
        }
        try:
            cloud = self._post_ingest(payload)
        except (httpx.HTTPError, ValueError):
            self._enqueue(payload)
            result["decision"] = "queued_offline"
        else:
            result.update(decision="cloud", verdict=cloud["verdict"], verdict_source="tier2",
                          cloud_probability=cloud["probability"])
        result["network_ms"] = (time.perf_counter() - t2) * 1000
        return result

    def _post_ingest(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._http.post("/api/v1/ingest", json=payload)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict) or body.get("verdict") not in _VALID_VERDICTS:
            raise ValueError(f"Malformed Tier 2 reply: {body!r}")
        try:
            probability = float(body["probability"])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Tier 2 reply missing a numeric probability: {body!r}") from exc
        return {"verdict": body["verdict"], "probability": probability}

    # ------------------------------------------------------------------ offline queue

    def _read_queue(self) -> list[dict[str, Any]]:
        if not self.queue_path.exists():
            return []
        return json.loads(self.queue_path.read_text(encoding="utf-8"))

    def _write_queue(self, items: list[dict[str, Any]]) -> None:
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.queue_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, indent=2), encoding="utf-8")
        os.replace(tmp, self.queue_path)

    def _enqueue(self, payload: dict[str, Any]) -> None:
        with self._queue_lock:
            items = self._read_queue()
            items.append(payload)
            self._write_queue(items)

    def pending_queue_size(self) -> int:
        with self._queue_lock:
            return len(self._read_queue())

    def flush_offline_queue(self) -> int:
        """Re-send queued escalations in order; stop at the first failure. Returns the number delivered."""
        with self._queue_lock:
            items = self._read_queue()
            delivered = 0
            for payload in items:
                try:
                    self._post_ingest(payload)
                except (httpx.HTTPError, ValueError):
                    break
                delivered += 1
            if delivered:
                self._write_queue(items[delivered:])
            return delivered

    # ------------------------------------------------------------------ OTA model sync

    def sync_model(self) -> dict[str, Any]:
        """Fetch the latest Tier 1 model from Tier 2 and hot-swap it if newer.

        The download is verified (version format, SHA-256, loadable 9-feature ONNX graph) before it is
        written to disk or activated, so a failed sync never disturbs the running model.
        """
        with self._sync_lock:
            previous = self.model_version
            try:
                info = self._http.get("/api/v1/model/latest")
                info.raise_for_status()
                latest = info.json()
                version = str(latest["version"])
                download_url = str(latest["download_url"])
                expected_sha = str(latest["sha256"]).lower()
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                raise ModelSyncError(f"Could not fetch latest model info: {exc}") from exc

            if not _SAFE_VERSION_RE.match(version):
                raise ModelSyncError(f"Refusing unsafe model version string: {version!r}")
            if not download_url.startswith("/") or download_url.startswith("//"):
                raise ModelSyncError(f"download_url must be a path on the Tier 2 server: {download_url!r}")
            if not _SHA256_RE.match(expected_sha):
                raise ModelSyncError("Tier 2 did not provide a valid SHA-256 checksum.")
            if version == previous:
                return {"updated": False, "previous_version": previous, "version": previous}

            try:
                download = self._http.get(download_url)
                download.raise_for_status()
            except httpx.HTTPError as exc:
                raise ModelSyncError(f"Model download failed: {exc}") from exc
            blob = download.content
            if hashlib.sha256(blob).hexdigest() != expected_sha:
                raise ModelSyncError("Checksum mismatch — downloaded model rejected.")
            try:
                new_model = _build_model(blob, fallback_version=version)
            except ValueError as exc:
                raise ModelSyncError(str(exc)) from exc
            if new_model.version != version:
                raise ModelSyncError(f"Model metadata says {new_model.version!r}, server advertised {version!r}.")

            target = self.models_dir / f"tier1_{version}.onnx"
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            tmp.write_bytes(blob)
            os.replace(tmp, target)
            self._active = new_model
            return {"updated": True, "previous_version": previous, "version": version}

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> "Tier1EdgeClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
