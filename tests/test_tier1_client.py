"""Tests for the Tier 1 confidence-gated edge client (`tier1_edge/inference_client.py`).

Tier 2 does not exist yet (Phase 3), so its HTTP API is simulated with `httpx.MockTransport`.
These tests pin down the contract Tier 2 must implement:

    POST /api/v1/ingest            -> {"verdict": "phishing" | "legitimate", "probability": float}
    GET  /api/v1/model/latest      -> {"version": str, "download_url": str, "sha256": str}
    GET  <download_url>            -> raw .onnx bytes
"""

import hashlib
import json

import httpx
import pytest

from shared.data_loader import load_stratified_splits
from shared.feature_schema import TIER1_FEATURES
from shared.model_training import export_tier1_onnx, train_tier1_model
from tier1_edge.inference_client import ModelSyncError, Tier1EdgeClient, route_decision

TIER2_URL = "http://tier2.test"
BENIGN_URL = "https://www.google.com/"


@pytest.fixture(scope="module")
def tier1_model():
    return train_tier1_model(load_stratified_splits().train)


@pytest.fixture
def v1_path(tmp_path, tier1_model):
    return export_tier1_onnx(tier1_model, tmp_path / "models" / "tier1_v1.onnx", version="v1")


def _client(v1_path, tmp_path, handler, **kwargs):
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url=TIER2_URL)
    return Tier1EdgeClient(
        v1_path,
        tier2_url=TIER2_URL,
        http_client=http,
        queue_path=tmp_path / "offline_queue.json",
        models_dir=v1_path.parent,
        **kwargs,
    )


def _unreachable(request):
    raise httpx.ConnectError("Tier 2 offline", request=request)


# --- Pure routing rule ------------------------------------------------------

@pytest.mark.parametrize(
    "p, expected",
    [(0.0, "local"), (0.35, "local"), (0.36, "escalate"), (0.5, "escalate"),
     (0.64, "escalate"), (0.65, "local"), (1.0, "local")],
)
def test_route_decision_gate_boundaries(p, expected):
    assert route_decision(p, low=0.35, high=0.65) == expected


def test_user_report_always_escalates_even_when_confident():
    assert route_decision(0.01, low=0.35, high=0.65, user_reported=True) == "escalate"


@pytest.mark.parametrize("low, high", [(0.7, 0.3), (-0.1, 0.5), (0.5, 1.1)])
def test_invalid_thresholds_rejected(v1_path, tmp_path, low, high):
    with pytest.raises(ValueError):
        _client(v1_path, tmp_path, _unreachable, low_thresh=low, high_thresh=high)


# --- Local resolution -------------------------------------------------------

def test_confident_prediction_resolves_locally_without_network(v1_path, tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(500)

    # low == high == 0.5 makes every probability "confident"
    client = _client(v1_path, tmp_path, handler, low_thresh=0.5, high_thresh=0.5)
    result = client.evaluate_url(BENIGN_URL)

    assert requests == []
    assert result["decision"] == "local"
    assert result["verdict_source"] == "tier1"
    assert result["verdict"] == ("phishing" if result["p_phishing"] >= 0.5 else "legitimate")
    assert list(result["features"]) == TIER1_FEATURES
    assert result["model_version"] == "v1"


def test_local_inference_is_under_10ms(v1_path, tmp_path):
    client = _client(v1_path, tmp_path, _unreachable, low_thresh=0.5, high_thresh=0.5)
    client.evaluate_url(BENIGN_URL)  # warm-up
    result = client.evaluate_url(BENIGN_URL)
    assert result["extract_ms"] + result["inference_ms"] < 10.0


def test_invalid_url_raises_value_error(v1_path, tmp_path):
    client = _client(v1_path, tmp_path, _unreachable)
    with pytest.raises(ValueError):
        client.evaluate_url("http://")


# --- Escalation to Tier 2 ---------------------------------------------------

def test_escalation_sends_features_and_uses_cloud_verdict(v1_path, tmp_path):
    sent = []

    def handler(request):
        assert request.method == "POST" and request.url.path == "/api/v1/ingest"
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"verdict": "phishing", "probability": 0.93})

    client = _client(v1_path, tmp_path, handler)
    result = client.evaluate_url(BENIGN_URL, user_reported=True)

    assert result["decision"] == "cloud"
    assert result["verdict"] == "phishing"
    assert result["verdict_source"] == "tier2"
    payload = sent[0]
    assert payload["url"] == BENIGN_URL
    assert payload["user_reported"] is True
    assert payload["tier1_model_version"] == "v1"
    assert list(payload["tier1_features"]) == TIER1_FEATURES
    assert 0.0 <= payload["tier1_probability"] <= 1.0


@pytest.mark.parametrize(
    "response",
    [httpx.Response(503), httpx.Response(200, json={"unexpected": True}),
     httpx.Response(200, json={"verdict": "phishing"}), httpx.Response(200, text="not json")],
)
def test_tier2_error_or_malformed_reply_falls_back_to_queue(v1_path, tmp_path, response):
    client = _client(v1_path, tmp_path, lambda request: response)
    result = client.evaluate_url(BENIGN_URL, user_reported=True)
    assert result["decision"] == "queued_offline"


def test_offline_escalation_is_queued_with_provisional_local_verdict(v1_path, tmp_path):
    client = _client(v1_path, tmp_path, _unreachable)
    result = client.evaluate_url(BENIGN_URL, user_reported=True)

    assert result["decision"] == "queued_offline"
    assert result["verdict_source"] == "tier1"
    assert result["verdict"] in {"phishing", "legitimate"}
    assert client.pending_queue_size() == 1
    queued = json.loads((tmp_path / "offline_queue.json").read_text(encoding="utf-8"))
    assert queued[0]["url"] == BENIGN_URL


def test_flush_offline_queue_delivers_when_tier2_returns(v1_path, tmp_path):
    offline = _client(v1_path, tmp_path, _unreachable)
    offline.evaluate_url(BENIGN_URL, user_reported=True)
    offline.evaluate_url("http://bit.ly/abc", user_reported=True)

    online = _client(
        v1_path, tmp_path,
        lambda request: httpx.Response(200, json={"verdict": "legitimate", "probability": 0.1}),
    )
    assert online.pending_queue_size() == 2
    assert online.flush_offline_queue() == 2
    assert online.pending_queue_size() == 0


# --- Over-the-air model sync ------------------------------------------------

def _ota_handler(version, onnx_bytes, sha256=None):
    def handler(request):
        if request.url.path == "/api/v1/model/latest":
            return httpx.Response(200, json={
                "version": version,
                "download_url": f"/api/v1/model/download/{version}",
                "sha256": sha256 or hashlib.sha256(onnx_bytes).hexdigest(),
            })
        if request.url.path == f"/api/v1/model/download/{version}":
            return httpx.Response(200, content=onnx_bytes)
        return httpx.Response(404)
    return handler


def test_sync_model_hot_swaps_to_newer_version(v1_path, tmp_path, tier1_model):
    v2_bytes = export_tier1_onnx(tier1_model, tmp_path / "build" / "v2.onnx", version="v2").read_bytes()
    client = _client(v1_path, tmp_path, _ota_handler("v2", v2_bytes))

    outcome = client.sync_model()

    assert outcome == {"updated": True, "previous_version": "v1", "version": "v2"}
    assert client.model_version == "v2"
    assert (v1_path.parent / "tier1_v2.onnx").read_bytes() == v2_bytes
    assert client.evaluate_url(BENIGN_URL)["model_version"] == "v2"


@pytest.mark.parametrize("version", ["../../evil", "v2/../../x", ""])
def test_sync_model_rejects_unsafe_version_strings(v1_path, tmp_path, version):
    client = _client(v1_path, tmp_path, _ota_handler(version, v1_path.read_bytes()))
    with pytest.raises(ModelSyncError):
        client.sync_model()
    assert client.model_version == "v1"


def test_sync_model_is_noop_when_already_current(v1_path, tmp_path):
    client = _client(v1_path, tmp_path, _ota_handler("v1", v1_path.read_bytes()))
    assert client.sync_model() == {"updated": False, "previous_version": "v1", "version": "v1"}


@pytest.mark.parametrize(
    "payload, sha256",
    [(b"not an onnx model", None), (b"tampered bytes", "0" * 64)],
)
def test_sync_model_rejects_bad_download_and_keeps_current_model(v1_path, tmp_path, payload, sha256):
    client = _client(v1_path, tmp_path, _ota_handler("v2", payload, sha256))
    with pytest.raises(ModelSyncError):
        client.sync_model()
    assert client.model_version == "v1"
    assert not (v1_path.parent / "tier1_v2.onnx").exists()


def test_sync_model_raises_when_tier2_unreachable(v1_path, tmp_path):
    client = _client(v1_path, tmp_path, _unreachable)
    with pytest.raises(ModelSyncError):
        client.sync_model()
    assert client.model_version == "v1"
