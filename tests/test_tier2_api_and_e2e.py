"""Tests for Tier 2 scorer, model registry, FastAPI endpoints, and end-to-end Tier 1 <-> Tier 2 integration."""

from datetime import datetime, timedelta, timezone
import hashlib
import shutil
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from shared.data_loader import load_stratified_splits
from shared.lexical_extractor import extract_lexical_features
from shared.model_training import export_tier1_onnx, train_tier1_model
from tier1_edge.inference_client import Tier1EdgeClient
from tier2_cloud.app import create_app
from tier2_cloud.config import Settings
from tier2_cloud.heavy_extractor import HeavyFeatureExtractor, HttpFetchResult, TlsProbeResult
from tier2_cloud.model_registry import register_model_artifact, resolve_tier1_onnx_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMMITTED_MODELS_DIR = PROJECT_ROOT / "models"

BENIGN_HTML = """
<html>
  <head><link rel="icon" href="/favicon.ico" /></head>
  <body>
    <a href="/a">A</a><a href="/b">B</a><a href="/c">C</a>
    <form action="/login"></form>
  </body>
</html>
"""


@pytest.fixture(scope="module")
def fake_extractor():
    now = datetime.now(timezone.utc)
    return HeavyFeatureExtractor(
        budget_s=1.0,
        blocklist_path=None,
        tranco_path=None,
        http_fetcher=lambda url, t: HttpFetchResult(html=BENIGN_HTML, redirect_count=0, final_url=url),
        tls_checker=lambda host, port, t: TlsProbeResult(has_tls=True, trusted_issuer=True, cert_age_days=400),
        dns_resolver=lambda host, t: True,
        whois_lookup=lambda host, t: {
            "domain_name": host,
            "creation_date": now - timedelta(days=900),
            "expiration_date": now + timedelta(days=500),
        },
    )


@pytest.fixture
def tier2_env(tmp_path, fake_extractor):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(COMMITTED_MODELS_DIR / "tier1_v1.onnx", models_dir / "tier1_v1.onnx")
    if (COMMITTED_MODELS_DIR / "tier2_v1.joblib").exists():
        shutil.copy2(COMMITTED_MODELS_DIR / "tier2_v1.joblib", models_dir / "tier2_v1.joblib")

    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'tier2_test.db').as_posix()}",
        models_dir=models_dir,
        extraction_budget_s=1.0,
        tranco_path=None,
        blocklist_path=None,
    )
    app = create_app(settings, extractor=fake_extractor)
    client = TestClient(app)
    yield app, client, models_dir
    app.state.engine.dispose()


def test_health_and_initial_stats(tier2_env):
    _, client, _ = tier2_env
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "tier1_latest_version": "v1",
        "tier2_active_version": "v1",
    }

    stats = client.get("/api/v1/stats")
    assert stats.status_code == 200
    assert stats.json()["total_scans"] == 0


def test_ingest_scores_30_features_and_persists_scan_event(tier2_env):
    _, client, _ = tier2_env
    url = "http://paypal-account.com/login"
    payload = {
        "url": url,
        "tier1_features": extract_lexical_features(url),
        "tier1_probability": 0.50,
        "tier1_model_version": "v1",
        "user_reported": False,
    }
    resp = client.post("/api/v1/ingest", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] in {"phishing", "legitimate"}
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_version"] == "v1"
    assert body["scan_id"] == 1

    events = client.get("/api/v1/events").json()
    assert len(events) == 1
    assert events[0]["url"] == url
    assert events[0]["label"] is None


def test_ingest_user_reported_records_phishing_label_for_phase4(tier2_env):
    _, client, _ = tier2_env
    url = "https://www.google.com/"
    payload = {
        "url": url,
        "tier1_features": extract_lexical_features(url),
        "tier1_probability": 0.08,
        "tier1_model_version": "v1",
        "user_reported": True,
    }
    resp = client.post("/api/v1/ingest", json=payload)
    assert resp.status_code == 200

    events = client.get("/api/v1/events").json()
    assert events[0]["user_reported"] is True
    assert events[0]["label"] == 1
    assert events[0]["label_source"] == "user_report"


def test_ingest_rejects_invalid_tier1_feature_domain(tier2_env):
    _, client, _ = tier2_env
    bad_features = extract_lexical_features("https://www.google.com/")
    bad_features["having_IP_Address"] = 99  # outside {-1, 1}
    resp = client.post(
        "/api/v1/ingest",
        json={
            "url": "https://www.google.com/",
            "tier1_features": bad_features,
            "tier1_probability": 0.5,
            "tier1_model_version": "v1",
            "user_reported": False,
        },
    )
    assert resp.status_code == 422


def test_model_latest_and_download_validation(tier2_env):
    app, client, models_dir = tier2_env
    latest = client.get("/api/v1/model/latest").json()
    assert latest["version"] == "v1"
    assert latest["download_url"] == "/api/v1/model/download/v1"

    dl = client.get(latest["download_url"])
    assert dl.status_code == 200
    assert hashlib.sha256(dl.content).hexdigest() == latest["sha256"]

    assert client.get("/api/v1/model/download/v99").status_code == 404
    assert client.get("/api/v1/model/download/bad_version").status_code == 400
    with pytest.raises(ValueError):
        resolve_tier1_onnx_path(app.state.session_factory, models_dir, "../evil")


def test_e2e_tier1_client_escalation_queue_flush_and_ota_hot_swap(tier2_env, tmp_path):
    app, client, server_models_dir = tier2_env
    edge_models_dir = tmp_path / "edge_models"
    edge_models_dir.mkdir(parents=True, exist_ok=True)
    edge_v1 = edge_models_dir / "tier1_v1.onnx"
    shutil.copy2(server_models_dir / "tier1_v1.onnx", edge_v1)

    edge = Tier1EdgeClient(
        edge_v1,
        tier2_url="http://testserver",
        http_client=client,
        queue_path=tmp_path / "offline_queue.json",
        models_dir=edge_models_dir,
    )

    # 1. Ambiguous URL escalates to Tier 2 and returns cloud verdict
    res = edge.evaluate_url("http://paypal-account.com/login")
    assert res["decision"] == "cloud"
    assert res["verdict_source"] == "tier2"
    assert isinstance(res["cloud_probability"], float)

    # 2. Simulate an offline queue item and flush it into Tier 2
    edge._enqueue({
        "url": "https://en.wikipedia.org/wiki/Phishing",
        "tier1_features": extract_lexical_features("https://en.wikipedia.org/wiki/Phishing"),
        "tier1_probability": 0.48,
        "tier1_model_version": "v1",
        "user_reported": False,
    })
    assert edge.pending_queue_size() == 1
    assert edge.flush_offline_queue() == 1
    assert edge.pending_queue_size() == 0
    assert client.get("/api/v1/stats").json()["total_scans"] == 2

    # 3. OTA sync is a no-op when both sides are at v1
    assert edge.sync_model() == {"updated": False, "previous_version": "v1", "version": "v1"}

    # 4. Publish tier1_v2.onnx on Tier 2 and verify live OTA hot-swap on Tier 1
    t1_model = train_tier1_model(load_stratified_splits().train)
    v2_server_path = export_tier1_onnx(t1_model, server_models_dir / "tier1_v2.onnx", version="v2")
    register_model_artifact(
        app.state.session_factory,
        tier="tier1",
        version="v2",
        path=v2_server_path,
        activate=True,
    )

    sync_res = edge.sync_model()
    assert sync_res == {"updated": True, "previous_version": "v1", "version": "v2"}
    assert edge.model_version == "v2"
    assert (edge_models_dir / "tier1_v2.onnx").is_file()
