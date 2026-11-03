"""Tests for the Tier 1 CLI (`tier1_edge/cli.py`) and Mobile/Edge Web Simulator (`tier1_edge/app.py`)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
import httpx

from shared.feature_schema import TIER1_FEATURES
from tier1_edge.app import PRESET_URLS, create_edge_app
from tier1_edge.cli import main as cli_main
from tier1_edge.inference_client import Tier1EdgeClient


def _offline_client(tmp_path):
    def unreachable(request):
        raise httpx.ConnectError("Tier 2 offline", request=request)

    http = httpx.Client(transport=httpx.MockTransport(unreachable), base_url="http://tier2.test")
    return Tier1EdgeClient(
        "models/tier1_v1.onnx",
        tier2_url="http://tier2.test",
        http_client=http,
        queue_path=tmp_path / "offline_queue.json",
    )


def test_cli_json_output_for_benign_and_phishing_urls(capsys, tmp_path):
    client = _offline_client(tmp_path)

    rc = cli_main(["https://www.google.com/", "--json"], client=client)
    assert rc == 0
    benign = json.loads(capsys.readouterr().out)
    assert benign["verdict"] == "legitimate"
    assert benign["decision"] == "local"
    assert benign["p_phishing"] <= 0.35

    rc = cli_main(
        ["http://192.168.10.5:8080/paypal/login@verify//secure", "--json"],
        client=client,
    )
    assert rc == 0
    phish = json.loads(capsys.readouterr().out)
    assert phish["verdict"] == "phishing"
    assert phish["decision"] == "local"
    assert phish["p_phishing"] >= 0.65


def test_cli_human_readable_output_shows_features_and_latency(capsys, tmp_path):
    client = _offline_client(tmp_path)
    rc = cli_main(["https://www.dlsu.edu.ph/"], client=client)
    out = capsys.readouterr().out
    assert rc == 0
    assert "LEGITIMATE" in out
    assert "having_IP_Address" in out
    assert "ms" in out


def test_simulator_ui_and_api_endpoints(tmp_path):
    edge_client = _offline_client(tmp_path)
    app = create_edge_app(edge_client)
    tc = TestClient(app)

    home = tc.get("/")
    assert home.status_code == 200
    assert "Knight of the Site" in home.text
    assert "Tier 1" in home.text

    status = tc.get("/api/status").json()
    assert status["model_version"] == "v1"
    assert status["low_thresh"] == 0.35
    assert status["high_thresh"] == 0.65
    assert len(status["presets"]) == len(PRESET_URLS)

    # 1. Benign local resolution
    r_benign = tc.post("/api/evaluate", json={"url": "https://www.google.com/"})
    assert r_benign.status_code == 200
    b_data = r_benign.json()
    assert b_data["decision"] == "local"
    assert b_data["verdict"] == "legitimate"
    assert list(b_data["features"]) == TIER1_FEATURES

    # 2. Phishing local block
    r_phish = tc.post(
        "/api/evaluate",
        json={"url": "http://192.168.10.5:8080/paypal/login@verify//secure"},
    )
    assert r_phish.status_code == 200
    p_data = r_phish.json()
    assert p_data["decision"] == "local"
    assert p_data["verdict"] == "phishing"

    # 3. Ambiguous URL escalates (and queues offline when Tier 2 is unreachable)
    r_ambig = tc.post("/api/evaluate", json={"url": "http://paypal-account.com/login"})
    assert r_ambig.status_code == 200
    a_data = r_ambig.json()
    assert a_data["decision"] == "queued_offline"
    assert a_data["pending_queue_size"] == 1

    # 4. Invalid URL returns 422
    r_bad = tc.post("/api/evaluate", json={"url": "http://"})
    assert r_bad.status_code == 422
