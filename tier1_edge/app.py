"""Tier 1 Mobile/Edge Web Simulator UI and REST endpoints (`tier1_edge/app.py`).

Run locally:
    uvicorn tier1_edge.app:app --reload --port 8501
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from tier1_edge.inference_client import DEFAULT_TIER2_URL, ModelSyncError, Tier1EdgeClient

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "tier1_v1.onnx"

PRESET_URLS: list[dict[str, str]] = [
    {
        "category": "benign",
        "label": "Benign — Google Search",
        "url": "https://www.google.com/",
        "expected": "Local Legitimate (P <= 0.35)",
    },
    {
        "category": "benign",
        "label": "Benign — DLSU Portal",
        "url": "https://www.dlsu.edu.ph/",
        "expected": "Local Legitimate (P <= 0.35)",
    },
    {
        "category": "ambiguous",
        "label": "Ambiguous — Hyphenated Login",
        "url": "http://paypal-account.com/login",
        "expected": "Escalate to Tier 2 (0.35 < P < 0.65)",
    },
    {
        "category": "ambiguous",
        "label": "Ambiguous — Subdomain Article",
        "url": "https://en.wikipedia.org/wiki/Phishing",
        "expected": "Escalate to Tier 2 (0.35 < P < 0.65)",
    },
    {
        "category": "phishing",
        "label": "Phishing — IP + Port 8080 + @ + //",
        "url": "http://192.168.10.5:8080/paypal/login@verify//secure",
        "expected": "Local Phishing Block (P >= 0.65)",
    },
    {
        "category": "phishing",
        "label": "Phishing — HTTPS Token Spoof",
        "url": "http://https-www-paypal-it-webapps-mpp-home.soft-hair.com/",
        "expected": "Local Phishing Block (P >= 0.65)",
    },
]


class EvaluateRequest(BaseModel):
    url: str
    user_reported: bool = False


_SIMULATOR_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Knight of the Site — Tier 1 Mobile/Edge Simulator</title>
  <style>
    :root {
      --bg: #0f172a;
      --surface: #1e293b;
      --surface-alt: #0b1120;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #0ea5e9;
      --legit: #10b981;
      --ambig: #f59e0b;
      --phish: #ef4444;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }
    .container {
      max-width: 1080px;
      margin: 0 auto;
      padding: 1.5rem 1rem 3rem;
    }
    header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
      padding-bottom: 1rem;
      border-bottom: 1px solid var(--border);
      margin-bottom: 1.5rem;
    }
    h1 { font-size: 1.35rem; margin: 0; letter-spacing: -0.02em; }
    .subtitle { color: var(--muted); font-size: 0.875rem; margin: 0.2rem 0 0; }
    .telemetry-pills { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; }
    .pill {
      background: var(--surface);
      border: 1px solid var(--border);
      padding: 0.35rem 0.75rem;
      border-radius: 6px;
      font-size: 0.8125rem;
      font-variant-numeric: tabular-nums;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 1.25rem;
    }
    @media (min-width: 820px) {
      .grid { grid-template-columns: 1.1fr 0.9fr; }
    }
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1.25rem;
    }
    .card h2 {
      font-size: 0.95rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
      margin: 0 0 1rem;
    }
    label { display: block; font-size: 0.875rem; font-weight: 600; margin-bottom: 0.4rem; }
    .url-row { display: flex; gap: 0.5rem; }
    input[type="text"] {
      flex: 1;
      padding: 0.65rem 0.75rem;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: var(--surface-alt);
      color: var(--text);
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      font-size: 0.875rem;
    }
    button {
      cursor: pointer;
      padding: 0.6rem 1rem;
      border-radius: 6px;
      border: 1px solid transparent;
      font-weight: 600;
      font-size: 0.875rem;
      background: var(--accent);
      color: #082f49;
    }
    button.secondary {
      background: var(--surface-alt);
      color: var(--text);
      border-color: var(--border);
      font-weight: 500;
      font-size: 0.8125rem;
      padding: 0.45rem 0.75rem;
    }
    button:focus-visible, input:focus-visible {
      outline: 2px solid var(--accent);
      outline-offset: 2px;
    }
    .options-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 0.75rem;
      font-size: 0.8125rem;
      color: var(--muted);
    }
    .presets {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 0.5rem;
      margin-top: 1rem;
    }
    .preset-btn {
      text-align: left;
      background: var(--surface-alt);
      color: var(--text);
      border: 1px solid var(--border);
      padding: 0.55rem 0.75rem;
    }
    .preset-btn small { display: block; color: var(--muted); font-weight: 400; font-size: 0.75rem; }
    .verdict-banner {
      padding: 1rem;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: var(--surface-alt);
      margin-bottom: 1rem;
    }
    .verdict-banner.legitimate { border-left: 4px solid var(--legit); }
    .verdict-banner.phishing { border-left: 4px solid var(--phish); }
    .verdict-banner.escalated { border-left: 4px solid var(--ambig); }
    .verdict-title { font-size: 1.15rem; font-weight: 700; margin: 0 0 0.25rem; }
    .verdict-meta { font-size: 0.8125rem; color: var(--muted); font-variant-numeric: tabular-nums; }
    .gate-bar {
      position: relative;
      height: 12px;
      border-radius: 6px;
      background: linear-gradient(
        to right,
        #065f46 0%, #065f46 35%,
        #78350f 35%, #78350f 65%,
        #7f1d1d 65%, #7f1d1d 100%
      );
      margin: 0.75rem 0 0.35rem;
    }
    .gate-marker {
      position: absolute;
      top: -4px;
      width: 4px;
      height: 20px;
      background: #fff;
      border-radius: 2px;
      transform: translateX(-50%);
    }
    .gate-labels {
      display: flex;
      justify-content: space-between;
      font-size: 0.75rem;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }
    table { width: 100%; border-collapse: collapse; font-size: 0.8125rem; }
    th, td {
      text-align: left;
      padding: 0.45rem 0.5rem;
      border-bottom: 1px solid var(--border);
      font-variant-numeric: tabular-nums;
    }
    .badge {
      display: inline-block;
      padding: 0.1rem 0.45rem;
      border-radius: 4px;
      font-weight: 600;
      font-size: 0.75rem;
    }
    .badge.v1 { background: rgba(16, 185, 129, 0.18); color: #6ee7b7; }
    .badge.v0 { background: rgba(245, 158, 11, 0.2); color: #fcd34d; }
    .badge.v-1 { background: rgba(239, 68, 68, 0.2); color: #fca5a5; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>Knight of the Site — Tier 1 Mobile/Edge Simulator</h1>
        <p class="subtitle">Zero-Network 9-Feature Lexical Extractor + Local ONNX Runtime (&lt; 10 ms)</p>
      </div>
      <div class="telemetry-pills">
        <span class="pill" id="model-pill">ONNX Model: v1</span>
        <span class="pill" id="queue-pill">Offline Queue: 0</span>
        <button class="secondary" type="button" id="sync-btn">Sync OTA Model</button>
        <button class="secondary" type="button" id="flush-btn">Flush Queue</button>
      </div>
    </header>

    <main class="grid">
      <section class="card" aria-labelledby="scan-heading">
        <h2 id="scan-heading">1. Edge Address-Bar Scanner</h2>
        <form id="scan-form">
          <label for="url-input">Target URL</label>
          <div class="url-row">
            <input id="url-input" type="text" required placeholder="https://www.example.com/" value="https://www.google.com/" />
            <button type="submit">Scan URL</button>
          </div>
          <div class="options-row">
            <label style="font-weight:400; margin:0;">
              <input id="report-check" type="checkbox" /> User-reported suspicious link (force Tier 2 escalation)
            </label>
            <span id="status-msg" role="status" aria-live="polite"></span>
          </div>
        </form>

        <h2 style="margin-top:1.5rem;">Preset Classroom Scenarios</h2>
        <div class="presets" id="presets-box"></div>

        <div style="margin-top:1.25rem;">
          <h2>Confidence Gate Routing ([0.35, 0.65])</h2>
          <div class="gate-bar" aria-hidden="true">
            <div class="gate-marker" id="gate-marker" style="left: 25.9%;"></div>
          </div>
          <div class="gate-labels">
            <span>0.00 Local Legitimate (&le; 0.35)</span>
            <span>0.35–0.65 Escalate Tier 2</span>
            <span>Local Phishing (&ge; 0.65) 1.00</span>
          </div>
        </div>
      </section>

      <section class="card" aria-labelledby="result-heading">
        <h2 id="result-heading">2. Verdict &amp; 9 Lexical Features</h2>
        <div id="verdict-box" class="verdict-banner legitimate" role="status" aria-live="polite">
          <p class="verdict-title" id="verdict-title">Ready to Scan</p>
          <p class="verdict-meta" id="verdict-meta">Select a preset or submit a URL to run local ONNX inference.</p>
        </div>

        <table aria-label="Extracted Tier 1 lexical features">
          <thead>
            <tr>
              <th>Feature Name</th>
              <th>Value</th>
              <th>Signal</th>
            </tr>
          </thead>
          <tbody id="features-tbody"></tbody>
        </table>
      </section>
    </main>
  </div>

  <script>
    const SIGNAL_MAP = {
      "1": { label: "Legitimate (+1)", cls: "v1" },
      "0": { label: "Suspicious (0)", cls: "v0" },
      "-1": { label: "Phishing (-1)", cls: "v-1" }
    };

    async function refreshStatus() {
      const res = await fetch("/api/status");
      const data = await res.json();
      document.getElementById("model-pill").textContent = "ONNX Model: " + data.model_version;
      document.getElementById("queue-pill").textContent = "Offline Queue: " + data.pending_queue_size;
      const box = document.getElementById("presets-box");
      if (!box.children.length) {
        data.presets.forEach(p => {
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "preset-btn";
          btn.innerHTML = "<strong>" + p.label + "</strong><small>" + p.expected + "</small>";
          btn.addEventListener("click", () => {
            document.getElementById("url-input").value = p.url;
            evaluateCurrentUrl();
          });
          box.appendChild(btn);
        });
      }
    }

    function renderResult(r) {
      const box = document.getElementById("verdict-box");
      const isEscalated = r.decision !== "local";
      box.className = "verdict-banner " + (isEscalated ? "escalated" : r.verdict);
      const decisionLabel = {
        local: "Resolved Locally on Tier 1 Edge",
        cloud: "Escalated & Verified by Tier 2 Cloud",
        queued_offline: "Uncertain (Escalation Queued — Tier 2 Offline)"
      }[r.decision] || r.decision;

      document.getElementById("verdict-title").textContent =
        r.verdict.toUpperCase() + " — " + decisionLabel;
      document.getElementById("verdict-meta").textContent =
        "P(phishing) = " + r.p_phishing.toFixed(3) +
        " | extract = " + r.extract_ms.toFixed(2) + " ms" +
        " | onnx = " + r.inference_ms.toFixed(2) + " ms" +
        " | model = " + r.model_version;

      document.getElementById("gate-marker").style.left = Math.min(100, Math.max(0, r.p_phishing * 100)) + "%";
      document.getElementById("model-pill").textContent = "ONNX Model: " + r.model_version;
      document.getElementById("queue-pill").textContent = "Offline Queue: " + r.pending_queue_size;

      const tbody = document.getElementById("features-tbody");
      tbody.innerHTML = "";
      Object.entries(r.features).forEach(([name, val]) => {
        const sig = SIGNAL_MAP[String(val)] || { label: String(val), cls: "v0" };
        const tr = document.createElement("tr");
        tr.innerHTML =
          "<td><code>" + name + "</code></td>" +
          "<td>" + val + "</td>" +
          "<td><span class='badge " + sig.cls + "'>" + sig.label + "</span></td>";
        tbody.appendChild(tr);
      });
    }

    async function evaluateCurrentUrl() {
      const url = document.getElementById("url-input").value.trim();
      const userReported = document.getElementById("report-check").checked;
      const msg = document.getElementById("status-msg");
      msg.textContent = "Scanning...";
      const res = await fetch("/api/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, user_reported: userReported })
      });
      const body = await res.json();
      if (!res.ok) {
        msg.textContent = body.detail || "Invalid URL";
        return;
      }
      msg.textContent = "";
      renderResult(body);
    }

    document.getElementById("scan-form").addEventListener("submit", e => {
      e.preventDefault();
      evaluateCurrentUrl();
    });

    document.getElementById("sync-btn").addEventListener("click", async () => {
      const msg = document.getElementById("status-msg");
      msg.textContent = "Checking Tier 2 for OTA model...";
      const res = await fetch("/api/sync-model", { method: "POST" });
      const body = await res.json();
      msg.textContent = res.ok
        ? (body.updated ? "Updated to " + body.version : "Model " + body.version + " is up to date")
        : (body.detail || "Tier 2 unreachable");
      await refreshStatus();
    });

    document.getElementById("flush-btn").addEventListener("click", async () => {
      const res = await fetch("/api/flush-queue", { method: "POST" });
      const body = await res.json();
      document.getElementById("status-msg").textContent =
        "Delivered " + body.delivered + " queued item(s)";
      await refreshStatus();
    });

    refreshStatus().then(evaluateCurrentUrl);
  </script>
</body>
</html>
"""


def create_edge_app(client: Tier1EdgeClient | None = None) -> FastAPI:
    """Create the FastAPI application serving the Tier 1 Mobile/Edge Simulator."""
    edge = client or Tier1EdgeClient(
        os.environ.get("TIER1_ONNX_PATH", str(DEFAULT_MODEL_PATH)),
        tier2_url=os.environ.get("TIER2_URL", DEFAULT_TIER2_URL),
    )
    app = FastAPI(title="Knight of the Site — Tier 1 Mobile/Edge Simulator", version="1.0.0")
    app.state.edge_client = edge

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(_SIMULATOR_HTML)

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return {
            "model_version": edge.model_version,
            "low_thresh": edge.low_thresh,
            "high_thresh": edge.high_thresh,
            "tier2_url": edge.tier2_url,
            "pending_queue_size": edge.pending_queue_size(),
            "presets": PRESET_URLS,
        }

    @app.post("/api/evaluate")
    def evaluate(req: EvaluateRequest) -> dict[str, Any]:
        try:
            result = edge.evaluate_url(req.url, user_reported=req.user_reported)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {**result, "pending_queue_size": edge.pending_queue_size()}

    @app.post("/api/sync-model")
    def sync_model() -> dict[str, Any]:
        try:
            return edge.sync_model()
        except ModelSyncError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/api/flush-queue")
    def flush_queue() -> dict[str, Any]:
        delivered = edge.flush_offline_queue()
        return {"delivered": delivered, "pending_queue_size": edge.pending_queue_size()}

    return app


app = create_edge_app()
