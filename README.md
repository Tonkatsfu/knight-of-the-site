# Knight of the Site: Multi-Tier Phishing URL Detection System

**Course:** CEMCLRN / MAC-LRN (1T2627) — Multi-Tier Machine Learning System  

## Overview

**Knight of the Site** is an end-to-end, two-tier machine learning system that combines fast, privacy-preserving edge URL classification with cloud-based deep feature analysis and continual retraining:

- **System Tier 1 (Mobile / Edge Client):** Extracts **9 zero-network Address-Bar lexical features** directly from a URL string in `< 1 ms` and runs a quantized **ONNX Runtime** classifier locally (`< 10 ms` latency). Confident predictions (`P(phishing) <= 0.35` or `>= 0.65`) are resolved locally on the device—saving ~85–90% of bandwidth and keeping browsing history private.
- **System Tier 2 (Cloud Server):** A containerized **FastAPI + PostgreSQL** server that ingests uncertain edge URLs (`0.35 < P(phishing) < 0.65`) and user-reported links, extracts **21 heavy network/SSL/WHOIS/DOM/traffic features**, scores all **30 features** with a higher-capacity classifier, stores telemetry, and executes a **continual retraining pipeline** that compiles updated `.onnx` weights for Over-The-Air (OTA) deployment back to Tier 1.

## Repository Structure

```text
├── shared/                  # Unified 30-feature schema, ARFF loader & lexical extractor
│   ├── feature_schema.py    # Canonical 9 Tier-1 and 21 Tier-2 feature definitions
│   └── data_loader.py       # UCI Phishing Websites ARFF parser & stratified splitter
├── tier1_edge/              # System Tier 1: Local ONNX Runtime client, CLI & Mobile/Edge UI
├── tier2_cloud/             # System Tier 2: FastAPI server, PostgreSQL ORM, heavy extractor & retrainer
├── scripts/                 # Model training, benchmarking & live demo walkthrough scripts
├── models/                  # Versioned ONNX (.onnx) and cloud (.joblib) model artifacts
├── reports/                 # Benchmark metrics, figures & ACM conference report
└── tests/                   # Unit and integration test suite (pytest)
```

## Quick Start (Phase 1 & Phase 2 Verification)

1. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
2. (Optional) Rebuild the 11,055 real-URL lexical corpus (`phishing+websites/tier1_urls.csv`) and retrain the initial models (`models/tier1_v1.onnx` and `models/tier2_v1.joblib`):
   ```powershell
   python scripts/prepare_tier1_urls.py
   python scripts/train_initial_models.py
   ```
3. Run the Phase 1 & Phase 2 test suite (83 tests):
   ```powershell
   python -m pytest tests/ -v
   ```
4. Run the Tier 1 Edge CLI or launch the Mobile/Edge Web Simulator UI:
   ```powershell
   python -m tier1_edge.cli https://www.google.com/
   python -m uvicorn tier1_edge.app:app --reload --port 8501
   ```
