"""Unit tests for the Tier 2 21-feature heavy extractor (`tier2_cloud/heavy_extractor.py`).

All tests run offline using injected probe callables and synthetic HTML/WHOIS/TLS fixtures.
"""

from datetime import datetime, timedelta, timezone
import time

import pytest

from shared.feature_schema import ALLOWED_FEATURE_VALUES, TIER2_HEAVY_FEATURES, validate_feature_vector
from tier2_cloud.heavy_extractor import (
    HEAVY_FEATURE_MODES,
    HeavyFeatureExtractor,
    HttpFetchResult,
    TlsProbeResult,
    extract_html_features,
    score_reputation,
    score_tls_info,
    score_whois_record,
)

BENIGN_HTML = """
<!doctype html>
<html>
  <head>
    <link rel="icon" href="/favicon.ico" />
    <link rel="stylesheet" href="/static/site.css" />
    <script src="/static/app.js"></script>
  </head>
  <body>
    <img src="/logo.png" />
    <a href="/about">About</a>
    <a href="/contact">Contact</a>
    <a href="/help">Help</a>
    <form action="/login" method="post">
      <input type="text" name="user" />
    </form>
  </body>
</html>
"""

PHISHING_HTML = """
<!doctype html>
<html>
  <head>
    <link rel="shortcut icon" href="http://evil-cdn.ru/ico.png" />
    <link rel="stylesheet" href="http://evil-cdn.ru/a.css" />
    <script src="http://evil-cdn.ru/b.js"></script>
  </head>
  <body oncontextmenu="return false">
    <img src="http://evil-cdn.ru/1.png" />
    <img src="http://evil-cdn.ru/2.png" />
    <a href="#" onmouseover="window.status='https://paypal.com';return true">Login</a>
    <a href="javascript:void(0)">Verify</a>
    <form action="about:blank" method="post">
      <a href="mailto:drop@evil-cdn.ru">Submit</a>
    </form>
    <iframe src="http://evil-cdn.ru/frame" frameborder="0"></iframe>
    <script>
      if (event.button == 2) { prompt("Enter password"); }
    </script>
  </body>
</html>
"""


def test_heavy_feature_modes_are_complete_and_valid():
    assert list(HEAVY_FEATURE_MODES) == TIER2_HEAVY_FEATURES
    vec = validate_feature_vector(HEAVY_FEATURE_MODES, TIER2_HEAVY_FEATURES, strict_domain=True)
    assert vec.shape == (21,)


def test_extract_html_features_benign_page():
    feats = extract_html_features(BENIGN_HTML, "https://www.dlsu.edu.ph/home", redirect_count=0)
    assert feats["Favicon"] == 1
    assert feats["Request_URL"] == 1
    assert feats["URL_of_Anchor"] == 1
    assert feats["Links_in_tags"] == 1
    assert feats["SFH"] == 1
    assert feats["Submitting_to_email"] == 1
    assert feats["Redirect"] == 0
    assert feats["on_mouseover"] == 1
    assert feats["RightClick"] == 1
    assert feats["popUpWidnow"] == 1
    assert feats["Iframe"] == 1
    assert feats["Links_pointing_to_page"] == 1
    for name, val in feats.items():
        assert val in ALLOWED_FEATURE_VALUES[name]


def test_extract_html_features_phishing_page():
    feats = extract_html_features(PHISHING_HTML, "http://paypal-secure-login.com/", redirect_count=3)
    assert feats["Favicon"] == -1
    assert feats["Request_URL"] == -1
    assert feats["URL_of_Anchor"] == -1
    assert feats["Links_in_tags"] == -1
    assert feats["SFH"] == -1
    assert feats["Submitting_to_email"] == -1
    assert feats["Redirect"] == 1
    assert feats["on_mouseover"] == -1
    assert feats["RightClick"] == -1
    assert feats["popUpWidnow"] == -1
    assert feats["Iframe"] == -1
    for name, val in feats.items():
        assert val in ALLOWED_FEATURE_VALUES[name]


def test_score_tls_info_states():
    assert score_tls_info(TlsProbeResult(has_tls=False, trusted_issuer=False)) == -1
    assert score_tls_info(TlsProbeResult(has_tls=True, trusted_issuer=False, cert_age_days=400)) == 0
    assert score_tls_info(TlsProbeResult(has_tls=True, trusted_issuer=True, cert_age_days=90)) == 0
    assert score_tls_info(TlsProbeResult(has_tls=True, trusted_issuer=True, cert_age_days=365)) == 1


def test_score_whois_record_established_vs_throwaway():
    now = datetime(2026, 11, 7, tzinfo=timezone.utc)
    established = {
        "domain_name": ["GOOGLE.COM", "google.com"],
        "creation_date": now - timedelta(days=3650),
        "expiration_date": now + timedelta(days=730),
    }
    assert score_whois_record(established, "www.google.com", now=now) == {
        "Domain_registeration_length": 1,
        "Abnormal_URL": 1,
        "age_of_domain": 1,
    }

    fresh = {
        "domain_name": "other-domain.xyz",
        "creation_date": now - timedelta(days=10),
        "expiration_date": now + timedelta(days=90),
    }
    assert score_whois_record(fresh, "login-paypal.xyz", now=now) == {
        "Domain_registeration_length": -1,
        "Abnormal_URL": -1,
        "age_of_domain": -1,
    }


def test_score_reputation_blocklist_and_ranking():
    blocked = score_reputation("bad-login.org", blocklist_hosts=frozenset({"bad-login.org"}))
    assert blocked == {
        "web_traffic": -1,
        "Page_Rank": -1,
        "Google_Index": -1,
        "Statistical_report": -1,
    }

    ranked = score_reputation("www.google.com")
    assert ranked["web_traffic"] == 1
    assert ranked["Page_Rank"] == 1
    assert ranked["Google_Index"] == 1
    assert ranked["Statistical_report"] == 1


def test_orchestrator_combines_all_21_features_when_probes_succeed():
    now = datetime.now(timezone.utc)
    extractor = HeavyFeatureExtractor(
        budget_s=1.0,
        blocklist_path=None,
        tranco_path=None,
        http_fetcher=lambda url, t: HttpFetchResult(html=BENIGN_HTML, redirect_count=0, final_url=url),
        tls_checker=lambda host, port, t: TlsProbeResult(has_tls=True, trusted_issuer=True, cert_age_days=365),
        dns_resolver=lambda host, t: True,
        whois_lookup=lambda host, t: {
            "domain_name": "google.com",
            "creation_date": now - timedelta(days=1000),
            "expiration_date": now + timedelta(days=500),
        },
    )
    res = extractor.extract("https://www.google.com/")
    assert list(res.features) == TIER2_HEAVY_FEATURES
    assert all(v == "ok" for v in res.status.values())
    assert res.features["SSLfinal_State"] == 1
    assert res.features["DNSRecord"] == 1
    assert res.features["age_of_domain"] == 1


def test_orchestrator_imputes_modes_and_respects_budget_when_probes_hang():
    def slow_whois(host: str, timeout_s: float):
        time.sleep(2.0)
        return {}

    def failing_http(url: str, timeout_s: float):
        raise RuntimeError("connection reset")

    extractor = HeavyFeatureExtractor(
        budget_s=0.15,
        blocklist_path=None,
        tranco_path=None,
        http_fetcher=failing_http,
        dns_resolver=lambda host, t: True,
        whois_lookup=slow_whois,
    )
    t0 = time.perf_counter()
    res = extractor.extract("http://ambiguous-example.com/login")
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.6  # Stopped at the budget instead of waiting 2.0 s for slow_whois
    assert res.status["Domain_registeration_length"] == "timeout"
    assert res.status["Favicon"] == "error"
    assert res.status["DNSRecord"] == "ok"
    assert res.status["SSLfinal_State"] == "ok"
    assert res.features["SSLfinal_State"] == -1  # plain HTTP
    assert res.features["Domain_registeration_length"] == HEAVY_FEATURE_MODES["Domain_registeration_length"]
    assert res.features["Favicon"] == HEAVY_FEATURE_MODES["Favicon"]
    validate_feature_vector(res.features, TIER2_HEAVY_FEATURES, strict_domain=True)
