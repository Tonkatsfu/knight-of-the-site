"""Tier 2 heavy feature extractor (21 network, SSL, WHOIS, DOM, and reputation features).

Architecture:
    * Pure parsing functions (`extract_html_features`, `score_whois_record`, `score_tls_info`,
      `score_reputation`) convert fetched artifacts into the exact {-1, 0, 1} domain values
      required by `shared.feature_schema.ALLOWED_FEATURE_VALUES`.
    * `HeavyFeatureExtractor` runs the network probes (HTTP GET, TLS handshake, DNS lookup,
      WHOIS query) concurrently under a hard wall-clock budget (`extraction_budget_s`, default
      2.5 s so the whole `/api/v1/ingest` call finishes inside Tier 1's 3.0 s HTTP timeout).
    * Any probe that times out or raises an error falls back to the training-set mode in
      `HEAVY_FEATURE_MODES` and records `"timeout"` or `"error"` in `status[feature_name]`.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import re
import socket
import ssl
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup
import httpx

from shared.feature_schema import ALLOWED_FEATURE_VALUES, TIER2_HEAVY_FEATURES, validate_feature_vector
from shared.lexical_extractor import LEGITIMATE, PHISHING, SUSPICIOUS, _is_ip_host, _normalize_url
from tier2_cloud.config import DEFAULT_BLOCKLIST_PATH, DEFAULT_EXTRACTION_BUDGET_S, DEFAULT_TRANCO_PATH

# Empirical modes of the 21 heavy features on the 70% UCI training split (used when a probe fails/times out)
HEAVY_FEATURE_MODES: dict[str, int] = {
    "SSLfinal_State": 1,
    "Domain_registeration_length": -1,
    "Favicon": 1,
    "Request_URL": 1,
    "URL_of_Anchor": 0,
    "Links_in_tags": 0,
    "SFH": -1,
    "Submitting_to_email": 1,
    "Abnormal_URL": 1,
    "Redirect": 0,
    "on_mouseover": 1,
    "RightClick": 1,
    "popUpWidnow": 1,
    "Iframe": 1,
    "age_of_domain": 1,
    "DNSRecord": 1,
    "web_traffic": 1,
    "Page_Rank": -1,
    "Google_Index": 1,
    "Links_pointing_to_page": 0,
    "Statistical_report": 1,
}

HTML_FEATURES: tuple[str, ...] = (
    "Favicon",
    "Request_URL",
    "URL_of_Anchor",
    "Links_in_tags",
    "SFH",
    "Submitting_to_email",
    "Redirect",
    "on_mouseover",
    "RightClick",
    "popUpWidnow",
    "Iframe",
    "Links_pointing_to_page",
)
WHOIS_FEATURES: tuple[str, ...] = (
    "Domain_registeration_length",
    "Abnormal_URL",
    "age_of_domain",
)
REPUTATION_FEATURES: tuple[str, ...] = (
    "web_traffic",
    "Page_Rank",
    "Google_Index",
    "Statistical_report",
)

# Built-in fallback top domains when no external Tranco CSV is present on disk
BUILTIN_HIGH_TRAFFIC_DOMAINS: frozenset[str] = frozenset({
    "google.com", "youtube.com", "facebook.com", "wikipedia.org", "github.com",
    "microsoft.com", "apple.com", "amazon.com", "linkedin.com", "reddit.com",
    "cloudflare.com", "mozilla.org", "python.org", "dlsu.edu.ph", "edu.ph",
    "gov.ph", "nytimes.com", "bbc.co.uk", "stackoverflow.com", "paypal.com",
})

_MOUSEOVER_STATUS_RE = re.compile(r"onmouseover\s*=\s*['\"][^'\"]*window\.status", re.IGNORECASE)
_RIGHTCLICK_RE = re.compile(
    r"(?:event\.button\s*==\s*2|oncontextmenu\s*=\s*['\"][^'\"]*return\s+false)",
    re.IGNORECASE,
)
_POPUP_PROMPT_RE = re.compile(r"(?:window\.open\s*\([^)]*\)|prompt\s*\()", re.IGNORECASE)
_MAILTO_RE = re.compile(r"(?:mailto:|mail\s*\()", re.IGNORECASE)


@dataclass(frozen=True)
class HttpFetchResult:
    """Result of fetching a URL over HTTP(S)."""

    html: str
    redirect_count: int
    final_url: str


@dataclass(frozen=True)
class TlsProbeResult:
    """Result of inspecting a host's TLS certificate."""

    has_tls: bool
    trusted_issuer: bool
    cert_age_days: int = 365


@dataclass(frozen=True)
class HeavyExtractionResult:
    """21 heavy features (ordered by `TIER2_HEAVY_FEATURES`), per-feature status, and elapsed ms."""

    features: dict[str, int]
    status: dict[str, str]
    extraction_ms: float


def _registrable_domain(host: str) -> str:
    """Best-effort registrable suffix for comparing whether two hosts belong to the same site."""
    host = host.lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in {"co", "ac", "edu", "gov", "org", "net", "com"}:
        return ".".join(parts[-3:])
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


def _is_same_domain(target_url: str, page_url: str, page_domain: str) -> bool:
    """Return True when `target_url` is relative or resolves to the same registrable domain as `page_url`."""
    stripped = target_url.strip()
    if not stripped or stripped.startswith("data:"):
        return True
    raw_parts = urlsplit(stripped)
    if raw_parts.scheme and raw_parts.scheme.lower() not in {"http", "https"}:
        return False
    resolved = urljoin(page_url, stripped)
    host = urlsplit(resolved).hostname
    if not host:
        return True
    return _registrable_domain(host) == page_domain


# ------------------------------------------------------------------ Pure feature rules


def extract_html_features(html: str, page_url: str, *, redirect_count: int = 0) -> dict[str, int]:
    """Compute the 12 HTTP/DOM/JavaScript heavy features from fetched HTML and redirect count."""
    normalized_url = _normalize_url(page_url)
    page_host = urlsplit(normalized_url).hostname or ""
    page_domain = _registrable_domain(page_host)
    soup = BeautifulSoup(html or "", "html.parser")

    # 1. Redirect ({0, 1} in Training Dataset.arff: 0 = legitimate <=1 hop, 1 = >=2 hops)
    redirect_val = 0 if redirect_count <= 1 else 1

    # 2. Favicon ({-1, 1})
    favicon_val = LEGITIMATE
    for link in soup.find_all("link"):
        rels = [r.lower() for r in (link.get("rel") or [])]
        if any("icon" in r for r in rels):
            href = str(link.get("href") or "")
            if href and not _is_same_domain(href, normalized_url, page_domain):
                favicon_val = PHISHING
                break

    # 3. Request_URL ({-1, 1} in Training Dataset.arff: 1 if <22% external media, else -1)
    media_urls = [
        str(tag.get("src"))
        for tag in soup.find_all(["img", "audio", "embed", "iframe", "video", "source"])
        if tag.get("src")
    ]
    if not media_urls:
        request_url_val = LEGITIMATE
    else:
        ext_media = sum(1 for u in media_urls if not _is_same_domain(u, normalized_url, page_domain))
        request_url_val = LEGITIMATE if (ext_media / len(media_urls)) < 0.22 else PHISHING

    # 4. URL_of_Anchor ({-1, 0, 1}: <0.31 -> 1, <=0.67 -> 0, >0.67 -> -1)
    anchors = soup.find_all("a")
    if not anchors:
        anchor_val = LEGITIMATE
    else:
        unsafe = 0
        for a in anchors:
            href = str(a.get("href") or "").strip()
            low = href.lower()
            if (
                not href
                or low in {"#", "#content", "#skip", "javascript:void(0)", "javascript:void(0);"}
                or low.startswith("javascript:")
                or not _is_same_domain(href, normalized_url, page_domain)
            ):
                unsafe += 1
        ratio = unsafe / len(anchors)
        if ratio < 0.31:
            anchor_val = LEGITIMATE
        elif ratio <= 0.67:
            anchor_val = SUSPICIOUS
        else:
            anchor_val = PHISHING

    # 5. Links_in_tags ({-1, 0, 1}: <0.17 -> 1, <=0.81 -> 0, >0.81 -> -1)
    tag_urls: list[str] = []
    for tag in soup.find_all(["meta", "script", "link"]):
        candidate = tag.get("src") or tag.get("href") or tag.get("content")
        if candidate and ("http://" in str(candidate) or "https://" in str(candidate) or str(candidate).startswith("/")):
            tag_urls.append(str(candidate))
    if not tag_urls:
        links_in_tags_val = LEGITIMATE
    else:
        ext_tags = sum(1 for u in tag_urls if not _is_same_domain(u, normalized_url, page_domain))
        tag_ratio = ext_tags / len(tag_urls)
        if tag_ratio < 0.17:
            links_in_tags_val = LEGITIMATE
        elif tag_ratio <= 0.81:
            links_in_tags_val = SUSPICIOUS
        else:
            links_in_tags_val = PHISHING

    # 6. SFH ({-1, 0, 1}: empty/about:blank -> -1, external domain -> 0, same domain/no form -> 1)
    forms = soup.find_all("form")
    sfh_val = LEGITIMATE
    for form in forms:
        action = str(form.get("action") or "").strip()
        if not action or action.lower() in {"about:blank", "#"}:
            sfh_val = PHISHING
            break
        if not _is_same_domain(action, normalized_url, page_domain):
            sfh_val = SUSPICIOUS

    # 7. Submitting_to_email ({-1, 1})
    submitting_email_val = PHISHING if _MAILTO_RE.search(html or "") else LEGITIMATE

    # 8. on_mouseover ({-1, 1})
    mouseover_val = PHISHING if _MOUSEOVER_STATUS_RE.search(html or "") else LEGITIMATE

    # 9. RightClick ({-1, 1})
    rightclick_val = PHISHING if _RIGHTCLICK_RE.search(html or "") else LEGITIMATE

    # 10. popUpWidnow ({-1, 1})
    popup_val = PHISHING if _POPUP_PROMPT_RE.search(html or "") else LEGITIMATE

    # 11. Iframe ({-1, 1}: invisible / borderless iframe -> -1, else 1)
    iframe_val = LEGITIMATE
    for frame in soup.find_all(["iframe", "frame"]):
        border = str(frame.get("frameborder") or "").strip().lower()
        style = str(frame.get("style") or "").lower()
        if border in {"0", "no"} or "border:0" in style or "border: none" in style or "display:none" in style:
            iframe_val = PHISHING
            break

    # 12. Links_pointing_to_page ({-1, 0, 1})
    total_links = len(anchors)
    if total_links == 0:
        links_pointing_val = PHISHING
    elif total_links <= 2:
        links_pointing_val = SUSPICIOUS
    else:
        links_pointing_val = LEGITIMATE

    return {
        "Favicon": favicon_val,
        "Request_URL": request_url_val,
        "URL_of_Anchor": anchor_val,
        "Links_in_tags": links_in_tags_val,
        "SFH": sfh_val,
        "Submitting_to_email": submitting_email_val,
        "Redirect": redirect_val,
        "on_mouseover": mouseover_val,
        "RightClick": rightclick_val,
        "popUpWidnow": popup_val,
        "Iframe": iframe_val,
        "Links_pointing_to_page": links_pointing_val,
    }


def score_tls_info(info: TlsProbeResult) -> int:
    """Convert a `TlsProbeResult` into `SSLfinal_State` in {-1, 0, 1}."""
    if not info.has_tls:
        return PHISHING
    if info.trusted_issuer and info.cert_age_days >= 365:
        return LEGITIMATE
    return SUSPICIOUS


def _first_datetime(value: Any) -> datetime | None:
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return None


def score_whois_record(record: dict[str, Any] | None, host: str, *, now: datetime | None = None) -> dict[str, int]:
    """Compute `Domain_registeration_length`, `Abnormal_URL`, and `age_of_domain` from WHOIS data."""
    if not record or _is_ip_host(host):
        return {
            "Domain_registeration_length": PHISHING,
            "Abnormal_URL": PHISHING,
            "age_of_domain": PHISHING,
        }

    ref_now = now or datetime.now(timezone.utc)
    created = _first_datetime(record.get("creation_date"))
    expires = _first_datetime(record.get("expiration_date"))
    raw_domain = record.get("domain_name")
    if isinstance(raw_domain, list):
        domain_names = [str(d).lower() for d in raw_domain if d]
    elif raw_domain:
        domain_names = [str(raw_domain).lower()]
    else:
        domain_names = []

    reg_domain = _registrable_domain(host)
    abnormal_val = (
        LEGITIMATE
        if any(reg_domain in d or d in host.lower() for d in domain_names)
        else PHISHING
    )
    reg_len_val = (
        LEGITIMATE
        if (expires is not None and (expires - ref_now).days >= 365)
        else PHISHING
    )
    age_val = (
        LEGITIMATE
        if (created is not None and (ref_now - created).days >= 180)
        else PHISHING
    )
    return {
        "Domain_registeration_length": reg_len_val,
        "Abnormal_URL": abnormal_val,
        "age_of_domain": age_val,
    }


@lru_cache(maxsize=4)
def _load_blocklist_hosts(csv_path_str: str) -> frozenset[str]:
    path = Path(csv_path_str)
    if not path.exists():
        return frozenset()
    hosts: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if int(row.get("label", 0)) == 1:
                try:
                    h = urlsplit(_normalize_url(row["url"])).hostname
                    if h:
                        hosts.add(h.lower())
                except ValueError:
                    continue
    return frozenset(hosts)


@lru_cache(maxsize=4)
def _load_tranco_ranks(csv_path_str: str) -> dict[str, int]:
    path = Path(csv_path_str)
    if not path.exists():
        return {}
    ranks: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2 and row[0].isdigit():
                ranks[row[1].strip().lower()] = int(row[0])
    return ranks


def score_reputation(
    host: str,
    *,
    blocklist_hosts: frozenset[str] = frozenset(),
    tranco_ranks: dict[str, int] | None = None,
) -> dict[str, int]:
    """Compute offline reputation features (`web_traffic`, `Page_Rank`, `Google_Index`, `Statistical_report`)."""
    lower_host = host.lower()
    reg_domain = _registrable_domain(lower_host)
    is_blocked = lower_host in blocklist_hosts or reg_domain in blocklist_hosts or _is_ip_host(lower_host)

    ranks = tranco_ranks or {}
    rank = ranks.get(lower_host) or ranks.get(reg_domain)
    is_builtin_top = reg_domain in BUILTIN_HIGH_TRAFFIC_DOMAINS or any(
        lower_host.endswith("." + suffix) for suffix in ("edu.ph", "gov.ph", "ac.uk", "edu")
    )

    if is_blocked:
        return {
            "web_traffic": PHISHING,
            "Page_Rank": PHISHING,
            "Google_Index": PHISHING,
            "Statistical_report": PHISHING,
        }

    if (rank is not None and rank <= 100_000) or is_builtin_top:
        traffic_val = LEGITIMATE
        pagerank_val = LEGITIMATE
    elif rank is not None and rank <= 1_000_000:
        traffic_val = SUSPICIOUS
        pagerank_val = PHISHING
    else:
        traffic_val = SUSPICIOUS
        pagerank_val = PHISHING

    return {
        "web_traffic": traffic_val,
        "Page_Rank": pagerank_val,
        "Google_Index": LEGITIMATE,
        "Statistical_report": LEGITIMATE,
    }


# ------------------------------------------------------------------ Default live probes


def _default_http_fetcher(url: str, timeout_s: float) -> HttpFetchResult:
    with httpx.Client(follow_redirects=True, timeout=timeout_s, verify=False) as client:
        resp = client.get(url, headers={"User-Agent": "KnightOfTheSite-Tier2/1.0"})
        return HttpFetchResult(
            html=resp.text[:200_000],
            redirect_count=len(resp.history),
            final_url=str(resp.url),
        )


def _default_tls_checker(host: str, port: int, timeout_s: float) -> TlsProbeResult:
    ctx = ssl.create_default_context()
    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        with ctx.wrap_socket(sock, server_hostname=host) as ssock:
            cert = ssock.getpeercert() or {}
            not_before = cert.get("notBefore")
            not_after = cert.get("notAfter")
            days = 365
            if isinstance(not_before, str) and isinstance(not_after, str):
                t0 = ssl.cert_time_to_seconds(not_before)
                t1 = ssl.cert_time_to_seconds(not_after)
                days = max(0, int((t1 - t0) // 86400))
            return TlsProbeResult(has_tls=True, trusted_issuer=bool(cert.get("issuer")), cert_age_days=days)


def _default_dns_resolver(host: str, timeout_s: float) -> bool:
    del timeout_s
    if _is_ip_host(host):
        return True
    socket.getaddrinfo(host, None)
    return True


def _default_whois_lookup(host: str, timeout_s: float) -> dict[str, Any]:
    import whois  # imported lazily so unit tests do not pay module import cost

    if _is_ip_host(host):
        return {}
    result = whois.whois(host, timeout=max(1, int(timeout_s)))
    return dict(result) if result else {}


# ------------------------------------------------------------------ Orchestrator


class HeavyFeatureExtractor:
    """Concurrent, time-budgeted extractor for the 21 Tier 2 heavy features."""

    def __init__(
        self,
        *,
        budget_s: float = DEFAULT_EXTRACTION_BUDGET_S,
        blocklist_path: str | Path | None = DEFAULT_BLOCKLIST_PATH,
        tranco_path: str | Path | None = DEFAULT_TRANCO_PATH,
        http_fetcher: Callable[[str, float], HttpFetchResult] | None = None,
        tls_checker: Callable[[str, int, float], TlsProbeResult] | None = None,
        dns_resolver: Callable[[str, float], bool] | None = None,
        whois_lookup: Callable[[str, float], dict[str, Any]] | None = None,
    ) -> None:
        if budget_s <= 0:
            raise ValueError(f"budget_s must be positive, got {budget_s}.")
        self.budget_s = budget_s
        self._blocklist_path = Path(blocklist_path) if blocklist_path else None
        self._tranco_path = Path(tranco_path) if tranco_path else None
        self._http_fetcher = http_fetcher or _default_http_fetcher
        self._tls_checker = tls_checker or _default_tls_checker
        self._dns_resolver = dns_resolver or _default_dns_resolver
        self._whois_lookup = whois_lookup or _default_whois_lookup

    def extract(self, url: str) -> HeavyExtractionResult:
        """Extract all 21 Tier 2 heavy features within `self.budget_s`, imputing any timed-out/failed probes."""
        t0 = time.perf_counter()
        normalized = _normalize_url(url)
        parts = urlsplit(normalized)
        host = parts.hostname
        if not host:
            raise ValueError(f"Could not parse a hostname from URL: {url!r}")
        scheme = parts.scheme.lower()
        tls_port = parts.port or 443

        # Start with training-set modes for every heavy feature
        features: dict[str, int] = {name: HEAVY_FEATURE_MODES[name] for name in TIER2_HEAVY_FEATURES}
        status: dict[str, str] = {name: "imputed" for name in TIER2_HEAVY_FEATURES}

        # 1. Offline reputation features are instantaneous (no network I/O)
        blocklist = (
            _load_blocklist_hosts(str(self._blocklist_path.resolve()))
            if self._blocklist_path and self._blocklist_path.exists()
            else frozenset()
        )
        tranco = (
            _load_tranco_ranks(str(self._tranco_path.resolve()))
            if self._tranco_path and self._tranco_path.exists()
            else {}
        )
        rep = score_reputation(host, blocklist_hosts=blocklist, tranco_ranks=tranco)
        for name, val in rep.items():
            features[name] = val
            status[name] = "ok"

        # 2. If scheme is plain HTTP, SSLfinal_State is deterministically -1 without a TLS probe
        run_tls_probe = scheme == "https"
        if not run_tls_probe:
            features["SSLfinal_State"] = PHISHING
            status["SSLfinal_State"] = "ok"

        probe_timeout = max(0.05, self.budget_s * 0.9)
        executor = ThreadPoolExecutor(max_workers=4)
        futures: dict[Any, tuple[str, tuple[str, ...]]] = {
            executor.submit(self._http_fetcher, normalized, probe_timeout): ("http", HTML_FEATURES),
            executor.submit(self._dns_resolver, host, probe_timeout): ("dns", ("DNSRecord",)),
            executor.submit(self._whois_lookup, host, probe_timeout): ("whois", WHOIS_FEATURES),
        }
        if run_tls_probe:
            futures[executor.submit(self._tls_checker, host, tls_port, probe_timeout)] = (
                "tls",
                ("SSLfinal_State",),
            )

        try:
            remaining = max(0.0, self.budget_s - (time.perf_counter() - t0))
            done, not_done = wait(futures.keys(), timeout=remaining, return_when=FIRST_COMPLETED)
            while not_done:
                remaining = self.budget_s - (time.perf_counter() - t0)
                if remaining <= 0:
                    break
                newly_done, not_done = wait(not_done, timeout=remaining, return_when=FIRST_COMPLETED)
                done |= newly_done

            for fut in done:
                kind, group_names = futures[fut]
                try:
                    outcome = fut.result(timeout=0)
                except Exception:
                    for name in group_names:
                        status[name] = "error"
                    continue

                if kind == "http":
                    html_vals = extract_html_features(
                        outcome.html,
                        outcome.final_url or normalized,
                        redirect_count=outcome.redirect_count,
                    )
                    for name, val in html_vals.items():
                        features[name] = val
                        status[name] = "ok"
                elif kind == "tls":
                    features["SSLfinal_State"] = score_tls_info(outcome)
                    status["SSLfinal_State"] = "ok"
                elif kind == "dns":
                    features["DNSRecord"] = LEGITIMATE if outcome else PHISHING
                    status["DNSRecord"] = "ok"
                elif kind == "whois":
                    whois_vals = score_whois_record(outcome, host)
                    for name, val in whois_vals.items():
                        features[name] = val
                        status[name] = "ok"

            for fut in not_done:
                fut.cancel()
                _, group_names = futures[fut]
                for name in group_names:
                    status[name] = "timeout"
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        ordered = {name: int(features[name]) for name in TIER2_HEAVY_FEATURES}
        validate_feature_vector(ordered, TIER2_HEAVY_FEATURES, strict_domain=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return HeavyExtractionResult(features=ordered, status=status, extraction_ms=elapsed_ms)


__all__ = [
    "ALLOWED_FEATURE_VALUES",
    "HEAVY_FEATURE_MODES",
    "HeavyExtractionResult",
    "HeavyFeatureExtractor",
    "HttpFetchResult",
    "TlsProbeResult",
    "extract_html_features",
    "score_reputation",
    "score_tls_info",
    "score_whois_record",
]
