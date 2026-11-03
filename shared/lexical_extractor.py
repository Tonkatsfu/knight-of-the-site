"""Zero-network Tier 1 lexical URL feature extractor (9 Address-Bar features).

Implements the Address Bar based rules from `Phishing Websites Features.docx`
(Mohammad, Thabtah & McCluskey) using only the URL string — no DNS, HTTP, or
WHOIS lookups — so it can run on a resource-constrained edge device in well
under 1 ms per URL.

Value convention matches the UCI ARFF: 1 = Legitimate, 0 = Suspicious, -1 = Phishing.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

import numpy as np

from shared.feature_schema import TIER1_FEATURES

LEGITIMATE: int = 1
SUSPICIOUS: int = 0
PHISHING: int = -1

# URL_Length thresholds from the paper: < 54 legitimate, 54..75 suspicious, > 75 phishing
URL_LENGTH_LEGIT_MAX_EXCLUSIVE: int = 54
URL_LENGTH_SUSPICIOUS_MAX_INCLUSIVE: int = 75

# Position of the last "//" beyond which the URL is considered a redirect ("https://" -> index 6)
DOUBLE_SLASH_MAX_POSITION: int = 7

# Ports with "Open" preferred status in the paper's port table; any other explicit port is phishing
STANDARD_PORTS: frozenset[int] = frozenset({80, 443})

# Well-known URL shortening services (matched against the full host or as a parent domain)
SHORTENING_SERVICES: frozenset[str] = frozenset({
    "bit.ly", "bitly.com", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "v.gd",
    "buff.ly", "adf.ly", "bit.do", "j.mp", "tr.im", "cli.gs", "tiny.cc", "lnkd.in",
    "db.tt", "qr.ae", "qr.net", "cur.lv", "ity.im", "q.gs", "po.st", "bc.vc", "u.to",
    "x.co", "su.pr", "wp.me", "rb.gy", "cutt.ly", "cutt.us", "shorturl.at", "t.ly",
    "tiny.one", "rebrand.ly", "shorte.st", "go2l.ink", "yfrog.com", "migre.me", "ff.im",
    "url4.eu", "twit.ac", "twurl.nl", "snipurl.com", "snipr.com", "short.to", "short.ie",
    "budurl.com", "ping.fm", "post.ly", "just.as", "bkite.com", "fic.kr", "loopt.us",
    "doiop.com", "kl.am", "rubyurl.com", "om.ly", "to.ly", "twitthis.com", "buzurl.com",
    "u.bb", "yourls.org", "scrnch.me", "filoops.info", "vzturl.com", "1url.com",
    "tweez.me", "link.zip.net", "s.id", "shorturl.com", "soo.gd", "clck.ru",
})

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
# Dotted IPv4 in decimal/octal/hex parts (1-4 parts), e.g. "0x58.0xCC.0xCA.0x62" or dword "3232235777"
_NUMERIC_IP_RE = re.compile(r"^(?:(?:0x[0-9a-f]+|\d+)\.){0,3}(?:0x[0-9a-f]+|\d+)$")
# Every ccTLD is exactly two ASCII letters
_CCTLD_RE = re.compile(r"\.[a-z]{2}$")


def _normalize_url(url: str) -> str:
    """Strip whitespace and prepend `http://` when no scheme is present, like a browser address bar."""
    if not isinstance(url, str):
        raise TypeError(f"URL must be a string, got {type(url).__name__}.")
    stripped = url.strip()
    if not stripped:
        raise ValueError("URL must be a non-empty string.")
    if not _SCHEME_RE.match(stripped):
        stripped = "http://" + stripped
    return stripped


def _is_ip_host(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return _NUMERIC_IP_RE.match(host) is not None


def _is_shortener(host: str) -> bool:
    return any(host == s or host.endswith("." + s) for s in SHORTENING_SERVICES)


def _sub_domain_value(host: str) -> int:
    """Omit leading `www.` and a trailing ccTLD, then count the remaining dots."""
    if host.startswith("www."):
        host = host[4:]
    host = _CCTLD_RE.sub("", host)
    dots = host.count(".")
    if dots <= 1:
        return LEGITIMATE
    if dots == 2:
        return SUSPICIOUS
    return PHISHING


def _port_value(parts) -> int:
    try:
        port = parts.port
    except ValueError:
        return PHISHING  # malformed / out-of-range explicit port
    if port is None or port in STANDARD_PORTS:
        return LEGITIMATE
    return PHISHING


def extract_lexical_features(url: str) -> dict[str, int]:
    """Extract the 9 Tier 1 lexical features from a URL string.

    Returns a dict keyed by `TIER1_FEATURES` (in that order) with values in {-1, 0, 1}.
    Raises `TypeError` for non-string input and `ValueError` when no hostname can be parsed.
    """
    normalized = _normalize_url(url)
    parts = urlsplit(normalized)
    host = parts.hostname  # lower-cased, userinfo and port removed, IPv6 brackets stripped
    if not host:
        raise ValueError(f"Could not parse a hostname from URL: {url!r}")

    length = len(normalized)
    if length < URL_LENGTH_LEGIT_MAX_EXCLUSIVE:
        url_length = LEGITIMATE
    elif length <= URL_LENGTH_SUSPICIOUS_MAX_INCLUSIVE:
        url_length = SUSPICIOUS
    else:
        url_length = PHISHING

    features = {
        "having_IP_Address": PHISHING if _is_ip_host(host) else LEGITIMATE,
        "URL_Length": url_length,
        "Shortining_Service": PHISHING if _is_shortener(host) else LEGITIMATE,
        "having_At_Symbol": PHISHING if "@" in normalized else LEGITIMATE,
        "double_slash_redirecting": (
            PHISHING if normalized.rfind("//") > DOUBLE_SLASH_MAX_POSITION else LEGITIMATE
        ),
        "Prefix_Suffix": PHISHING if "-" in host else LEGITIMATE,
        "having_Sub_Domain": _sub_domain_value(host),
        "port": _port_value(parts),
        "HTTPS_token": PHISHING if "https" in host else LEGITIMATE,
    }
    return {name: features[name] for name in TIER1_FEATURES}


def extract_tier1_vector(url: str) -> np.ndarray:
    """Return the 9 Tier 1 features as a float32 vector of shape (9,), ordered by `TIER1_FEATURES`."""
    features = extract_lexical_features(url)
    return np.asarray([features[name] for name in TIER1_FEATURES], dtype=np.float32)
