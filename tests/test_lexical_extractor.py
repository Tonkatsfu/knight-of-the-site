"""Unit tests for the zero-network Tier 1 lexical URL extractor (`shared/lexical_extractor.py`).

Each test encodes one rule from `Phishing Websites Features.docx` (Address Bar based Features).
Value convention matches the UCI ARFF: 1 = Legitimate, 0 = Suspicious, -1 = Phishing.
"""

import time

import numpy as np
import pytest

from shared.feature_schema import TIER1_FEATURES, validate_tier1_features
from shared.lexical_extractor import extract_lexical_features, extract_tier1_vector


# --- having_IP_Address -------------------------------------------------------

@pytest.mark.parametrize(
    "url",
    [
        "http://125.98.3.123/fake.html",
        "http://0x58.0xCC.0xCA.0x62/2/paypal.ca/index.html",
        "http://[2001:db8::1]/login",
        "http://3232235777/login",  # dword-encoded IPv4
    ],
)
def test_ip_address_host_is_phishing(url):
    assert extract_lexical_features(url)["having_IP_Address"] == -1


def test_domain_name_host_is_legitimate_for_ip_feature():
    assert extract_lexical_features("https://www.google.com/")["having_IP_Address"] == 1


# --- URL_Length --------------------------------------------------------------

def test_url_length_boundaries():
    base = "http://example.com/"  # 19 chars
    assert extract_lexical_features(base + "a" * (53 - len(base)))["URL_Length"] == 1
    assert extract_lexical_features(base + "a" * (54 - len(base)))["URL_Length"] == 0
    assert extract_lexical_features(base + "a" * (75 - len(base)))["URL_Length"] == 0
    assert extract_lexical_features(base + "a" * (76 - len(base)))["URL_Length"] == -1


# --- Shortining_Service ------------------------------------------------------

@pytest.mark.parametrize(
    "url", ["http://bit.ly/19DXSk4", "https://tinyurl.com/abc", "https://www.goo.gl/x"]
)
def test_known_shortener_is_phishing(url):
    assert extract_lexical_features(url)["Shortining_Service"] == -1


def test_domain_that_merely_contains_shortener_text_is_legitimate():
    # "rabbit.lyrics.com" contains the substring "bit.ly" but is not the bit.ly host
    assert extract_lexical_features("http://rabbit.lyrics.com/")["Shortining_Service"] == 1


# --- having_At_Symbol --------------------------------------------------------

def test_at_symbol_is_phishing():
    assert extract_lexical_features("http://paypal.com@evil.example/")["having_At_Symbol"] == -1
    assert extract_lexical_features("http://paypal.com/")["having_At_Symbol"] == 1


# --- double_slash_redirecting ------------------------------------------------

def test_double_slash_after_scheme_is_phishing():
    url = "http://www.legitimate.com//http://www.phishing.com"
    assert extract_lexical_features(url)["double_slash_redirecting"] == -1


def test_scheme_double_slash_only_is_legitimate():
    assert extract_lexical_features("https://www.example.com/a/b")["double_slash_redirecting"] == 1
    assert extract_lexical_features("http://www.example.com/a/b")["double_slash_redirecting"] == 1


# --- Prefix_Suffix -----------------------------------------------------------

def test_dash_in_domain_is_phishing():
    assert extract_lexical_features("http://www.Confirme-paypal.com/")["Prefix_Suffix"] == -1


def test_dash_only_in_path_is_legitimate():
    assert extract_lexical_features("http://www.paypal.com/my-account")["Prefix_Suffix"] == 1


# --- having_Sub_Domain -------------------------------------------------------

@pytest.mark.parametrize(
    "url, expected",
    [
        ("http://www.hud.ac.uk/students/", 1),   # www + ccTLD stripped -> "hud.ac" (1 dot)
        ("https://google.com", 1),
        ("https://www.google.com", 1),
        ("https://mail.google.com", 0),          # 2 dots -> suspicious
        ("https://a.b.google.com", -1),          # 3 dots -> phishing
        ("https://secure.login.paypal.co.uk", -1),
    ],
)
def test_sub_domain_dot_counting(url, expected):
    assert extract_lexical_features(url)["having_Sub_Domain"] == expected


# --- port --------------------------------------------------------------------

@pytest.mark.parametrize("url", ["http://example.com:8080/", "http://example.com:21/"])
def test_non_standard_port_is_phishing(url):
    assert extract_lexical_features(url)["port"] == -1


@pytest.mark.parametrize(
    "url", ["http://example.com/", "http://example.com:80/", "https://example.com:443/"]
)
def test_default_or_standard_port_is_legitimate(url):
    assert extract_lexical_features(url)["port"] == 1


# --- HTTPS_token -------------------------------------------------------------

def test_https_token_in_domain_is_phishing():
    url = "http://https-www-paypal-it-webapps-mpp-home.soft-hair.com/"
    assert extract_lexical_features(url)["HTTPS_token"] == -1


def test_https_scheme_alone_is_legitimate():
    assert extract_lexical_features("https://www.paypal.com/")["HTTPS_token"] == 1


# --- Input handling & output contract ---------------------------------------

def test_scheme_less_input_is_parsed_like_a_browser_would():
    features = extract_lexical_features("bit.ly/19DXSk4")
    assert features["Shortining_Service"] == -1
    assert features["double_slash_redirecting"] == 1


def test_whitespace_is_stripped_and_host_is_case_insensitive():
    assert extract_lexical_features("  HTTP://BIT.LY/x  ")["Shortining_Service"] == -1


@pytest.mark.parametrize("bad", ["", "   ", "http://", "http:///path"])
def test_invalid_url_raises_value_error(bad):
    with pytest.raises(ValueError):
        extract_lexical_features(bad)


def test_non_string_raises_type_error():
    with pytest.raises(TypeError):
        extract_lexical_features(None)  # type: ignore[arg-type]


def test_output_is_ordered_and_schema_valid():
    features = extract_lexical_features("https://www.google.com/search?q=knight")
    assert list(features) == TIER1_FEATURES
    validate_tier1_features(features)  # raises if any value is outside the ARFF domain


def test_vector_matches_dict_in_tier1_order():
    url = "http://192.168.0.1:8080/https-login@x//y"
    features = extract_lexical_features(url)
    vec = extract_tier1_vector(url)
    assert vec.dtype == np.float32
    assert vec.shape == (9,)
    assert vec.tolist() == [float(features[name]) for name in TIER1_FEATURES]


def test_extraction_is_sub_millisecond_on_average():
    urls = [
        "https://www.google.com/",
        "http://0x58.0xCC.0xCA.0x62/2/paypal.ca/index.html",
        "http://https-www-paypal-it-webapps-mpp-home.soft-hair.com/",
        "http://bit.ly/19DXSk4",
    ] * 250
    start = time.perf_counter()
    for url in urls:
        extract_lexical_features(url)
    mean_ms = (time.perf_counter() - start) * 1000 / len(urls)
    assert mean_ms < 1.0
