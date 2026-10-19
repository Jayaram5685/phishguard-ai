"""Lexical (offline) feature extraction tests.

These lock the semantics of the URL-syntax features that do not require
network access. The WHOIS/DNS/HTML features are exercised separately in the
P2 pipeline rewrite.
"""

from URLFeatureExtraction import (
    getDepth,
    getLength,
    haveAtSign,
    havingIP,
    httpDomain,
    prefixSuffix,
    redirection,
    tinyURL,
)

SAFE_URL = "https://example.com/products/item?id=42"
PHISHY_URL = (
    "http://secure-paypal-login.verify-account.example.com@"
    "redirect/path/that/is/very/long/and/deep/for/no/real/reason/12345?x=1"
)


def test_have_at_sign():
    assert haveAtSign(SAFE_URL) == 0
    assert haveAtSign(PHISHY_URL) == 1


def test_url_length_threshold_is_54_characters():
    assert getLength("https://a.co") == 0
    assert getLength("https://example.com/" + "a" * 40) == 1


def test_url_depth_counts_path_segments():
    assert getDepth("https://example.com") == 0
    assert getDepth("https://example.com/a/b/c") == 3


def test_https_token_in_hostname_is_flagged():
    assert httpDomain("https://https-secure.example.com/") == 1
    assert httpDomain(SAFE_URL) == 0


def test_known_shortener_detected():
    assert tinyURL("https://bit.ly/3abcXYZ") == 1
    assert tinyURL(SAFE_URL) == 0


def test_hyphen_in_hostname_is_flagged():
    assert prefixSuffix("https://my-site.example/") == 1
    assert prefixSuffix(SAFE_URL) == 0


def test_extra_double_slash_after_scheme_is_redirection():
    assert redirection(SAFE_URL) == 0
    assert redirection("https://example.com//evil.example/path") == 1


def test_having_ip_inspects_hostname_not_whole_url():
    """features-v2 semantics (P2): Have_IP is a hostname-scoped dotted-quad
    test. Verified against the dataset: the hostname rule matches all 4,964
    reconstructed rows, the legacy whole-URL variant is falsified by 37 rows
    with IPs embedded in the path (labelled legitimate)."""
    assert havingIP("http://192.168.1.10/login") == 1
    assert havingIP("http://example.com/192.168.1.10/") == 0
    assert havingIP("https://example.com/search?q=10.0.0.1") == 0
    assert havingIP(SAFE_URL) == 0
