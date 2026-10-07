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


def test_having_ip_must_inspect_hostname_not_whole_url():
    """Documents a known defect: the current implementation tests the entire
    URL string, so a genuine IP-literal host is never detected. Kept as an
    xfail so the P2 rewrite flips it to a real assertion."""
    import pytest

    pytest.xfail(
        "Have_IP is computed on the whole URL (URLFeatureExtraction.havingIP); "
        "fixed in milestone P2 where the feature pipeline is unified."
    )
    assert havingIP("http://192.168.1.10/login") == 1
