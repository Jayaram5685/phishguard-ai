"""Lexical (offline) feature extraction tests.

These lock the semantics of the URL-syntax features that do not require
network access. The live WHOIS/DNS/HTML path is exercised end to end by the
``network``-marked tests; the WHOIS date arithmetic is covered offline below,
because mixing timezone-aware and naive datetimes from python-whois used to
crash ``/predict`` with a 500.
"""

from datetime import UTC, datetime

from URLFeatureExtraction import (
    domainAge,
    domainEnd,
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


class _WhoisStub:
    """Minimal stand-in for a python-whois result object."""

    def __init__(self, creation_date, expiration_date):
        self.creation_date = creation_date
        self.expiration_date = expiration_date


def test_domain_dates_accept_timezone_aware_values():
    """python-whois returns timezone-aware datetimes for some registries
    (Verisign/.com does). Subtracting those from naive ones raised TypeError
    and took the whole /predict request down with a 500 -- live regression
    caught by scanning http://example.com/login."""
    stub = _WhoisStub(
        datetime(2015, 6, 1, tzinfo=UTC),
        datetime(2030, 6, 1, tzinfo=UTC),
    )
    assert domainAge(stub) == 0  # 15 years old: not a young domain
    assert domainEnd(stub) == 1  # expires well beyond six months


def test_domain_dates_mixed_aware_and_naive_values():
    stub = _WhoisStub(datetime(2015, 6, 1), datetime(2030, 6, 1, tzinfo=UTC))
    assert domainAge(stub) == 0


def test_domain_dates_unusable_values_degrade_to_unavailable():
    """Anything python-whois cannot hand back as a timestamp must behave like
    a failed lookup (value 1), never raise."""
    assert domainAge(_WhoisStub("not-a-date", datetime(2030, 1, 1))) == 1
    assert domainAge(_WhoisStub(None, datetime(2030, 1, 1))) == 1
    assert domainEnd(_WhoisStub(datetime(2015, 1, 1), "not-a-date")) == 1
    assert domainEnd(_WhoisStub(datetime(2015, 1, 1), None)) == 1
