"""SSRF guard tests — every case here must fail closed, offline.

DNS behaviour is exercised with a monkeypatched ``socket.getaddrinfo`` so the
suite stays hermetic (marker ``network`` is reserved for genuinely live tests).
"""

import socket

import pytest

from phishguard.security.ssrf import (
    BlockedURLError,
    is_public_domain_candidate,
    scheme_of,
    validate_url,
)


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        # --- dangerous schemes: rejected before any I/O ---
        ("javascript:alert(1)", "scheme_not_allowed"),
        ("file:///etc/passwd", "scheme_not_allowed"),
        ("data:text/html,<script>x</script>", "scheme_not_allowed"),
        ("ftp://example.com/pub", "scheme_not_allowed"),
        ("gopher://example.com/", "scheme_not_allowed"),
        # --- loopback / unspecified ---
        ("http://localhost/admin", "blocked_hostname"),
        ("http://LOCALHOST./", "blocked_hostname"),
        ("http://127.0.0.1/", "non_global_ip"),
        ("http://0.0.0.0/", "non_global_ip"),
        # --- private IPv4 (RFC1918) ---
        ("http://10.0.0.5/", "non_global_ip"),
        ("http://172.16.0.1/", "non_global_ip"),
        ("http://192.168.1.1/", "non_global_ip"),
        # --- cloud metadata / link-local ---
        ("http://169.254.169.254/latest/meta-data/", "non_global_ip"),
        ("http://metadata.google.internal/computeMetadata/v1/", "blocked_hostname"),
        # --- IPv6 ---
        ("http://[::1]/", "non_global_ip"),
        ("http://[fc00::1]/", "non_global_ip"),
        ("http://[fe80::1]/", "non_global_ip"),
        ("http://[::ffff:127.0.0.1]/", "non_global_ip"),
        # --- alternative IP spellings browsers also interpret ---
        ("http://2130706433/", "non_global_ip"),  # decimal == 127.0.0.1
        ("http://0x7f000001/", "non_global_ip"),  # hex == 127.0.0.1
        # --- internal name suffixes ---
        ("http://build-server.internal/", "blocked_hostname"),
        ("http://printer.local/", "blocked_hostname"),
        ("http://db.lan/", "blocked_hostname"),
        # --- unusual ports block internal service discovery ---
        ("http://example.com:22/", "blocked_port"),
        ("http://example.com:6379/", "blocked_port"),
        ("http://example.com:8443/", "blocked_port"),
        # --- structural attacks ---
        ("http:///", "host_missing"),
        ("http://", "host_missing"),
        ("http://[::1", "malformed_url"),
        ("", "empty_url"),
        ("   ", "empty_url"),
        ("http://exa\tmple.com/", "control_characters"),
        ("https://example.com/" + "a" * 3000, "url_too_long"),
        (12345, "invalid_type"),
    ],
)
def test_blocked(url, reason):
    with pytest.raises(BlockedURLError) as excinfo:
        validate_url(url, resolve_dns=False)
    assert excinfo.value.reason == reason


@pytest.mark.parametrize(
    "url",
    [
        "https://93.184.216.34/index.html",
        "https://example.com/path?x=1",
        "http://example.com/",
        "http://example.com:443/",
        "http://Example.COM./trailing-dot",
    ],
)
def test_allowed_without_dns(url):
    validated = validate_url(url, resolve_dns=False)
    assert validated.scheme in ("http", "https")
    assert validated.port in (80, 443)


def test_default_port_follows_scheme():
    assert validate_url("https://example.com/", resolve_dns=False).port == 443
    assert validate_url("http://example.com/", resolve_dns=False).port == 80


def test_trailing_dot_and_case_are_normalised():
    validated = validate_url("http://EXAMPLE.com./", resolve_dns=False)
    assert validated.hostname == "example.com"


def test_hostname_resolving_to_private_ip_is_blocked(monkeypatch):
    def fake_getaddrinfo(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(BlockedURLError) as excinfo:
        validate_url("http://rebind.example.com/")
    assert excinfo.value.reason == "non_global_ip"


def test_hostname_resolving_to_metadata_ip_is_blocked(monkeypatch):
    def fake_getaddrinfo(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 80))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(BlockedURLError) as excinfo:
        validate_url("http://sneaky.example.com/")
    assert excinfo.value.reason == "non_global_ip"


def test_hostname_resolving_publicly_is_allowed(monkeypatch):
    def fake_getaddrinfo(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    validated = validate_url("http://rebind.example.com/")
    assert validated.resolved_ips == ("93.184.216.34",)


def test_resolver_failure_fails_closed(monkeypatch):
    def fake_getaddrinfo(*args, **kwargs):
        raise socket.gaierror(-2, "Name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(BlockedURLError) as excinfo:
        validate_url("http://does-not-exist.example/")
    assert excinfo.value.reason == "dns_resolution_failed"


@pytest.mark.parametrize(
    ("hostname", "expected"),
    [
        ("example.com", True),
        ("sub.example.co.uk", True),
        ("127.0.0.1", False),
        ("::1", False),
        ("2130706433", False),
        ("localhost", False),
        ("metadata.google.internal", False),
        ("host.lan", False),
        ("", False),
        (None, False),
        ("no-tld", False),
    ],
)
def test_whois_eligibility(hostname, expected):
    assert is_public_domain_candidate(hostname) is expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://example.com/", "https"),
        ("HTTP://example.com/", "http"),
        ("chrome://extensions", "chrome"),
        ("file:///tmp/x", "file"),
        ("not a url", ""),
        ("example.com/path", ""),
        ("http://[::1", ""),
        (None, ""),
    ],
)
def test_scheme_of(url, expected):
    assert scheme_of(url) == expected
