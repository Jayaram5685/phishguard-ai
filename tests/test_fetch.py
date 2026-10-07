"""Guarded-fetch tests: redirect validation, caps and fail-closed behaviour."""

import socket

import pytest

from phishguard.security.fetch import (
    FetchedPage,
    RawResponse,
    _requests_transport,
    safe_fetch,
)
from URLFeatureExtraction import forwarding


@pytest.fixture(autouse=True)
def _offline_dns(monkeypatch):
    """Resolve every hostname to one fixed public address so the suite never
    depends on live DNS."""
    def fake_getaddrinfo(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


def _response(status=200, headers=None, body=b"<html>ok</html>"):
    return RawResponse(status_code=status, headers=headers or {}, body=body)


def _transport(mapping):
    """Fake transport recording every URL it was asked to fetch."""
    calls: list[str] = []

    def transport(url, *, timeout, max_bytes):
        calls.append(url)
        return mapping.get(url)

    transport.calls = calls
    return transport


def test_simple_fetch_decodes_body():
    transport = _transport({"https://example.com/": _response()})
    page = safe_fetch("https://example.com/", transport=transport)
    assert isinstance(page, FetchedPage)
    assert page.text == "<html>ok</html>"
    assert page.history == []
    assert transport.calls == ["https://example.com/"]


def test_charset_from_content_type_is_honoured():
    transport = _transport(
        {
            "https://example.com/": _response(
                headers={"Content-Type": "text/html; charset=latin-1"},
                body="café".encode("latin-1"),
            )
        }
    )
    page = safe_fetch("https://example.com/", transport=transport)
    assert page.text == "café"


def test_unknown_charset_falls_back_to_utf8():
    transport = _transport(
        {
            "https://example.com/": _response(
                headers={"Content-Type": "text/html; charset=bogus-charset"},
                body=b"hi",
            )
        }
    )
    page = safe_fetch("https://example.com/", transport=transport)
    assert page.text == "hi"


def test_redirect_chain_is_followed_and_history_kept():
    mapping = {
        "https://start.example/": _response(
            302, {"Location": "https://start.example/mid"}
        ),
        "https://start.example/mid": _response(
            301, {"location": "/final"}
        ),
        "https://start.example/final": _response(200, body=b"done"),
    }
    transport = _transport(mapping)
    page = safe_fetch("https://start.example/", transport=transport)
    assert page.url == "https://start.example/final"
    assert page.text == "done"
    assert page.history == ["https://start.example/", "https://start.example/mid"]
    # 2 hops: compatible with the legacy Web_Forwards heuristic (<=2 is benign)
    assert forwarding(page) == 0


def test_redirect_to_internal_host_is_blocked_before_any_request():
    mapping = {
        "https://example.com/": _response(
            302, {"Location": "http://169.254.169.254/latest/meta-data/"}
        )
    }
    transport = _transport(mapping)
    assert safe_fetch("https://example.com/", transport=transport) is None
    assert transport.calls == ["https://example.com/"], (
        "the metadata address must never reach the transport"
    )


def test_redirect_to_loopback_is_blocked():
    mapping = {"https://example.com/": _response(302, {"Location": "http://127.0.0.1/"})}
    transport = _transport(mapping)
    assert safe_fetch("https://example.com/", transport=transport) is None
    assert transport.calls == ["https://example.com/"]


def test_redirect_to_file_scheme_is_blocked():
    mapping = {"https://example.com/": _response(302, {"Location": "file:///etc/passwd"})}
    transport = _transport(mapping)
    assert safe_fetch("https://example.com/", transport=transport) is None
    assert transport.calls == ["https://example.com/"]


def test_excessive_redirects_fail_closed():
    mapping = {}
    for i in range(10):
        mapping[f"https://example.com/{i}"] = _response(
            302, {"Location": f"https://example.com/{i + 1}"}
        )
    transport = _transport(mapping)
    assert safe_fetch("https://example.com/0", transport=transport, max_redirects=3) is None


def test_transport_failure_returns_none():
    assert safe_fetch("https://example.com/", transport=lambda url, **kwargs: None) is None


def test_redirect_without_location_returns_final_page():
    transport = _transport(
        {"https://example.com/": _response(301, body=b"moved body")}
    )
    page = safe_fetch("https://example.com/", transport=transport)
    assert page.status_code == 301
    assert page.text == "moved body"


def test_transport_receives_time_and_byte_budget():
    seen = {}

    def transport(url, *, timeout, max_bytes):
        seen["timeout"] = timeout
        seen["max_bytes"] = max_bytes
        return _response()

    safe_fetch("https://example.com/", transport=transport)
    assert seen["timeout"] == (2.05, 5.0)
    assert seen["max_bytes"] == 1_048_576


class _FakeRequestsResponse:
    """Stand-in for requests.Response used to test the real transport."""

    def __init__(self, chunks, headers=None, status=200):
        self.status_code = status
        self.headers = headers or {}
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_content(self, chunk_size):
        yield from self._chunks


def test_default_transport_truncates_oversized_bodies(monkeypatch):
    """A hostile server cannot make us buffer an unbounded body."""
    import phishguard.security.fetch as fetch_module

    oversized = [b"A" * 65_536 for _ in range(100)]  # ~6.5 MB total

    def fake_get(url, **kwargs):
        return _FakeRequestsResponse(oversized, headers={"Content-Type": "text/html"})

    monkeypatch.setattr(fetch_module.requests, "get", fake_get)
    raw = _requests_transport("https://example.com/", timeout=(1, 1), max_bytes=100_000)
    assert raw is not None
    assert len(raw.body) <= 100_000


def test_default_transport_swallows_network_errors(monkeypatch):
    import requests

    import phishguard.security.fetch as fetch_module

    def fake_get(url, **kwargs):
        raise requests.RequestException("boom")

    monkeypatch.setattr(fetch_module.requests, "get", fake_get)
    assert _requests_transport("https://example.com/", timeout=(1, 1), max_bytes=1000) is None
