"""API behaviour tests.

Restricted to request paths that perform **no outbound network access**
(whitelisted domains, malformed input, unsupported schemes). Full-pipeline
extraction is covered by the network-marked tests in milestone P2.
"""

import dataclasses

import pytest

import app as app_module
from phishguard.security.ratelimit import SlidingWindowRateLimiter

EXTENSION_ORIGIN = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"


@pytest.fixture()
def client(monkeypatch):
    app_module.app.config["TESTING"] = True
    app_module.rate_limiter = SlidingWindowRateLimiter(limit=1000, window_seconds=60)
    yield app_module.app.test_client()
    app_module.rate_limiter = SlidingWindowRateLimiter(limit=1000, window_seconds=60)


def test_root_reports_running(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.get_json()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["model"] == "random_forest.joblib"


def test_predict_whitelisted_domain_is_skipped(client):
    response = client.post("/predict", json={"url": "https://www.google.com/search?q=x"})
    assert response.status_code == 200
    body = response.get_json()
    assert body["prediction"] == 0
    assert body["analysis"] == "whitelisted"
    assert body["domain"] == "google.com"


def test_predict_trusted_subdomain_is_whitelisted(client):
    response = client.post("/predict", json={"url": "https://mail.google.com/mail"})
    assert response.status_code == 200
    assert response.get_json()["prediction"] == 0


def test_missing_url_is_a_structured_client_error(client):
    response = client.post("/predict", json={})
    assert response.status_code == 400
    error = response.get_json()["error"]
    assert error["code"] == "invalid_request"
    assert "url" in error["message"]


def test_non_json_body_is_rejected(client):
    response = client.post("/predict", data="not json", content_type="text/plain")
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_json"


def test_oversized_url_is_rejected_without_analysis(client):
    response = client.post("/predict", json={"url": "https://example.com/" + "a" * 3000})
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "url_too_long"


def test_url_without_scheme_is_rejected(client):
    response = client.post("/predict", json={"url": "example.com/phish"})
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"


@pytest.mark.parametrize(
    "url",
    [
        "chrome://extensions",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "data:text/html,hello",
        "ftp://example.com/",
    ],
)
def test_unsupported_schemes_are_skipped_and_never_fetched(client, url):
    response = client.post("/predict", json={"url": url})
    assert response.status_code == 200
    body = response.get_json()
    assert body["prediction"] == 0
    assert body["analysis"] == "skipped"
    assert body["reason"].startswith("unsupported_scheme:")


def test_client_supplied_features_are_ignored(client):
    """The legacy endpoint accepted raw feature vectors from the client,
    letting anyone force a 'safe' verdict. Only URLs are accepted now."""
    response = client.post(
        "/predict",
        json={"features": {"Have_IP": 0, "URL_Length": 1}, "label": 0},
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_request"


def test_security_headers_present(client):
    response = client.get("/")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Cache-Control"] == "no-store"


def test_request_id_is_returned_and_attached_to_errors(client):
    response = client.get("/")
    request_id = response.headers["X-Request-ID"]
    assert request_id

    error_response = client.post("/predict", json={})
    assert error_response.headers["X-Request-ID"]
    assert error_response.get_json()["error"]["request_id"]


def test_errors_never_leak_internals(client):
    response = client.post("/predict", json={})
    message = response.get_json()["error"]["message"]
    assert "Traceback" not in message
    assert "File \"" not in message
    assert "werkzeug" not in message.lower()


def test_rate_limit_returns_429_with_retry_after(monkeypatch):
    app_module.app.config["TESTING"] = True
    app_module.rate_limiter = SlidingWindowRateLimiter(limit=2, window_seconds=60)
    client = app_module.app.test_client()

    assert client.post("/predict", json={"url": "chrome://extensions"}).status_code == 200
    assert client.post("/predict", json={"url": "chrome://extensions"}).status_code == 200
    response = client.post("/predict", json={"url": "chrome://extensions"})
    assert response.status_code == 429
    assert response.get_json()["error"]["code"] == "rate_limited"
    assert int(response.headers["Retry-After"]) >= 1

    app_module.rate_limiter = SlidingWindowRateLimiter(limit=1000, window_seconds=60)


def test_cors_allows_extension_origin(client):
    response = client.options(
        "/predict",
        headers={
            "Origin": EXTENSION_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert response.headers.get("Access-Control-Allow-Origin") == EXTENSION_ORIGIN


def test_cors_blocks_arbitrary_website_origins(client):
    response = client.options(
        "/predict",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers.get("Access-Control-Allow-Origin") is None


def test_api_key_enforced_when_configured(monkeypatch):
    hardened = dataclasses.replace(app_module.settings, api_key="sekrit-key")
    monkeypatch.setattr(app_module, "settings", hardened)
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    denied = client.post("/predict", json={"url": "chrome://extensions"})
    assert denied.status_code == 401
    assert denied.get_json()["error"]["code"] == "unauthorized"

    allowed = client.post(
        "/predict",
        json={"url": "chrome://extensions"},
        headers={"X-API-Key": "sekrit-key"},
    )
    assert allowed.status_code == 200
