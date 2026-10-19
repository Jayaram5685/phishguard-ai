"""Optional end-to-end test against the deployed API.

Skipped unless PHISHGUARD_LIVE_URL is set, so neither local runs nor CI (which
deselects the network marker) depend on a live service:

    PHISHGUARD_LIVE_URL=https://phishguard-api-ttou.onrender.com \
        python -m pytest tests/test_live_api.py -v
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        not os.environ.get("PHISHGUARD_LIVE_URL"),
        reason="PHISHGUARD_LIVE_URL not set",
    ),
]

TIMEOUT = 90  # free instances may need a cold start


def _base() -> str:
    return os.environ["PHISHGUARD_LIVE_URL"].rstrip("/")


def _decode(stream) -> dict:
    return json.loads(stream.read().decode("utf-8"))


def _post(path: str, payload: bytes) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{_base()}{path}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, _decode(response)
    except urllib.error.HTTPError as error:
        return error.code, _decode(error)


def test_health():
    with urllib.request.urlopen(f"{_base()}/health", timeout=TIMEOUT) as response:
        assert response.status == 200
        body = _decode(response)
        assert body["status"] == "ok"
        assert body["model"].endswith(".joblib")


def test_predict_phishing_url():
    status, body = _post(
        "/predict", b'{"url":"http://secure-paypal-verify.account-check.top/signin"}'
    )
    assert status == 200
    assert body["prediction"] == 1
    assert body["reasons"]  # explain() must be wired into the live service


def test_predict_whitelist_and_skip():
    status, body = _post("/predict", b'{"url":"https://github.com"}')
    assert (status, body["analysis"]) == (200, "whitelisted")

    status, body = _post("/predict", b'{"url":"chrome://version"}')
    assert (status, body["analysis"]) == (200, "skipped")


def test_security_headers_live():
    with urllib.request.urlopen(f"{_base()}/health", timeout=TIMEOUT) as response:
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        csp = response.headers["Content-Security-Policy"]
        assert csp.startswith("default-src 'none'")
        assert response.headers["Strict-Transport-Security"].startswith("max-age=")


def test_invalid_request_shape():
    status, body = _post("/predict", b'{"nourl":1}')
    assert status == 400
    assert body["error"]["code"] == "invalid_request"
    assert body["error"]["request_id"]
