"""PhishGuard AI — Flask API.

Security posture (milestone P1):
* SSRF-guarded outbound fetching (see phishguard/security/ssrf.py)
* Strict request validation, structured JSON errors, no stack traces
* Sliding-window rate limiting keyed on the real client IP
* Origin allowlist (extension + localhost) instead of wide-open CORS
* Optional API key (``API_KEY`` env) for automated clients
* Request IDs, structured JSON access logs, security headers
"""

import ipaddress
import json
import logging
import os
import sys
import time
import uuid
from urllib.parse import urlsplit

import joblib
import pandas as pd
from flask import Flask, Response, g, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from phishguard.config import Settings
from phishguard.security.ratelimit import SlidingWindowRateLimiter
from phishguard.security.ssrf import ALLOWED_SCHEMES, scheme_of
from URLFeatureExtraction import featureExtraction

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")

settings = Settings.from_env()

logging.basicConfig(level=settings.log_level, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("phishguard.api")

with open(REGISTRY_PATH, encoding="utf-8") as registry_file:
    MODEL_REGISTRY = json.load(registry_file)

if settings.model_name not in MODEL_REGISTRY:
    raise RuntimeError(f"model {settings.model_name!r} is not present in models/registry.json")

FEATURE_COLUMNS = MODEL_REGISTRY[settings.model_name]["feature_columns"]
MODEL_PATH = os.path.join(MODELS_DIR, settings.model_name)
model = joblib.load(MODEL_PATH)

rate_limiter = SlidingWindowRateLimiter(
    limit=settings.rate_limit_per_minute,
    window_seconds=60.0,
)

# Trusted domain whitelist to prevent false positives on highly popular domains
TRUSTED_DOMAINS = {
    "google.com", "github.com", "youtube.com", "wikipedia.org",
    "microsoft.com", "apple.com", "amazon.com", "gmail.com",
    "facebook.com", "twitter.com", "linkedin.com", "netflix.com",
    "google.co.in", "yahoo.com", "bing.com", "duckduckgo.com",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = settings.max_request_bytes

CORS(
    app,
    origins=list(settings.cors_origins),
    methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
    max_age=600,
)


def _log_event(event: str, *, level: int = logging.INFO, **fields) -> None:
    payload = {"event": event}
    payload.update({k: v for k, v in fields.items() if v is not None})
    logger.log(level, json.dumps(payload, sort_keys=True, default=str))


def _error(code: str, message: str, status: int) -> Response:
    body = {"error": {"code": code, "message": message}}
    request_id = getattr(g, "request_id", None)
    if request_id:
        body["error"]["request_id"] = request_id
    response = jsonify(body)
    response.status_code = status
    return response


def _is_private_peer(address: str | None) -> bool:
    if not address:
        return False
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    return not ip.is_global


def _client_key() -> str:
    """Rate-limit key for the real client.

    ``X-Forwarded-For`` is trusted only when explicitly configured
    (``TRUST_PROXY=1``) or when the direct peer is itself a private/loopback
    address — i.e. we are behind a reverse proxy such as Render's router.
    Directly exposed deployments therefore cannot have the header spoofed to
    mint unlimited buckets.
    """
    peer = request.remote_addr or "unknown"
    if settings.trust_proxy or _is_private_peer(peer):
        forwarded = request.headers.get("X-Forwarded-For", "")
        first_hop = forwarded.split(",")[0].strip()
        if first_hop:
            return first_hop
    return peer


@app.before_request
def _authenticate_and_limit():
    g.request_id = uuid.uuid4().hex[:12]
    g.started_at = time.perf_counter()

    if request.method == "OPTIONS":
        return None  # CORS preflight — no body, no rate-limit consumption
    if request.endpoint != "predict":
        return None

    if settings.api_key is not None and request.headers.get("X-API-Key") != settings.api_key:
        _log_event("auth_failed", level=logging.WARNING, request_id=g.request_id, ip=_client_key())
        return _error("unauthorized", "Missing or invalid API key", 401)

    result = rate_limiter.check(_client_key())
    if not result.allowed:
        _log_event(
            "rate_limited",
            level=logging.WARNING,
            request_id=g.request_id,
            ip=_client_key(),
            retry_after=result.retry_after,
        )
        response = _error("rate_limited", "Too many requests, slow down", 429)
        response.headers["Retry-After"] = str(result.retry_after)
        return response
    return None


@app.after_request
def _harden_response(response):
    request_id = getattr(g, "request_id", None)
    if request_id:
        response.headers["X-Request-ID"] = request_id

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"

    if request.method != "OPTIONS":
        started = getattr(g, "started_at", None)
        duration_ms = round((time.perf_counter() - started) * 1000, 2) if started else None
        # URL bodies are intentionally never logged: they may contain tokens.
        _log_event(
            "request",
            request_id=request_id,
            method=request.method,
            path=request.path,
            status=response.status_code,
            duration_ms=duration_ms,
            ip=_client_key(),
        )
    return response


@app.errorhandler(HTTPException)
def _handle_http_error(exc: HTTPException):
    code = (exc.name or "http_error").lower().replace(" ", "_")
    return _error(code, exc.description or exc.name, exc.code or 500)


@app.errorhandler(Exception)
def _handle_unexpected_error(exc: Exception):
    _log_event(
        "unhandled_error",
        level=logging.ERROR,
        request_id=getattr(g, "request_id", None),
        error_type=type(exc).__name__,
    )
    return _error("internal_error", "Unexpected server error", 500)


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {"status": "API is running. Send a POST request to /predict to use the model."}
    )


@app.route("/health", methods=["GET"])
def health():
    """Liveness probe used by render.yaml / load balancers."""
    return jsonify({"status": "ok", "model": settings.model_name})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return _error("invalid_json", "Request body must be a JSON object", 400)

    url = data.get("url")
    if not isinstance(url, str) or not url.strip():
        return _error(
            "invalid_request",
            "Field 'url' is required and must be a non-empty string",
            400,
        )
    url = url.strip()

    if len(url) > settings.max_url_length:
        return _error(
            "url_too_long",
            f"URL exceeds {settings.max_url_length} characters",
            400,
        )

    scheme = scheme_of(url)
    if not scheme:
        return _error("invalid_request", "Field 'url' must be an absolute URL", 400)
    if scheme not in ALLOWED_SCHEMES:
        # Browser-internal / local content (chrome://, file://, data:...):
        # nothing to analyse, and never fetched. Reported explicitly so
        # clients can distinguish "not applicable" from "checked and safe".
        return jsonify(
            {
                "prediction": 0,
                "analysis": "skipped",
                "reason": f"unsupported_scheme:{scheme}",
            }
        )

    hostname = (urlsplit(url).hostname or "").lower()
    domain = hostname[4:] if hostname.startswith("www.") else hostname
    if domain in TRUSTED_DOMAINS or any(
        domain.endswith("." + trusted) for trusted in TRUSTED_DOMAINS
    ):
        return jsonify({"prediction": 0, "analysis": "whitelisted", "domain": domain})

    try:
        features = featureExtraction(url)
        frame = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        prediction = int(model.predict(frame)[0])
    except Exception as exc:
        _log_event(
            "prediction_failed",
            level=logging.ERROR,
            request_id=g.request_id,
            error_type=type(exc).__name__,
        )
        return _error("internal_error", "Analysis failed", 500)

    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(port=5000)
