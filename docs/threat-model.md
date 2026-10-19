# Threat Model — PhishGuard AI

> Status: living document. Every mitigation named here has a test; every
> residual risk is stated as accepted, not implied away.

## 1. System under review

```
[ Chrome extension ]  --HTTPS, untrusted-->  [ Flask API /predict ]
        |                                          |
        | chrome.storage.local                     +--> feature extraction
        | (api key, base URL)                      |     +--> outbound fetch (SSRF-guarded)
        v                                          |     +--> DNS / WHOIS lookups
   user's browser                                 +--> joblib model (repo artifact)
                                                  +--> structured logs (stdout)
```

Components and their trust level:

| Component | Trust | Notes |
|---|---|---|
| Chrome extension (`Phishing_detection_app/chrome_extension/`) | **Untrusted client** | ships to users; may be modified, replayed, or bypassed entirely (the API is reachable with `curl`) |
| Flask API (`app.py`) | Trusted | runs on operator infrastructure |
| Feature extraction (`URLFeatureExtraction.py`, `phishguard/`) | Trusted code, **untrusted input** | parses attacker-controlled URLs and pages |
| Model artifacts (`models/*.joblib`) | Trusted, integrity-checked | loading a pickle executes arbitrary code |
| Dataset (`DataFiles/`) | Trusted for development | provenance/licence unverified |
| Target website being scanned | **Hostile** | attacker-influenced content, redirects, DNS |

## 2. Trust boundaries

| # | Boundary | Crossing |
|---|---|---|
| B1 | Internet → `/predict` | JSON body with an attacker-chosen URL; no auth by default |
| B2 | API → scanned target | server-initiated HTTP(S) fetch of attacker-chosen host (the SSRF boundary) |
| B3 | API → model artifacts | `joblib.load` of files from `models/` |
| B4 | Extension → API | cross-origin `fetch`, `chrome-extension://` origin, optional `X-API-Key` |
| B5 | Operator → configuration | environment variables only; never committed |

## 3. Assets worth protecting

1. **The server's network position** — loopback services, RFC1918 peers,
   cloud metadata (169.254.169.254) must be unreachable through the API.
2. **API availability** — one client must not be able to exhaust workers,
   sockets or disk.
3. **Secrets** — `API_KEY`, and any tokens embedded in scanned URLs.
4. **Model and code integrity** — a swapped `.joblib` is remote code execution
   at boot.
5. **User safety** — a wrong "Safe" verdict is the product's failure mode.

## 4. Threats and mitigations

| # | Threat | Mitigation (where implemented) | Status |
|---|---|---|---|
| T1 | **SSRF open proxy**: `/predict` fetches `http://169.254.169.254/`, `http://127.0.0.1:6379/`, RFC1918 hosts | `validate_url()` in [`phishguard/security/ssrf.py`](../phishguard/security/ssrf.py): `http/https` only, ports 80/443 only, internal-name blocklist, all resolved addresses must be `is_global`, **fail-closed** on DNS errors | ✅ `tests/test_ssrf.py` |
| T2 | Scheme smuggling: `file:`, `javascript:`, `data:`, `gopher:` | scheme allowlist; unsupported schemes return an explicit `{"prediction":0,"analysis":"skipped"}` and are **never fetched** | ✅ `tests/test_ssrf.py`, `tests/test_api.py` |
| T3 | IP literal evasion: decimal (`2130706433`), hex (`0x7f000001`), IPv4-mapped IPv6, trailing dot, tab/newline in host | parsed as IPv4/IPv6 the way browsers do; non-global rejected; control chars rejected; hostname normalised | ✅ `tests/test_ssrf.py` |
| T4 | **DNS rebinding (TOCTOU)**: validated address ≠ address later connected to | every redirect hop re-validated, short HTTP timeouts | ⚠️ **accepted residual** — connection pinning is out of scope for this layer (§6) |
| T5 | Slow-resolver denial of service | HTTP-level timeouts; rate limiting bounds concurrent work; `getaddrinfo` itself has no portable timeout | ⚠️ **accepted residual** (availability only) |
| T6 | Request floods / oversized bodies | sliding-window rate limit per client (default 60/min), `MAX_URL_LENGTH` (2048), `MAX_CONTENT_LENGTH` (32 KB), 15 s client-side timeout | ✅ `tests/test_ratelimit.py`, `tests/test_config.py` |
| T7 | **Rate-limit key spoofing** via `X-Forwarded-For` | the header is trusted only when `TRUST_PROXY=1` **or** the direct peer is private/loopback (behind a proxy); otherwise the socket peer is used — a directly-exposed deployment cannot be minted unlimited buckets | ✅ `tests/test_ratelimit.py`, `tests/test_api.py` |
| T8 | Unauthorised model use / API abuse | optional `API_KEY` → `X-API-Key` (401 + `auth_failed` log event). Off by default so the extension works zero-config | ⚠️ by design — enable in production (§6) |
| T9 | Malicious web page calling the API cross-origin | CORS allowlist: `chrome-extension://[a-p]{32}`, `localhost`, `127.0.0.1` — not `*`; preflights allowed, `Content-Type` + `X-API-Key` headers only | ✅ `tests/test_api.py` — **CORS is not authentication** (§6) |
| T10 | Clickjacking / response sniffing | no HTML surface; `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, `Cache-Control: no-store` on every response | ✅ `tests/test_api.py` |
| T11 | **Information disclosure**: stack traces, URLs containing tokens in logs | errors are structured `{"error":{code,message,request_id}}` with no internals; **URLs are never logged** (path/method/status/duration/ip only); `BlockedURLError.reason` is a stable code that never embeds the URL | ✅ `tests/test_api.py` |
| T12 | Client-supplied feature vectors: a caller POSTs `{"url":…, "features":…}` to force a verdict | only `url` is read; extra fields are ignored and the server re-extracts features itself; extracted features are schema-validated (`SchemaError` → 422) before reaching a model | ✅ `tests/test_api.py` |
| T13 | Whitelist spoof / parser confusion in the trusted-domain check | exact-domain or `.domain` suffix match against a fixed set; hostname parsed by `urlsplit`, lowercased, `www.` stripped | ✅ `tests/test_api.py` |
| T14 | **Model artifact tampering** (pickle = RCE on load) | artifacts are content-addressed: `registry.json` carries SHA-256 for every artifact, enforced by `tests/test_model_registry.py`; `MODEL_NAME` may be a bare `*.joblib` filename only (path traversal rejected); artifacts ship through the reviewed repo, not uploads | ⚠️ no signature yet (§6) |
| T15 | Wrong default configuration | fail-fast at import: `MODEL_NAME` must exist in the registry and its `feature_columns` must equal features-v2 exactly; bad env values (`RATE_LIMIT_PER_MINUTE=0`, `MAX_URL_LENGTH=10`, paths in `MODEL_NAME`) raise `ValueError` at boot | ✅ `tests/test_config.py`, app import-time assertion |
| T16 | Dependency compromise / known CVEs | pinned ranges with security floors (`flask-cors>=6.0` for PYSEC-2026-1383/4/5, `requests>=2.32` for CVE-2024-35195); `pip-audit` runs in CI on every PR | ✅ `.github/workflows/ci.yml` |
| T17 | Extension over-privilege | manifest v1.1.0: `activeTab`, `tabs`, `storage` only; no `host_permissions`; the API is reached through CORS from the extension origin, not by granting the extension blanket host access | ✅ manifest review |
| T18 | Extension lies about the verdict | server errors (401/429/5xx, timeout, `data.error`) are shown as **"Could not complete"**, never "Safe"; whitelisted and skipped results are rendered as distinct states; each verdict shows the model's own `score` and `reasons` | ✅ `popup.js` hardening, `tests/test_explain.py` |
| T19 | Data poisoning / unlicensed training data | dataset SHA-256 recorded and checked; provenance gaps documented as unverified in [`DataFiles/DATASETS.md`](../DataFiles/DATASETS.md) and §9 of [`METHODOLOGY.md`](METHODOLOGY.md) | ⚠️ documented, not fixable retroactively |

## 5. What the extension is allowed to do

| Permission | Why it is needed |
|---|---|
| `activeTab` | read the current tab's URL when the user opens the popup |
| `tabs` | surface the URL in the popup |
| `storage` | remember the API base URL and optional API key locally |

No `host_permissions`, no content scripts, no background workers, no
`webRequest` — the extension never intercepts traffic; it asks the user's
explicitly-opened popup to scan one URL.

## 6. Accepted residual risks

1. **No authentication by default.** `API_KEY` is unset until an operator sets
   it. Rationale: the shipped extension must work on first install. Action:
   set `API_KEY` (and restrict `CORS_ORIGINS`) on any internet-facing deploy.
2. **CORS is not an access-control mechanism.** A page served from
   `http://localhost:<port>` is on the allowlist, so a local process or a
   malicious local page can call `/predict`. Rate limiting bounds the blast
   radius; `API_KEY` closes it.
3. **DNS rebinding TOCTOU** between validation and connect. Mitigated, not
   eliminated: every redirect hop re-validated, short timeouts. Full address
   pinning would require owning the HTTP client's resolver — out of scope.
4. **Resolver latency** has no portable timeout; a slow DNS server delays the
   verdict (availability, not confidentiality or integrity).
5. **Artifacts are not signed.** Integrity relies on repo review + SHA-256
   manifests. A signature scheme is the natural next step before third-party
   contributions of artifacts.
6. **Model quality bounds user safety.** On the primary protocol the shipped
   models miss 27–34% of phishing and mis-flag 7.5–15% of legitimate sites
   (§6.1 of [`METHODOLOGY.md`](METHODOLOGY.md)). The UI must never present a
   verdict as certainty — hence score + reasons + explicit "could not
   complete" states.
7. **Unverified dataset provenance.** Collection dates and licences were never
   recorded upstream; the data is fit for method development, not for
   published detection-rate claims.

## 7. Out of scope

* Compromise of the browser or operating system.
* Adversarial ML in feature space (crafted URLs that flip the verdict) — no
  robustness evaluation is claimed.
* Availability of third-party dependencies (WHOIS registries, DNS providers).
* At-rest encryption of the dataset or artifacts (host/CI responsibility).
* Multi-tenant authentication, per-user quotas, audit retention.

## 8. Verification map

| Guarantee | Test |
|---|---|
| SSRF checks and fail-closed behaviour | `tests/test_ssrf.py` |
| Rate limiting, `Retry-After`, key derivation | `tests/test_ratelimit.py` |
| API contract: errors, headers, whitelist, ignored extra fields, verdict payload | `tests/test_api.py` |
| Configuration bounds and `MODEL_NAME` rejection | `tests/test_config.py` |
| Feature schema integrity end to end | `tests/test_features.py`, `tests/test_data.py` |
| Artifact checksums and registry↔metrics consistency | `tests/test_model_registry.py`, `tests/test_ml_pipeline.py` |
| Explanations agree with the scored vector | `tests/test_explain.py` |

Run everything offline with `pytest -m "not network"`.
