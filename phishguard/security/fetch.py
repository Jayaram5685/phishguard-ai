"""Guarded HTTP fetching with SSRF-safe redirect handling.

Every hop of a redirect chain is re-validated through
:func:`phishguard.security.ssrf.validate_url`, so a public URL that redirects
to ``http://169.254.169.254/`` is stopped before the request is issued.

Design notes
------------
* The network call is behind an injectable ``transport`` callable so the
  redirect/limit logic is unit-testable without sockets.
* Responses are streamed and hard-capped at ``max_bytes``; a hostile server
  cannot exhaust memory.
* Failure returns ``None`` (fail closed) and is never raised to callers.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urljoin

import requests

from phishguard.security.ssrf import BlockedURLError, validate_url

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = (2.05, 5.0)
MAX_REDIRECTS = 5
MAX_RESPONSE_BYTES = 1_048_576
CHUNK_SIZE = 65_536
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


@dataclass(frozen=True)
class RawResponse:
    """Transport-level response. Header keys are lowercased by consumers."""

    status_code: int
    headers: dict[str, str]
    body: bytes


@dataclass(frozen=True)
class FetchedPage:
    """Static snapshot of a fetched page. Never executes page JavaScript."""

    url: str
    status_code: int
    text: str
    history: list[str]


Transport = Callable[..., RawResponse | None]


def _requests_transport(
    url: str,
    *,
    timeout: tuple[float, float],
    max_bytes: int,
) -> RawResponse | None:
    try:
        with requests.get(
            url,
            timeout=timeout,
            allow_redirects=False,
            stream=True,
        ) as response:
            body = bytearray()
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                body.extend(chunk)
                if len(body) >= max_bytes:
                    break
            return RawResponse(
                status_code=response.status_code,
                headers={k.lower(): v for k, v in response.headers.items()},
                body=bytes(body[:max_bytes]),
            )
    except requests.RequestException:
        return None


def _decode(body: bytes, headers: dict[str, str]) -> str:
    charset = "utf-8"
    content_type = headers.get("content-type", "")
    if "charset=" in content_type:
        charset = content_type.split("charset=", 1)[1].split(";", 1)[0].strip() or "utf-8"
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def safe_fetch(
    url: str,
    *,
    transport: Transport | None = None,
    timeout: tuple[float, float] = DEFAULT_TIMEOUT,
    max_redirects: int = MAX_REDIRECTS,
    max_bytes: int = MAX_RESPONSE_BYTES,
) -> FetchedPage | None:
    """Fetch ``url`` defensively, or return ``None`` if anything is unsafe.

    ``None`` means "no page intelligence available" -- callers must degrade to
    their no-response path rather than assume the page is benign or malicious.
    """
    do_transport = transport or _requests_transport
    current = url
    history: list[str] = []

    for _hop in range(max_redirects + 1):
        try:
            validated = validate_url(current)
        except BlockedURLError as exc:
            log.info("fetch_blocked reason=%s hop=%d", exc.reason, len(history))
            return None

        raw = do_transport(validated.url, timeout=timeout, max_bytes=max_bytes)
        if raw is None:
            return None

        headers = {k.lower(): v for k, v in raw.headers.items()}

        if raw.status_code in REDIRECT_STATUSES:
            location = headers.get("location")
            if not location:
                return FetchedPage(
                    url=current,
                    status_code=raw.status_code,
                    text=_decode(raw.body, headers),
                    history=history,
                )
            if _hop == max_redirects:
                log.info("fetch_blocked reason=too_many_redirects hop=%d", _hop)
                return None
            history.append(current)
            current = urljoin(current, location)
            continue

        return FetchedPage(
            url=current,
            status_code=raw.status_code,
            text=_decode(raw.body, headers),
            history=history,
        )

    return None
