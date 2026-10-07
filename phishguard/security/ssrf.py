"""SSRF protection for every outbound network operation.

Threat model
------------
``/predict`` accepts attacker-influenced URLs (sent by the browser extension
or any API client) and the feature pipeline fetches the referenced resource to
extract page signals. Without guards, the endpoint becomes an open proxy into
the server's own network -- loopback services, RFC1918 hosts, cloud metadata
endpoints. That is the classic SSRF primitive, and it is mandatory to close
before any new feature ships.

Guarantees
----------
1. Only ``http``/``https`` are ever fetched. ``file:``, ``javascript:``,
   ``data:``, ``ftp:``, ``gopher:`` and friends are rejected before any I/O.
2. Only ports 80/443 are fetched (blocks internal service discovery through
   unusual ports such as 22, 6379, 11211).
3. The hostname must be non-empty, free of control characters (tab/newline
   smuggling), and not a blocklisted internal name (``localhost``,
   ``metadata.google.internal``, ``*.local``, ``*.internal``, ``*.svc``, ...).
4. IP literals are parsed and rejected unless globally routable: dotted-quad,
   IPv6, IPv4-mapped IPv6, decimal (``2130706433``) and hex (``0x7f000001``)
   forms -- the same forms browsers interpret.
5. Every other hostname is resolved and **all** resolved addresses must satisfy
   ``ip.is_global`` (covers loopback, RFC1918, link-local/169.254 metadata,
   CGNAT, unique-local IPv6, unspecified and reserved ranges).
6. **Fail-closed**: DNS errors, empty answers, malformed URLs and unexpected
   parse errors all raise ``BlockedURLError``; callers must not fetch.

Residual risks (explicit, tracked in docs/threat-model.md)
----------------------------------------------------------
* **DNS rebinding / TOCTOU** -- the address validated here may differ from the
  one the HTTP client ultimately connects to. Mitigated by re-validating every
  redirect hop and by short timeouts; full connection pinning is out of scope
  for this layer.
* **Resolver latency** -- ``socket.getaddrinfo`` has no portable timeout, so a
  slow resolver delays the decision (availability risk only).
"""

from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

ALLOWED_SCHEMES = frozenset({"http", "https"})
ALLOWED_PORTS = frozenset({80, 443})
MAX_URL_LENGTH = 2048

BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "metadata",
        "metadata.google.internal",
        "metadata.goog",
        "instance-data",
        "kubernetes.default",
        "kubernetes.default.svc",
    }
)

BLOCKED_HOSTNAME_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".intranet",
    ".lan",
    ".localdomain",
    ".home.arpa",
    ".svc",
    ".cluster.local",
)

_FORBIDDEN_RAW_CHARS = ("\t", "\n", "\r")
_DECIMAL_IP = re.compile(r"\d{1,12}")
_HEX_IP = re.compile(r"0x[0-9a-f]{1,8}", re.IGNORECASE)


class BlockedURLError(ValueError):
    """Raised when a URL must not be fetched.

    ``reason`` is a stable, machine-readable code suitable for logs and
    metrics; it never contains the URL itself (URLs may carry secrets).
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True)
class ValidatedURL:
    """A URL that passed every SSRF check and may be fetched."""

    url: str
    scheme: str
    hostname: str
    port: int
    resolved_ips: tuple[str, ...]


def _default_port(scheme: str) -> int:
    return 443 if scheme == "https" else 80


def _literal_ip(hostname: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse IPv4/IPv6 literals including decimal and hex spellings."""
    try:
        return ipaddress.ip_address(hostname)
    except ValueError:
        pass

    # Browsers treat a bare decimal host as an IPv4 address
    # (http://2130706433/ == http://127.0.0.1/); so must we.
    if _DECIMAL_IP.fullmatch(hostname):
        try:
            return ipaddress.ip_address(int(hostname))
        except ValueError:
            return None
    if _HEX_IP.fullmatch(hostname):
        try:
            return ipaddress.ip_address(int(hostname, 16))
        except ValueError:
            return None
    return None


def _is_globally_routable(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        return mapped.is_global
    return ip.is_global


def _check_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    if not _is_globally_routable(ip):
        raise BlockedURLError("non_global_ip", str(ip))


def _check_hostname_rules(hostname: str) -> None:
    if not hostname:
        raise BlockedURLError("host_missing")
    if any(ch in hostname for ch in _FORBIDDEN_RAW_CHARS):
        raise BlockedURLError("control_characters", "host")
    if hostname in BLOCKED_HOSTNAMES:
        raise BlockedURLError("blocked_hostname", hostname)
    if any(hostname.endswith(suffix) for suffix in BLOCKED_HOSTNAME_SUFFIXES):
        raise BlockedURLError("blocked_hostname", hostname)


def _resolve(hostname: str, port: int) -> tuple[str, ...]:
    try:
        answers = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, OSError) as exc:
        raise BlockedURLError("dns_resolution_failed", type(exc).__name__) from exc
    if not answers:
        raise BlockedURLError("dns_resolution_failed", "empty answer")

    resolved: list[str] = []
    for family, _type, _proto, _canon, sockaddr in answers:
        if family not in (socket.AF_INET, socket.AF_INET6):
            continue
        raw_ip = sockaddr[0].split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(raw_ip)
        except ValueError as exc:
            raise BlockedURLError("dns_resolution_failed", "unparsable address") from exc
        _check_ip(ip)
        resolved.append(str(ip))
    if not resolved:
        raise BlockedURLError("dns_resolution_failed", "no INET answers")
    return tuple(sorted(set(resolved)))


def validate_url(url: str, *, resolve_dns: bool = True) -> ValidatedURL:
    """Validate ``url`` for outbound fetching.

    Returns a :class:`ValidatedURL` when the URL is safe to fetch, otherwise
    raises :class:`BlockedURLError`. ``resolve_dns=False`` skips step 5 and is
    only appropriate for offline unit tests or literal-IP inputs.
    """
    if not isinstance(url, str):
        raise BlockedURLError("invalid_type", type(url).__name__)
    if not url.strip():
        raise BlockedURLError("empty_url")
    if len(url) > MAX_URL_LENGTH:
        raise BlockedURLError("url_too_long", str(len(url)))
    if any(ch in url for ch in _FORBIDDEN_RAW_CHARS):
        raise BlockedURLError("control_characters", "url")

    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise BlockedURLError("malformed_url", type(exc).__name__) from exc

    scheme = (parts.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise BlockedURLError("scheme_not_allowed", scheme or "none")

    try:
        hostname = parts.hostname
        port = parts.port
    except ValueError as exc:
        raise BlockedURLError("malformed_url", type(exc).__name__) from exc

    if hostname is None:
        raise BlockedURLError("host_missing")
    hostname = hostname.rstrip(".").lower()
    _check_hostname_rules(hostname)

    if port is None:
        port = _default_port(scheme)
    elif port not in ALLOWED_PORTS:
        raise BlockedURLError("blocked_port", str(port))

    literal = _literal_ip(hostname)
    if literal is not None:
        _check_ip(literal)
        return ValidatedURL(url, scheme, hostname, port, (str(literal),))

    if not resolve_dns:
        return ValidatedURL(url, scheme, hostname, port, ())

    return ValidatedURL(url, scheme, hostname, port, _resolve(hostname, port))


def is_public_domain_candidate(hostname: str | None) -> bool:
    """True when ``hostname`` looks like a registrable public domain.

    Gates WHOIS lookups: internal names, IP literals and malformed hosts are
    rejected so we never query a registry about ``127.0.0.1`` or
    ``metadata.google.internal``. This does *not* replace
    :func:`validate_url` -- it is a cheap pre-check for lookups that do not
    fetch from the target host.
    """
    if not hostname:
        return False
    normalized = hostname.rstrip(".").lower()
    if any(ch in normalized for ch in _FORBIDDEN_RAW_CHARS):
        return False
    if normalized in BLOCKED_HOSTNAMES:
        return False
    if any(normalized.endswith(suffix) for suffix in BLOCKED_HOSTNAME_SUFFIXES):
        return False
    if _literal_ip(normalized) is not None:
        return False
    return "." in normalized


def scheme_of(url: str) -> str:
    """Best-effort lowercase scheme (``""`` when unparsable) for API routing.

    Never raises: this is only used to decide whether a request can be
    analysed at all, not to authorise a fetch.
    """
    if not isinstance(url, str):
        return ""
    try:
        return urlsplit(url).scheme.lower()
    except ValueError:
        return ""


def is_supported_scheme(url: str) -> bool:
    return scheme_of(url) in ALLOWED_SCHEMES
