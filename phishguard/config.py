"""Environment-driven configuration.

Every setting has a safe default so the app boots with zero configuration;
production overrides come from environment variables only (never committed).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Origins always allowed to call the API cross-origin.
# - ``chrome-extension://`` IDs are exactly 32 characters from the a-p alphabet.
# - localhost/127.0.0.1 with any port cover the dashboard dev server.
DEFAULT_CORS_ORIGINS: tuple[str, ...] = (
    r"chrome-extension://[a-p]{32}$",
    r"http://localhost:\d+$",
    r"http://127.0.0.1:\d+$",
)

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    cors_extra_origins: tuple[str, ...] = ()
    rate_limit_per_minute: int = 60
    api_key: str | None = None
    trust_proxy: bool = False
    max_url_length: int = 2048
    max_request_bytes: int = 32_768
    log_level: str = "INFO"
    model_name: str = "random_forest.joblib"
    cors_origins: tuple[str, ...] = field(default=DEFAULT_CORS_ORIGINS)

    @classmethod
    def from_env(cls) -> Settings:
        extra = tuple(
            origin.strip()
            for origin in os.environ.get("CORS_ORIGINS", "").split(",")
            if origin.strip()
        )
        api_key = os.environ.get("API_KEY", "").strip() or None
        trust_proxy = os.environ.get("TRUST_PROXY", "").strip().lower() in _TRUE_VALUES
        return cls(
            cors_extra_origins=extra,
            cors_origins=DEFAULT_CORS_ORIGINS + extra,
            rate_limit_per_minute=_env_int("RATE_LIMIT_PER_MINUTE", 60),
            api_key=api_key,
            trust_proxy=trust_proxy,
            max_url_length=_env_int("MAX_URL_LENGTH", 2048, minimum=64),
            log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        )
