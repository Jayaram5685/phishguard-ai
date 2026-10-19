"""Settings contract: safe defaults, validated overrides, no silent fallbacks.

The API fails fast on a bad ``MODEL_NAME`` (path traversal, wrong extension),
and every override must be reachable from the environment -- a settings field
nobody can set is a support ticket waiting to happen.
"""

import pytest

from phishguard.config import DEFAULT_CORS_ORIGINS, Settings

ENV_VARS = (
    "MODEL_NAME",
    "API_KEY",
    "RATE_LIMIT_PER_MINUTE",
    "MAX_URL_LENGTH",
    "MAX_REQUEST_BYTES",
    "CORS_ORIGINS",
    "TRUST_PROXY",
    "LOG_LEVEL",
)


@pytest.fixture()
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_defaults_need_no_configuration(clean_env):
    settings = Settings.from_env()
    assert settings.model_name == "random_forest.joblib"
    assert settings.api_key is None
    assert settings.trust_proxy is False
    assert settings.rate_limit_per_minute == 60
    assert settings.max_url_length == 2048
    assert settings.max_request_bytes == 32_768
    assert settings.log_level == "INFO"
    assert settings.cors_origins == DEFAULT_CORS_ORIGINS


def test_model_name_can_select_any_registered_artifact(clean_env):
    clean_env.setenv("MODEL_NAME", "lightgbm.joblib")
    assert Settings.from_env().model_name == "lightgbm.joblib"


def test_empty_model_name_falls_back_to_the_default(clean_env):
    clean_env.setenv("MODEL_NAME", "   ")
    assert Settings.from_env().model_name == "random_forest.joblib"


@pytest.mark.parametrize(
    "value",
    [
        "../../etc/passwd",
        "models/../../evil.joblib",
        "nested\\evil.joblib",
        "..",
        "random_forest.pkl",
    ],
)
def test_model_name_rejects_paths_and_non_joblib_files(clean_env, value):
    clean_env.setenv("MODEL_NAME", value)
    with pytest.raises(ValueError, match="MODEL_NAME must be a bare"):
        Settings.from_env()


def test_cors_extras_extend_the_allowlist_instead_of_replacing_it(clean_env):
    clean_env.setenv("CORS_ORIGINS", "https://ops.example, https://dash.example")
    settings = Settings.from_env()
    assert settings.cors_origins[: len(DEFAULT_CORS_ORIGINS)] == DEFAULT_CORS_ORIGINS
    assert settings.cors_extra_origins == ("https://ops.example", "https://dash.example")
    assert settings.cors_origins[-2:] == settings.cors_extra_origins


def test_rate_limit_is_bounded_below(clean_env):
    clean_env.setenv("RATE_LIMIT_PER_MINUTE", "0")
    with pytest.raises(ValueError, match="RATE_LIMIT_PER_MINUTE"):
        Settings.from_env()


def test_max_url_length_is_bounded_below(clean_env):
    clean_env.setenv("MAX_URL_LENGTH", "10")
    with pytest.raises(ValueError, match="MAX_URL_LENGTH"):
        Settings.from_env()


def test_max_request_bytes_is_bounded_below(clean_env):
    clean_env.setenv("MAX_REQUEST_BYTES", "16")
    with pytest.raises(ValueError, match="MAX_REQUEST_BYTES"):
        Settings.from_env()


def test_trust_proxy_accepts_common_truthy_spellings(clean_env):
    clean_env.setenv("TRUST_PROXY", "yes")
    assert Settings.from_env().trust_proxy is True


def test_log_level_is_normalized(clean_env):
    clean_env.setenv("LOG_LEVEL", "debug")
    assert Settings.from_env().log_level == "DEBUG"


def test_api_key_is_optional_and_stripped(clean_env):
    clean_env.setenv("API_KEY", "  sekrit  ")
    assert Settings.from_env().api_key == "sekrit"
    clean_env.setenv("API_KEY", "   ")
    assert Settings.from_env().api_key is None
