"""Credential resolution — the part that decided whether v1 started at all."""

from __future__ import annotations

import pytest

from geoexplorer.config import (
    MissingCredentialError,
    Settings,
    get_settings,
    load_streamlit_secrets_into_env,
)

MODEL_KEY_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "CLAUDE_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("HERE_API_KEY", "GEOEXPLORER_MODEL", "GEOEXPLORER_CACHE_TTL", *MODEL_KEY_VARS):
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_missing_here_key_names_every_place_it_could_go():
    """v1 raised a bare KeyError from `st.secrets["HERE_API"]` and showed a traceback."""
    with pytest.raises(MissingCredentialError) as excinfo:
        Settings(_env_file=None).require_here_key()

    message = str(excinfo.value)
    assert "HERE_API_KEY" in message
    assert ".env" in message and "secrets.toml" in message
    assert "platform.here.com" in message


def test_settings_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("HERE_API_KEY", "here-123")
    monkeypatch.setenv("GEOEXPLORER_CACHE_TTL", "60")

    settings = Settings(_env_file=None)

    assert settings.require_here_key() == "here-123"
    assert settings.cache_ttl_seconds == 60


def test_defaults_are_usable_without_any_configuration():
    settings = Settings(_env_file=None)

    assert settings.request_timeout_seconds > 0
    assert settings.max_retries >= 0
    assert settings.resolve_model_credentials() is None


def test_settings_are_cached_so_streamlit_reruns_are_cheap():
    assert get_settings() is get_settings()


# --- provider resolution ------------------------------------------------------------


@pytest.mark.parametrize(
    ("env_var", "expected_provider"),
    [
        ("OPENAI_API_KEY", "openai"),
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("CLAUDE_API_KEY", "anthropic"),
        ("GOOGLE_API_KEY", "google"),
        ("GEMINI_API_KEY", "google"),
    ],
)
def test_any_single_provider_key_is_enough_to_start(monkeypatch, env_var, expected_provider):
    monkeypatch.setenv(env_var, "test-key")

    provider, api_key, model = Settings(_env_file=None).resolve_model_credentials()

    assert provider.id == expected_provider
    assert api_key == "test-key"
    assert model == provider.default_model


def test_the_model_setting_selects_among_configured_providers(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("GEOEXPLORER_MODEL", "claude-sonnet-5")

    provider, api_key, model = Settings(_env_file=None).resolve_model_credentials()

    assert provider.id == "anthropic"
    assert api_key == "anthropic-key"
    assert model == "claude-sonnet-5"


def test_naming_a_model_whose_provider_has_no_key_does_not_pick_it(monkeypatch):
    """Falling through to a different vendor's model would be a surprising silent switch."""
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("GEOEXPLORER_MODEL", "claude-opus-5")

    provider, _, model = Settings(_env_file=None).resolve_model_credentials()

    assert provider.id == "openai"
    assert model == provider.default_model


def test_an_unknown_model_name_falls_back_to_the_provider_default(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("GEOEXPLORER_MODEL", "gpt-imaginary")

    provider, _, model = Settings(_env_file=None).resolve_model_credentials()

    assert model == provider.default_model


# --- streamlit secrets --------------------------------------------------------------


def test_secrets_are_copied_into_the_environment(monkeypatch):
    monkeypatch.setattr("streamlit.secrets", {"HERE_API_KEY": "from-secrets"}, raising=False)
    load_streamlit_secrets_into_env()

    assert Settings(_env_file=None).require_here_key() == "from-secrets"


def test_model_provider_keys_come_through_secrets_too(monkeypatch):
    monkeypatch.setattr(
        "streamlit.secrets", {"ANTHROPIC_API_KEY": "sk-ant-secret"}, raising=False
    )
    load_streamlit_secrets_into_env()

    provider, api_key, _ = Settings(_env_file=None).resolve_model_credentials()
    assert provider.id == "anthropic"
    assert api_key == "sk-ant-secret"


def test_the_legacy_here_api_name_still_works(monkeypatch):
    """v1 deployments used `HERE_API`; those should not break on upgrade."""
    monkeypatch.setattr("streamlit.secrets", {"HERE_API": "legacy-key"}, raising=False)
    load_streamlit_secrets_into_env()

    assert Settings(_env_file=None).require_here_key() == "legacy-key"


def test_an_explicit_environment_variable_beats_a_secrets_file(monkeypatch):
    monkeypatch.setenv("HERE_API_KEY", "from-env")
    monkeypatch.setattr("streamlit.secrets", {"HERE_API_KEY": "from-secrets"}, raising=False)
    load_streamlit_secrets_into_env()

    assert Settings(_env_file=None).require_here_key() == "from-env"


def test_no_secrets_file_is_not_an_error(monkeypatch):
    class Exploding:
        def __iter__(self):
            raise FileNotFoundError("no secrets.toml here")

    monkeypatch.setattr("streamlit.secrets", Exploding(), raising=False)
    load_streamlit_secrets_into_env()  # must not raise
