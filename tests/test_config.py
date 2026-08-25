"""Credential resolution — the part that decided whether v1 started at all."""

from __future__ import annotations

import pytest

from geoexplorer.config import (
    SUPPORTED_MODELS,
    MissingCredentialError,
    Settings,
    get_settings,
    load_streamlit_secrets_into_env,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("HERE_API_KEY", "OPENAI_API_KEY", "GEOEXPLORER_MODEL", "GEOEXPLORER_CACHE_TTL"):
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

    assert settings.model == SUPPORTED_MODELS[0]
    assert settings.request_timeout_seconds > 0
    assert settings.max_retries >= 0


def test_settings_are_cached_so_streamlit_reruns_are_cheap():
    assert get_settings() is get_settings()


def test_secrets_are_copied_into_the_environment(monkeypatch):
    monkeypatch.setattr(
        "streamlit.secrets", {"HERE_API_KEY": "from-secrets"}, raising=False
    )
    load_streamlit_secrets_into_env()

    assert Settings(_env_file=None).require_here_key() == "from-secrets"


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


def test_every_supported_model_is_distinct_and_non_empty():
    assert len(set(SUPPORTED_MODELS)) == len(SUPPORTED_MODELS)
    assert all(SUPPORTED_MODELS)
