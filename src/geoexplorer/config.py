"""Configuration, resolved from environment variables or Streamlit secrets.

The original prototype read `st.secrets["HERE_API"]` directly and crashed with a bare
KeyError on any machine where that was not configured. Settings are resolved here once,
with a clear error message naming the missing key and where to put it.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Models known to handle multi-step tool calling well. The first entry is the default.
SUPPORTED_MODELS: tuple[str, ...] = (
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-5.1",
    "gpt-4.1-mini",
    "gpt-4o-mini",
)


class MissingCredentialError(RuntimeError):
    """Raised when a required API key cannot be found in any configured source."""


class Settings(BaseSettings):
    """Runtime configuration. Every field can be overridden by an environment variable."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    here_api_key: str = Field(default="", alias="HERE_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    model: str = Field(default=SUPPORTED_MODELS[0], alias="GEOEXPLORER_MODEL")
    cache_ttl_seconds: int = Field(default=900, alias="GEOEXPLORER_CACHE_TTL")
    request_timeout_seconds: float = Field(default=10.0, alias="GEOEXPLORER_TIMEOUT")
    max_retries: int = Field(default=2, alias="GEOEXPLORER_MAX_RETRIES")

    def require_here_key(self) -> str:
        if not self.here_api_key:
            raise MissingCredentialError(
                "No HERE API key found. Set HERE_API_KEY in your environment or in a "
                ".env file, or add it to .streamlit/secrets.toml as HERE_API_KEY. "
                "Free keys: https://platform.here.com/"
            )
        return self.here_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, read once.

    Streamlit re-executes the script top to bottom on every interaction, so this is
    cached to avoid re-parsing .env on each keystroke.
    """
    return Settings()


def load_streamlit_secrets_into_env() -> None:
    """Copy `st.secrets` entries into os.environ so pydantic-settings can see them.

    Streamlit Community Cloud injects secrets only via `st.secrets`, never the
    environment. Importing streamlit is deferred so the package stays usable — and
    testable — outside a Streamlit runtime.
    """
    import os

    try:
        import streamlit as st

        secrets = dict(st.secrets)
    except Exception:
        return

    for key in ("HERE_API_KEY", "OPENAI_API_KEY", "GEOEXPLORER_MODEL"):
        value = secrets.get(key)
        # An explicitly-set environment variable always wins over a secrets file.
        if value and key not in os.environ:
            os.environ[key] = str(value)

    # The prototype used the name HERE_API; accept it so old deployments keep working.
    legacy = secrets.get("HERE_API")
    if legacy and "HERE_API_KEY" not in os.environ:
        os.environ["HERE_API_KEY"] = str(legacy)

    get_settings.cache_clear()
