"""Model providers: OpenAI, Anthropic and Google Gemini.

The agent loop needs nothing from a provider beyond reliable function calling, so the
provider is a runtime choice rather than a rewrite. This module is the only place that
knows any provider-specific detail; `agent.py` just asks for an LLM.

Keys are recognised by prefix, which is what makes the UI simple: paste a key and the
app works out who issued it. Nobody should have to tell an app something it can read
off the credential itself.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — import cost is real, the type is not
    from llama_index.core.llms.function_calling import FunctionCallingLLM

# Anthropic's integration defaults to 512 output tokens, which truncates mid-answer.
MAX_OUTPUT_TOKENS = 4096
TEMPERATURE = 0.1
REQUEST_TIMEOUT = 60.0


@dataclass(frozen=True)
class Provider:
    """Everything the app needs to know about one model vendor."""

    id: str
    label: str
    env_vars: tuple[str, ...]
    key_prefixes: tuple[str, ...]
    models: tuple[str, ...]
    console_url: str
    key_example: str
    _build: Callable[[str, str], FunctionCallingLLM] = field(repr=False, default=None)  # type: ignore[assignment]

    @property
    def default_model(self) -> str:
        return self.models[0]

    def key_from_env(self) -> str:
        """The first non-empty value among this provider's accepted variable names."""
        for name in self.env_vars:
            value = os.environ.get(name, "").strip()
            if value:
                return value
        return ""

    def build_llm(self, model: str, api_key: str) -> FunctionCallingLLM:
        if model not in self.models:
            raise ValueError(
                f"{model!r} is not a known {self.label} model. Choose one of: "
                f"{', '.join(self.models)}."
            )
        return self._build(model, api_key)


def _build_openai(model: str, api_key: str) -> FunctionCallingLLM:
    from llama_index.llms.openai import OpenAI

    return OpenAI(
        model=model,
        api_key=api_key,
        temperature=TEMPERATURE,
        timeout=REQUEST_TIMEOUT,
        max_retries=2,
    )


def _build_anthropic(model: str, api_key: str) -> FunctionCallingLLM:
    from llama_index.llms.anthropic import Anthropic

    return Anthropic(
        model=model,
        api_key=api_key,
        temperature=TEMPERATURE,
        # Without this the integration caps output at 512 tokens and answers get cut off.
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=REQUEST_TIMEOUT,
        max_retries=2,
    )


def _build_google(model: str, api_key: str) -> FunctionCallingLLM:
    from llama_index.llms.google_genai import GoogleGenAI

    # Note: this constructor calls the Gemini API to read model metadata, so an invalid
    # key raises here rather than on the first message. build_session catches it.
    return GoogleGenAI(
        model=model,
        api_key=api_key,
        temperature=TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
        max_retries=2,
    )


PROVIDERS: dict[str, Provider] = {
    "openai": Provider(
        id="openai",
        label="OpenAI",
        env_vars=("OPENAI_API_KEY",),
        key_prefixes=("sk-proj-", "sk-"),
        models=("gpt-5.4-mini", "gpt-5.4", "gpt-5.1", "gpt-4.1-mini"),
        console_url="https://platform.openai.com/api-keys",
        key_example="sk-…",
        _build=_build_openai,
    ),
    "anthropic": Provider(
        id="anthropic",
        label="Anthropic (Claude)",
        env_vars=("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
        key_prefixes=("sk-ant-",),
        models=("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"),
        console_url="https://console.anthropic.com/settings/keys",
        key_example="sk-ant-…",
        _build=_build_anthropic,
    ),
    "google": Provider(
        id="google",
        label="Google (Gemini)",
        env_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        key_prefixes=("AIza",),
        models=("gemini-3.7-flash", "gemini-3.6-flash", "gemini-2.5-flash"),
        console_url="https://aistudio.google.com/apikey",
        key_example="AIza…",
        _build=_build_google,
    ),
}

# Longest prefix first, so `sk-ant-` is never swallowed by OpenAI's `sk-`.
_PREFIX_TO_PROVIDER: tuple[tuple[str, str], ...] = tuple(
    sorted(
        ((prefix, provider.id) for provider in PROVIDERS.values()
         for prefix in provider.key_prefixes),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
)


def detect_provider(api_key: str) -> Provider | None:
    """Work out who issued a key from its prefix.

    Returns None for anything unrecognised — a proxy key or a self-hosted gateway —
    so the caller can fall back to asking rather than guessing wrong.
    """
    candidate = api_key.strip()
    for prefix, provider_id in _PREFIX_TO_PROVIDER:
        if candidate.startswith(prefix):
            return PROVIDERS[provider_id]
    return None


def providers_configured_in_env() -> list[Provider]:
    """Providers whose key is already present in the environment, in listing order."""
    return [provider for provider in PROVIDERS.values() if provider.key_from_env()]


def provider_for_model(model: str) -> Provider | None:
    """Reverse lookup, used to keep a saved model choice pointed at the right vendor."""
    for provider in PROVIDERS.values():
        if model in provider.models:
            return provider
    return None


def all_models() -> list[str]:
    return [model for provider in PROVIDERS.values() for model in provider.models]
