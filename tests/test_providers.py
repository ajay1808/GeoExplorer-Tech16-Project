"""Provider registry and key detection — the bit that makes setup one paste."""

from __future__ import annotations

import pytest

from geoexplorer.providers import (
    PROVIDERS,
    all_models,
    detect_provider,
    provider_for_model,
    providers_configured_in_env,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for provider in PROVIDERS.values():
        for name in provider.env_vars:
            monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("api_key", "expected"),
    [
        ("sk-abc123", "openai"),
        ("sk-proj-abc123", "openai"),
        ("sk-ant-api03-abc", "anthropic"),
        ("AIzaSyAbc123", "google"),
        ("  sk-ant-padded  ", "anthropic"),
    ],
)
def test_keys_are_recognised_by_prefix(api_key, expected):
    assert detect_provider(api_key).id == expected


def test_anthropic_keys_are_not_mistaken_for_openai():
    """Both start with `sk-`; the longer prefix has to win or Claude keys go to OpenAI."""
    assert detect_provider("sk-ant-api03-xyz").id == "anthropic"


def test_an_unrecognised_key_returns_none_rather_than_guessing():
    """Proxy and gateway keys are real; guessing wrong is worse than asking."""
    assert detect_provider("my-corporate-gateway-token") is None
    assert detect_provider("") is None


def test_env_detection_finds_configured_providers(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-y")

    found = {p.id for p in providers_configured_in_env()}

    assert found == {"anthropic", "google"}


def test_blank_env_values_do_not_count_as_configured(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "   ")

    assert providers_configured_in_env() == []


def test_alias_variables_are_accepted(monkeypatch):
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-ant-alias")

    assert PROVIDERS["anthropic"].key_from_env() == "sk-ant-alias"


def test_the_canonical_variable_wins_over_its_alias(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "canonical")
    monkeypatch.setenv("CLAUDE_API_KEY", "alias")

    assert PROVIDERS["anthropic"].key_from_env() == "canonical"


def test_every_model_maps_back_to_exactly_one_provider():
    models = all_models()
    assert len(set(models)) == len(models), "a duplicate model id would make lookup ambiguous"
    for model in models:
        assert provider_for_model(model) is not None


def test_unknown_model_has_no_provider():
    assert provider_for_model("gpt-imaginary") is None


def test_building_an_unlisted_model_is_refused():
    with pytest.raises(ValueError, match="not a known"):
        PROVIDERS["openai"].build_llm("gpt-imaginary", "sk-test")


def test_every_provider_is_described_well_enough_to_show_a_user():
    for provider in PROVIDERS.values():
        assert provider.models, f"{provider.id} needs at least one model"
        assert provider.default_model == provider.models[0]
        assert provider.console_url.startswith("https://")
        assert provider.key_example
        assert provider.env_vars


def test_openai_llm_is_built_with_the_requested_model():
    llm = PROVIDERS["openai"].build_llm("gpt-4.1-mini", "sk-test")
    assert llm.metadata.model_name == "gpt-4.1-mini"


def test_anthropic_llm_raises_its_output_cap_above_the_sdk_default():
    """The integration defaults to 512 output tokens, which truncates answers."""
    llm = PROVIDERS["anthropic"].build_llm("claude-opus-5", "sk-ant-test")

    assert llm.max_tokens >= 4096
    assert llm.metadata.is_function_calling_model


def test_every_provider_supports_function_calling():
    """The agent loop is tool calls; a provider without them cannot be offered."""
    from llama_index.core.llms.function_calling import FunctionCallingLLM
    from llama_index.llms.anthropic import Anthropic
    from llama_index.llms.google_genai import GoogleGenAI
    from llama_index.llms.openai import OpenAI

    # Gemini's constructor calls the API to read model metadata, so assert on the
    # class rather than instantiating it with a fake key.
    for cls in (OpenAI, Anthropic, GoogleGenAI):
        assert issubclass(cls, FunctionCallingLLM)
