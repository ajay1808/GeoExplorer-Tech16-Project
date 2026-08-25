"""GeoExplorer — Streamlit entry point.

Run with:  streamlit run app.py

This module is deliberately thin. It owns widgets, session lifecycle and rendering;
everything else lives in `src/geoexplorer/` where it can be tested without a browser.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

import streamlit as st

# Support `streamlit run app.py` from a checkout without an editable install.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from geoexplorer import __version__
from geoexplorer.agent import AgentSession, build_session, humanise_error
from geoexplorer.config import (
    MissingCredentialError,
    get_settings,
    load_streamlit_secrets_into_env,
)
from geoexplorer.here import AddressNotFoundError, HereAPIError, HereClient
from geoexplorer.providers import (
    PROVIDERS,
    Provider,
    detect_provider,
    providers_configured_in_env,
)
from geoexplorer.ui.map import build_deck

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

EXAMPLE_QUESTIONS = [
    "What's this neighbourhood like?",
    "Where's the nearest good coffee?",
    "How far is the closest pharmacy?",
    "What can I reach in a 15 minute walk?",
]


def main() -> None:
    st.set_page_config(
        page_title="GeoExplorer",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_streamlit_secrets_into_env()
    settings = get_settings()

    st.title("GeoExplorer 🌍")
    st.caption(
        "Ask questions about any address. Every answer is grounded in a live lookup "
        "against the HERE Location Services APIs — nothing is answered from memory."
    )

    credentials = render_sidebar(settings)

    try:
        here_key = settings.require_here_key()
    except MissingCredentialError as exc:
        st.error(str(exc))
        st.stop()

    address = st.text_input(
        "Address",
        placeholder="350 5th Ave, New York, NY 10118",
        help="A street address, a landmark, or a neighbourhood name.",
    )

    if not address:
        st.info("Enter an address above to begin.")
        return
    if credentials is None:
        st.warning(
            "Add an API key in the sidebar to start asking questions — "
            "OpenAI, Anthropic (Claude) or Google (Gemini) all work."
        )
        return

    provider, api_key, model = credentials
    session = ensure_session(
        address=address, here_key=here_key, provider=provider,
        api_key=api_key, model=model, settings=settings,
    )
    if session is None:
        return

    chat_column, map_column = st.columns([3, 2], gap="large")
    with map_column:
        render_map(session)
    with chat_column:
        render_chat(session)


# --------------------------------------------------------------------------- sidebar


def render_sidebar(settings: object) -> tuple[Provider, str, str] | None:
    """Collect model credentials.

    Three paths, in order of how little the user has to do:

    1. A key is already in the environment — nothing to fill in.
    2. Several are — one radio to pick between them.
    3. None are — one text box. The provider is inferred from the key's prefix, so
       there is no "which vendor is this?" question to answer first.
    """
    with st.sidebar:
        st.subheader("Model")

        from_env = settings.resolve_model_credentials()  # type: ignore[attr-defined]
        configured = providers_configured_in_env()

        if len(configured) > 1:
            provider = st.radio(
                "Provider",
                configured,
                format_func=lambda p: p.label,
                help="Keys for more than one provider were found in the environment.",
            )
            api_key = provider.key_from_env()
            st.caption(f"Key loaded from `{_env_var_in_use(provider)}`.")
        elif from_env is not None:
            provider, api_key, _ = from_env
            st.success(f"{provider.label} key loaded from the environment.")
        else:
            provider, api_key = _ask_for_a_key()

        model = None
        if provider is not None:
            default_model = _preferred_model(provider, settings)
            model = st.selectbox(
                "Model",
                provider.models,
                index=provider.models.index(default_model),
                help="If your key lacks access to a model, pick another one here.",
            )

        st.divider()
        if st.button("Reset conversation", width="stretch"):
            reset_session()
            st.rerun()

        session = st.session_state.get("session")
        if isinstance(session, AgentSession):
            stats = session.client.cache_stats
            st.caption(
                f"{session.tool_calls} tool calls this session · "
                f"{stats['hits']} cached HERE responses reused"
            )

        st.divider()
        st.caption(f"GeoExplorer v{__version__}")

    if provider is None or not api_key or model is None:
        return None
    return provider, api_key, model


def _ask_for_a_key() -> tuple[Provider | None, str]:
    """One box for any provider's key, identified by its prefix as it is typed."""
    api_key = st.text_input(
        "API key",
        type="password",
        placeholder="sk-… · sk-ant-… · AIza…",
        help="Used only for this session and never stored.",
    ).strip()

    if not api_key:
        st.caption("Paste a key from any of these:")
        for candidate in PROVIDERS.values():
            st.caption(
                f"· [{candidate.label}]({candidate.console_url}) — `{candidate.key_example}`"
            )
        return None, ""

    provider = detect_provider(api_key)
    if provider is None:
        # An unrecognised prefix is not necessarily wrong — proxies and gateways
        # reissue keys — so offer the choice instead of rejecting it.
        st.warning("Could not tell which provider issued that key.")
        provider = st.selectbox(
            "Provider", list(PROVIDERS.values()), format_func=lambda p: p.label
        )
    else:
        st.success(f"{provider.label} key detected.")
    return provider, api_key


def _preferred_model(provider: Provider, settings: object) -> str:
    """Keep the user's selection when it belongs to this provider, else use the default."""
    configured = getattr(settings, "model", "") or ""
    if configured in provider.models:
        return configured
    return provider.default_model


def _env_var_in_use(provider: Provider) -> str:
    import os

    for name in provider.env_vars:
        if os.environ.get(name, "").strip():
            return name
    return provider.env_vars[0]


# --------------------------------------------------------------------------- session


def session_fingerprint(address: str, provider_id: str, model: str, api_key: str) -> str:
    """Identify a session by its inputs, so it is rebuilt only when one actually changes.

    The key is hashed so no credential ends up sitting in Streamlit's session state
    under a readable name.
    """
    digest = hashlib.sha256(
        f"{address}|{provider_id}|{model}|{api_key}".encode()
    ).hexdigest()
    return digest[:16]


def reset_session() -> None:
    session = st.session_state.pop("session", None)
    if isinstance(session, AgentSession):
        session.client.close()
    st.session_state.pop("session_fingerprint", None)
    st.session_state["messages"] = []


def ensure_session(
    *,
    address: str,
    here_key: str,
    provider: Provider,
    api_key: str,
    model: str,
    settings: object,
) -> AgentSession | None:
    """Return the live session, building a new one when the inputs have changed."""
    fingerprint = session_fingerprint(address, provider.id, model, api_key)
    if st.session_state.get("session_fingerprint") == fingerprint:
        return st.session_state["session"]

    reset_session()

    client = HereClient(
        api_key=here_key,
        timeout=getattr(settings, "request_timeout_seconds", 10.0),
        max_retries=getattr(settings, "max_retries", 2),
        cache_ttl=getattr(settings, "cache_ttl_seconds", 900),
    )

    with st.spinner(f"Locating {address}…"):
        try:
            geocode = client.geocode(address)
        except AddressNotFoundError:
            client.close()
            st.error(
                f"HERE could not find “{address}”. Try adding a city, postcode or country."
            )
            return None
        except HereAPIError as exc:
            client.close()
            st.error(str(exc))
            return None

    try:
        with st.spinner(f"Starting {provider.label}…"):
            session = build_session(
                here_client=client,
                provider=provider,
                api_key=api_key,
                model=model,
                geocode=geocode,
            )
    except Exception as exc:
        client.close()
        st.error(f"Could not start {provider.label}: {humanise_error(exc)}")
        return None

    st.session_state["session"] = session
    st.session_state["session_fingerprint"] = fingerprint
    st.session_state["messages"] = []

    if not geocode.is_precise:
        st.warning(
            f"That matched **{geocode.address.label}**, which is a broad area rather than "
            "a specific building. Answers will be relative to its centre point."
        )
    return session


# ---------------------------------------------------------------------------- render


def render_map(session: AgentSession) -> None:
    snapshot = session.map_snapshot()
    st.markdown(f"**{snapshot.anchor.label}**")
    st.pydeck_chart(build_deck(snapshot), width="stretch")
    if snapshot.places:
        st.caption(
            f"{len(snapshot.places)} places found so far · red marker is the address"
        )
    else:
        st.caption("Places the agent finds will appear here.")


def render_chat(session: AgentSession) -> None:
    messages = st.session_state.setdefault("messages", [])

    if not messages:
        st.markdown("**Try asking**")
        columns = st.columns(2)
        for index, question in enumerate(EXAMPLE_QUESTIONS):
            if columns[index % 2].button(question, width="stretch", key=f"eg{index}"):
                st.session_state["pending_question"] = question
                st.rerun()

    for message in messages:
        with st.chat_message(message["role"]):
            if message.get("tools"):
                render_tool_trace(message["tools"])
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about this neighbourhood…") or st.session_state.pop(
        "pending_question", None
    )
    if not prompt:
        return

    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer, tool_trace = stream_turn(session, prompt)

    messages.append({"role": "assistant", "content": answer, "tools": tool_trace})
    # Re-run so the map picks up whatever the tools just discovered.
    st.rerun()


def stream_turn(session: AgentSession, prompt: str) -> tuple[str, list[dict[str, object]]]:
    """Stream one agent turn into the page, returning the answer and the tool trace."""
    status = st.status("Working…", expanded=False)
    body = st.empty()

    text = ""
    trace: list[dict[str, object]] = []
    failed = False

    for event in session.stream(prompt):
        if event.kind == "token":
            text += event.text
            body.markdown(text + "▌")
        elif event.kind == "tool_start":
            status.update(label=f"Calling `{event.tool_name}`…")
            status.write({"tool": event.tool_name, "arguments": event.payload})
            trace.append({"tool": event.tool_name, "arguments": event.payload})
        elif event.kind == "tool_end":
            status.write({"result": event.payload})
            if trace:
                trace[-1]["result"] = event.payload
        elif event.kind == "error":
            failed = True
            text = f":red[{event.text}]"
        elif event.kind == "final" and not text:
            text = event.text

    body.markdown(text)
    status.update(
        label="Failed" if failed else f"Used {len(trace)} tool call(s)",
        state="error" if failed else "complete",
    )
    return text, trace


def render_tool_trace(trace: list[dict[str, object]]) -> None:
    with st.expander(f"{len(trace)} tool call(s)", expanded=False):
        for step in trace:
            st.write(step)


if __name__ == "__main__":
    main()
