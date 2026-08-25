"""Agent wiring, the streaming bridge, and the memory bug the prototype shipped with."""

from __future__ import annotations

import pytest
from llama_index.core.agent.workflow import FunctionAgent

from conftest import geocode_payload
from geoexplorer.agent import (
    MAX_TOOL_ITERATIONS,
    AgentSession,
    StreamEvent,
    _summarise_tool_output,
    build_session,
    humanise_error,
)
from geoexplorer.models import GeocodeResult
from geoexplorer.providers import PROVIDERS


@pytest.fixture
def geocode() -> GeocodeResult:
    return GeocodeResult.from_here("350 5th Ave", geocode_payload()["items"][0])


def test_session_builds_with_every_tool_registered(client_factory, geocode):
    session = build_session(
        here_client=client_factory({}),
        provider=PROVIDERS["openai"],
        api_key="sk-test",
        model="gpt-4.1-mini",
        geocode=geocode,
    )

    assert isinstance(session, AgentSession)
    assert len(session.agent.tools) == 7
    assert session.anchor_label.startswith("350 5th Ave")


def test_session_anchors_the_map_on_the_resolved_address(client_factory, geocode):
    session = build_session(
        here_client=client_factory({}),
        provider=PROVIDERS["openai"],
        api_key="sk-test",
        model="gpt-4.1-mini",
        geocode=geocode,
    )
    snapshot = session.map_snapshot()

    assert snapshot.anchor.coordinate.lat == pytest.approx(40.748444)
    assert snapshot.places == []


def test_memory_and_context_persist_across_turns(client_factory, geocode):
    """The prototype rebuilt the agent on every Streamlit rerun, wiping memory each turn.

    Building a session must therefore produce objects the caller can hold onto — this
    asserts the identity that the Streamlit layer depends on.
    """
    session = build_session(
        here_client=client_factory({}),
        provider=PROVIDERS["openai"],
        api_key="sk-test",
        model="gpt-4.1-mini",
        geocode=geocode,
    )
    first_context, first_memory = session.context, session.memory

    session.state.record_places([])

    assert session.context is first_context
    assert session.memory is first_memory


def test_streaming_surfaces_errors_instead_of_crashing(client_factory, geocode, monkeypatch):
    session = build_session(
        here_client=client_factory({}),
        provider=PROVIDERS["openai"],
        api_key="sk-test",
        model="gpt-4.1-mini",
        geocode=geocode,
    )

    def explode(*args, **kwargs):
        raise RuntimeError("Error code: 401 - invalid_api_key")

    # FunctionAgent is a pydantic model, so the patch goes on the class.
    monkeypatch.setattr(FunctionAgent, "run", explode)
    events = list(session.stream("anything"))

    assert [e.kind for e in events] == ["error"]
    assert "rejected" in events[0].text


def test_tool_iteration_cap_is_set():
    """Without a cap, a confused agent can loop until the request times out."""
    assert 1 < MAX_TOOL_ITERATIONS <= 20


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Error code: 401 invalid_api_key", "rejected"),
        ("insufficient_quota: you exceeded your current quota", "quota"),
        ("Rate limit reached", "rate-limited"),
        ("The model `gpt-9` does not exist", "does not have access"),
    ],
)
def test_provider_errors_become_actionable_advice(message, expected):
    assert expected in humanise_error(RuntimeError(message))


def test_unknown_errors_are_passed_through_not_swallowed():
    assert "something odd" in humanise_error(RuntimeError("something odd"))


def test_tool_output_summary_keeps_traces_small():
    summary = _summarise_tool_output(
        {"count": 40, "results": [{"name": f"Place {i}"} for i in range(40)]}
    )

    assert summary["count"] == 40
    assert len(summary["names"]) == 5


def test_tool_output_summary_preserves_errors():
    assert _summarise_tool_output({"error": "403 forbidden"}) == {"error": "403 forbidden"}


def test_stream_event_defaults():
    event = StreamEvent(kind="token", text="hi")
    assert event.tool_name == "" and event.payload is None


# --- the streaming bridge ----------------------------------------------------------


class FakeHandler:
    """Stands in for a WorkflowHandler: async-iterable events, and awaitable for a result."""

    def __init__(self, events: list, result: str) -> None:
        self._events = events
        self._result = result

    async def stream_events(self):
        for event in self._events:
            yield event

    def __await__(self):
        async def _result() -> str:
            return self._result

        return _result().__await__()


def _session(client_factory, geocode):
    return build_session(
        here_client=client_factory({}),
        provider=PROVIDERS["openai"],
        api_key="sk-test",
        model="gpt-4.1-mini",
        geocode=geocode,
    )


def test_stream_translates_workflow_events_in_order(client_factory, geocode, monkeypatch):
    from llama_index.core.agent.workflow import AgentStream, ToolCall, ToolCallResult
    from llama_index.core.tools import ToolOutput

    session = _session(client_factory, geocode)
    events = [
        ToolCall(tool_name="find_nearby", tool_kwargs={"query": "coffee"}, tool_id="1"),
        ToolCallResult(
            tool_name="find_nearby",
            tool_kwargs={"query": "coffee"},
            tool_id="1",
            tool_output=ToolOutput(
                content="…",
                tool_name="find_nearby",
                raw_input={"query": "coffee"},
                raw_output={"count": 2, "results": [{"name": "A"}, {"name": "B"}]},
            ),
            return_direct=False,
        ),
        AgentStream(
            delta="Two ", response="Two ", current_agent_name="GeoExplorer",
            tool_calls=[], raw={},
        ),
        AgentStream(
            delta="cafes.", response="Two cafes.", current_agent_name="GeoExplorer",
            tool_calls=[], raw={},
        ),
    ]
    monkeypatch.setattr(
        FunctionAgent, "run", lambda *a, **k: FakeHandler(events, "Two cafes.")
    )

    streamed = list(session.stream("where is coffee"))

    assert [e.kind for e in streamed] == [
        "tool_start", "tool_end", "token", "token", "final",
    ]
    assert "".join(e.text for e in streamed if e.kind == "token") == "Two cafes."
    assert streamed[0].tool_name == "find_nearby"
    assert streamed[0].payload == {"query": "coffee"}
    assert streamed[1].payload == {"count": 2, "names": ["A", "B"]}
    assert session.tool_calls == 1


def test_stream_falls_back_to_the_final_result_when_nothing_streamed(
    client_factory, geocode, monkeypatch
):
    """Some models return a complete message with no deltas; the answer must still show."""
    session = _session(client_factory, geocode)
    monkeypatch.setattr(FunctionAgent, "run", lambda *a, **k: FakeHandler([], "All done."))

    streamed = list(session.stream("hello"))

    assert [e.kind for e in streamed] == ["final"]
    assert streamed[0].text == "All done."


def test_stream_joins_its_worker_thread(client_factory, geocode, monkeypatch):
    """A leaked thread per turn would accumulate across a long Streamlit session."""
    import threading

    session = _session(client_factory, geocode)
    monkeypatch.setattr(FunctionAgent, "run", lambda *a, **k: FakeHandler([], "done"))

    before = threading.active_count()
    list(session.stream("hello"))

    assert threading.active_count() <= before


# --- provider independence ----------------------------------------------------------


@pytest.mark.parametrize("provider_id", ["openai", "anthropic"])
def test_a_session_can_be_built_on_any_provider(client_factory, geocode, provider_id):
    """Nothing below the LLM construction differs between vendors — assert that.

    Gemini is excluded here only because its client validates the key over the network
    at construction time; `test_providers.py` covers its function-calling contract.
    """
    provider = PROVIDERS[provider_id]
    session = build_session(
        here_client=client_factory({}),
        provider=provider,
        api_key="test-key",
        model=provider.default_model,
        geocode=geocode,
    )

    assert session.provider.id == provider_id
    assert session.model == provider.default_model
    assert len(session.agent.tools) == 7
    assert session.map_snapshot().anchor.label.startswith("350 5th Ave")


def test_the_agent_prompt_and_tools_do_not_vary_by_provider(client_factory, geocode):
    sessions = [
        build_session(
            here_client=client_factory({}),
            provider=PROVIDERS[pid],
            api_key="test-key",
            model=PROVIDERS[pid].default_model,
            geocode=geocode,
        )
        for pid in ("openai", "anthropic")
    ]

    prompts = {str(s.agent.system_prompt) for s in sessions}
    tool_names = {
        tuple(sorted(t.metadata.name for t in s.agent.tools)) for s in sessions
    }

    assert len(prompts) == 1
    assert len(tool_names) == 1
