"""Agent construction and a synchronous streaming bridge for Streamlit.

The prototype had two structural bugs here.

First, the agent was rebuilt inside the top-level `if OPEN_AI_API_KEY and address:`
block. Streamlit re-runs the script top to bottom on every widget interaction, so a
fresh `ReActAgent` — with a fresh, empty memory — replaced the old one before every
single message. The advertised "built-in memory" never survived a turn.

Second, memory was faked by concatenating past user messages into the prompt string.
That grows without bound, omits the assistant's own replies, and omits tool results
entirely — so the model could not actually refer back to anything it had found.

Both are fixed by holding one `AgentSession` in `st.session_state`, keyed by
configuration, carrying a long-lived workflow `Context` and a token-bounded `Memory`
across turns.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from llama_index.core.agent.workflow import (
    AgentOutput,
    AgentStream,
    FunctionAgent,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.memory import Memory
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

from .here import HereClient
from .models import GeocodeResult
from .state import Anchor, MapSnapshot, SessionState
from .tools import build_tools

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 12
MEMORY_TOKEN_LIMIT = 24_000

SYSTEM_PROMPT = """\
You are GeoExplorer, a local-knowledge assistant answering questions about one specific \
address. Every tool you have already knows that address — you never need to pass it.

Ground rules:

* Answer only from tool results. You have no reliable prior knowledge about what stands \
at any particular address, and a plausible-sounding guess about a real neighbourhood is \
worse than admitting you do not know.
* If a search comes back empty, say so plainly. An empty result means HERE has no record \
within that radius, which is not the same as "nothing exists there" — widen the radius or \
rephrase once before concluding.
* Prefer profile_neighbourhood for open-ended "what is it like here" questions, \
find_by_category for common amenities, and find_nearby for specific or unusual ones.
* For "how far" or "how long to get to" questions use travel_time_to. Never estimate \
travel time from the straight-line distance in a search result; in a city the two differ \
substantially.
* Report distances the way a person would: "about a 5 minute walk" alongside "400 m".
* Name the places you found. A useful answer lists them; it does not just count them.
* Never invent a phone number, opening hours, rating or price. If a tool did not return \
the field, say it is not available.
* Keep answers short. Two or three sentences plus a list, unless asked for more.
"""


@dataclass
class StreamEvent:
    """One thing that happened during a turn, flattened for the UI to render."""

    kind: str  # "token" | "tool_start" | "tool_end" | "final" | "error"
    text: str = ""
    tool_name: str = ""
    payload: Any = None


class AgentSession:
    """One conversation: an agent, its tools, and the state that outlives a turn."""

    def __init__(
        self,
        *,
        agent: FunctionAgent,
        context: Context,
        memory: Memory,
        client: HereClient,
        state: SessionState,
        model: str,
    ) -> None:
        self.agent = agent
        self.context = context
        self.memory = memory
        self.client = client
        self.state = state
        self.model = model
        self.tool_calls = 0

    @property
    def anchor_label(self) -> str:
        return self.state.anchor.label

    def map_snapshot(self) -> MapSnapshot:
        """Everything the map should draw right now."""
        return self.state.snapshot()

    def stream(self, user_message: str) -> Iterator[StreamEvent]:
        """Run one turn, yielding events as they happen.

        Streamlit's script thread is synchronous, so the async workflow runs on its own
        event loop in a worker thread and hands events back over a queue. Consuming this
        generator to exhaustion is what joins that thread.
        """
        events: queue.Queue[StreamEvent | None] = queue.Queue()

        def worker() -> None:
            try:
                asyncio.run(self._pump(user_message, events))
            except Exception as exc:
                logger.exception("Agent turn failed")
                events.put(StreamEvent(kind="error", text=_humanise_error(exc)))
            finally:
                events.put(None)

        thread = threading.Thread(target=worker, name="geoexplorer-agent", daemon=True)
        thread.start()
        try:
            while True:
                event = events.get()
                if event is None:
                    break
                yield event
        finally:
            thread.join(timeout=5)

    async def _pump(self, user_message: str, events: queue.Queue[StreamEvent | None]) -> None:
        handler = self.agent.run(
            user_msg=user_message,
            ctx=self.context,
            memory=self.memory,
            max_iterations=MAX_TOOL_ITERATIONS,
        )

        async for event in handler.stream_events():
            if isinstance(event, AgentStream):
                if event.delta:
                    events.put(StreamEvent(kind="token", text=event.delta))
            elif isinstance(event, ToolCallResult):
                events.put(
                    StreamEvent(
                        kind="tool_end",
                        tool_name=event.tool_name,
                        payload=_summarise_tool_output(event.tool_output),
                    )
                )
            elif isinstance(event, ToolCall):
                self.tool_calls += 1
                events.put(
                    StreamEvent(
                        kind="tool_start",
                        tool_name=event.tool_name,
                        payload=dict(event.tool_kwargs),
                    )
                )

        result: AgentOutput = await handler
        events.put(StreamEvent(kind="final", text=str(result)))


def build_session(
    *,
    here_client: HereClient,
    openai_api_key: str,
    model: str,
    geocode: GeocodeResult,
) -> AgentSession:
    """Create a session anchored on an already-resolved address."""
    llm = OpenAI(
        model=model,
        api_key=openai_api_key,
        temperature=0.1,
        # A stalled tool-calling loop should fail fast rather than hang the UI.
        timeout=60.0,
        max_retries=2,
    )
    state = SessionState(Anchor.from_geocode(geocode))
    agent = FunctionAgent(
        name="GeoExplorer",
        description="Answers questions about the neighbourhood around one address.",
        tools=build_tools(here_client, state),
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
    )
    return AgentSession(
        agent=agent,
        context=Context(agent),
        memory=Memory.from_defaults(
            session_id="geoexplorer", token_limit=MEMORY_TOKEN_LIMIT
        ),
        client=here_client,
        state=state,
        model=model,
    )


def _summarise_tool_output(tool_output: Any) -> Any:
    """Trim a tool result down to what is worth showing in the UI trace."""
    raw = getattr(tool_output, "raw_output", tool_output)
    if isinstance(raw, dict):
        if "error" in raw:
            return {"error": raw["error"]}
        results = raw.get("results")
        if isinstance(results, list):
            return {
                "count": raw.get("count", len(results)),
                "names": [r.get("name") for r in results[:5] if isinstance(r, dict)],
            }
        return raw
    return str(raw)[:400]


def _humanise_error(exc: Exception) -> str:
    """Turn provider exceptions into something a user can act on."""
    text = str(exc)
    lowered = text.lower()
    if "authentication" in lowered or "invalid_api_key" in lowered or "401" in lowered:
        return "That OpenAI API key was rejected. Check it and try again."
    if "insufficient_quota" in lowered or "exceeded your current quota" in lowered:
        return "This OpenAI key has no remaining quota — add credit or use another key."
    if "rate limit" in lowered or "429" in lowered:
        return "OpenAI rate-limited the request. Wait a few seconds and ask again."
    if "model_not_found" in lowered or "does not exist" in lowered:
        return (
            "This OpenAI key does not have access to the selected model. "
            "Pick a different one in the sidebar."
        )
    return f"The agent could not complete that turn: {text}"
