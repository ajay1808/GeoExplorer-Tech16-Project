# GeoExplorer

A conversational agent for exploring a neighbourhood by address. Ask it anything about
a location — what's nearby, how far the pharmacy is, whether the area is walkable — and
it answers from live lookups against the [HERE Location Services](https://platform.here.com/)
APIs, drawing what it finds on a map beside the conversation.

Every claim in an answer comes from a tool call. The agent is instructed to say it does
not know rather than fill a gap from the model's own priors, because a confident guess
about a real street corner is worse than no answer at all.

Bring your own key from **OpenAI, Anthropic (Claude), or Google (Gemini)** — the app
identifies the provider from the key itself.

```
┌─ Streamlit UI ──────────────────────────────────────────────┐
│  chat + streaming tool trace          │  pydeck map         │
└───────────────┬───────────────────────┴──────────┬──────────┘
                │                                  │
       ┌────────▼─────────┐              ┌─────────▼─────────┐
       │  AgentSession    │              │   SessionState    │
       │  FunctionAgent   │─────────────▶│  anchor, places,  │
       │  Context+Memory  │   tools      │  isoline          │
       └────────┬─────────┘   write      └───────────────────┘
                │
       ┌────────▼─────────┐
       │   HereClient     │  typed · retried · TTL-cached
       └────────┬─────────┘
                │
    geocode · discover · browse · routes · isolines
```

## Quickstart

You need a [HERE API key](https://platform.here.com/) (the free tier is generous) and a
key from any one model provider.

```bash
git clone https://github.com/ajay1808/GeoExplorer-Tech16-Project.git
cd GeoExplorer-Tech16-Project
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env      # then fill in HERE_API_KEY
streamlit run app.py
```

That is the whole setup. Paste a model key into the sidebar and the app works out who
issued it — you are never asked to pick a vendor before pasting. Keys entered this way
live for the session only and are never written to disk.

### Model providers

| Provider | Get a key | Key looks like | Models offered |
| --- | --- | --- | --- |
| OpenAI | [platform.openai.com](https://platform.openai.com/api-keys) | `sk-…` | `gpt-5.4-mini`, `gpt-5.4`, `gpt-5.1`, `gpt-4.1-mini` |
| Anthropic (Claude) | [console.anthropic.com](https://console.anthropic.com/settings/keys) | `sk-ant-…` | `claude-sonnet-5`, `claude-opus-5`, `claude-haiku-4-5` |
| Google (Gemini) | [aistudio.google.com](https://aistudio.google.com/apikey) | `AIza…` | `gemini-3.7-flash`, `gemini-3.6-flash`, `gemini-2.5-flash` |

To skip the sidebar, put a key in `.env` (or Streamlit secrets) under any of
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY` / `CLAUDE_API_KEY`, or `GOOGLE_API_KEY` /
`GEMINI_API_KEY`. Whichever is present gets used; set several and a provider picker
appears, with `GEOEXPLORER_MODEL` deciding the default.

Only function calling is required of a provider, so the system prompt, the tools and
the agent loop are identical across all three — `providers.py` is the only module that
knows any vendor-specific detail.

## What the agent can do

| Tool | What it answers |
| --- | --- |
| `find_nearby` | Free-text search: "vegan ramen", "climbing gym", a named business |
| `find_by_category` | Exhaustive search over 16 standard categories, via HERE's category index |
| `profile_neighbourhood` | An amenity census in one call — for "what's this area like?" |
| `travel_time_to` | Real distance and duration along the network, by car, foot or bike |
| `show_reachable_area` | Draws a travel-time isoline: everything within a 15-minute walk |
| `resolve_address` | Coordinates and canonical form of some *other* address |
| `describe_this_address` | What HERE resolved the anchor address to, and how precisely |

## What changed in v2

v1 was a 137-line single-file prototype. It did not run from a clean checkout, and
several of its advertised features were not connected to anything. The rewrite keeps
the idea and replaces the implementation.

**It now installs.** `requirements.txt` listed `llama_index-llms-openai` — an
underscore where the package name has a hyphen, so it does not exist on PyPI and
`pip install -r requirements.txt` failed outright. `llama-index-core` and `requests`
were missing despite being imported; `langchain-openai` was listed but never used.

**Memory works.** v1 rebuilt the `ReActAgent` inside a top-level `if` block. Streamlit
re-runs the script on every interaction, so a fresh agent with empty memory replaced
the old one before each message — the advertised memory never survived a turn. What
stood in for it was a string of past *user* messages concatenated into the prompt,
which omitted the assistant's replies and every tool result. v2 holds one session in
`st.session_state`, keyed by configuration, with a real `Context` and a token-bounded
`Memory`.

**The map exists.** v1's Folium map was commented out and would have crashed if
enabled — the helper referenced an undefined `map_object` and was never registered as
a tool. v2 renders every discovered place and isoline with pydeck, which ships inside
Streamlit.

**Addresses with punctuation work.** v1 interpolated the raw address into an f-string
URL, so `Marks & Spencer` truncated at the ampersand. All parameters are now encoded.

**Failures are legible.** v1 had no timeouts (a hung connection froze the app), no
retries, and returned `{"error": ...}` dicts into the model's context — which the model
would summarise as though it were data. v2 raises typed exceptions, retries transient
failures with backoff, and turns provider errors into advice a user can act on.

**Tool results are small.** v1 handed the model entire HERE payloads. v2 normalises
them through pydantic models and sends a handful of named fields per place.

**The agent API is current.** `ReActAgent.from_tools` is superseded by the workflow
agents; v2 uses `FunctionAgent`, which supports native tool calling, event streaming
and durable context.

**One vendor is no longer assumed.** v1 hardcoded `gpt-4o-mini` and set
`os.environ["OPENAI_API_KEY"]` as a global side effect. v2 passes credentials
explicitly and treats the provider as a runtime choice across OpenAI, Anthropic and
Google.

## Development

```bash
pytest              # 120 tests, 98% coverage, no network and no API keys required
ruff check .
mypy                # strict
```

Every HERE call in the suite is served by an in-process `httpx.MockTransport`.

### Layout

| Path | Role |
| --- | --- |
| `app.py` | Streamlit widgets, session lifecycle, rendering |
| `src/geoexplorer/here.py` | HERE client: encoding, timeouts, retries, caching |
| `src/geoexplorer/models.py` | Pydantic models and the compact tool payloads |
| `src/geoexplorer/tools.py` | Tool definitions and their model-facing docstrings |
| `src/geoexplorer/agent.py` | Agent construction, streaming bridge, error humanising |
| `src/geoexplorer/providers.py` | Provider registry, key detection, LLM construction |
| `src/geoexplorer/state.py` | Thread-safe per-conversation state |
| `src/geoexplorer/ui/map.py` | pydeck rendering |

## Deployment

**Streamlit Community Cloud** — point it at `app.py`. It reads `requirements.txt`, and
injects secrets only through `st.secrets`, so add `HERE_API_KEY` — and optionally a
model key — under *Settings → Secrets*. `load_streamlit_secrets_into_env()` bridges those into the environment, and
also accepts the old `HERE_API` name so existing deployments keep working.

**Docker**

```bash
docker build -t geoexplorer .
docker run -p 8501:8501 -e HERE_API_KEY=... -e ANTHROPIC_API_KEY=... geoexplorer
```

## Known limits

* **Category coverage is partial.** The 16 categories in `CATEGORY_IDS` are the ones
  verified against HERE's published category system. HERE's `800-*` facility range
  (hospitals, schools, parking, petrol) is not included rather than guessed at —
  `find_nearby` reaches those by free text.
* **Public transit routing is not supported.** HERE serves it from a separate API;
  `travel_time_to` covers car, pedestrian and bicycle.
* **One address per conversation.** Changing the address starts a new session, by
  design — mixing anchors mid-conversation makes answers hard to attribute.
* **HERE coverage varies by country.** An empty result means HERE has no record there,
  not that nothing exists; the agent is instructed to say so.
* **Gemini validates its key at construction.** Unlike the other two, the Google client
  calls the API when it is built, so a bad Gemini key fails when the session starts
  rather than on the first message. The error is reported either way.
