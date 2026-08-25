"""Tool behaviour — the layer the model actually talks to."""

from __future__ import annotations

import httpx
import pytest

from conftest import ROUTE_PAYLOAD, geocode_payload, places_payload
from geoexplorer.here import HereClient
from geoexplorer.state import SessionState
from geoexplorer.tools import build_tools


def tools_by_name(client: HereClient, state: SessionState) -> dict:
    return {t.metadata.name: t for t in build_tools(client, state)}


@pytest.fixture
def tools(client_factory, session_state):
    client = client_factory(
        {
            "/v1/geocode": geocode_payload(),
            "/v1/discover": places_payload(3),
            "/v1/browse": places_payload(2),
            "/v8/routes": ROUTE_PAYLOAD,
        }
    )
    return tools_by_name(client, session_state), session_state


async def test_find_nearby_returns_compact_results_and_populates_the_map(tools):
    registry, state = tools
    result = await registry["find_nearby"].acall(query="coffee", radius_meters=800)

    payload = result.raw_output
    assert payload["count"] == 3
    assert payload["results"][0]["name"] == "Test Cafe 0"
    assert len(state.snapshot().places) == 3


async def test_empty_search_tells_the_model_not_to_overconclude(client_factory, session_state):
    client = client_factory({"/v1/discover": {"items": []}})
    registry = tools_by_name(client, session_state)

    payload = (await registry["find_nearby"].acall(query="submarine dealership")).raw_output

    assert payload["results"] == []
    assert "before concluding none exist" in payload["note"]


async def test_unknown_category_is_refused_with_the_valid_list(tools):
    registry, _ = tools
    payload = (await registry["find_by_category"].acall(category="nightclub")).raw_output

    assert "error" in payload
    assert "restaurant" in payload["supported_categories"]
    assert "find_nearby" in payload["hint"]


async def test_category_search_records_places(tools):
    registry, state = tools
    payload = (await registry["find_by_category"].acall(category="park")).raw_output

    assert payload["count"] == 2
    assert len(state.snapshot().places) == 2


async def test_neighbourhood_profile_counts_every_category(tools):
    registry, _ = tools
    payload = (await registry["profile_neighbourhood"].acall(radius_meters=1000)).raw_output

    assert payload["radius_meters"] == 1000
    assert payload["amenity_counts"]["restaurants"] == 2
    assert set(payload["amenity_counts"]) == {
        "restaurants", "coffee shops", "bars and pubs", "grocery stores",
        "pharmacies", "parks", "transit stops", "banks",
    }


async def test_profile_degrades_gracefully_when_one_category_fails(session_state):
    """A partial outage should shrink the answer, not abort the turn."""
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if "600-6300-0066" in str(request.url):  # grocery
            return httpx.Response(500, json={"title": "boom"})
        return httpx.Response(200, json=places_payload(1))

    client = HereClient(
        api_key="k",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    registry = tools_by_name(client, session_state)
    payload = (await registry["profile_neighbourhood"].acall()).raw_output

    assert "grocery stores" not in payload["amenity_counts"]
    assert payload["categories_that_failed_to_load"] == ["grocery stores"]
    assert payload["amenity_counts"]["restaurants"] == 1


async def test_travel_time_geocodes_then_routes(tools):
    registry, state = tools
    payload = (await registry["travel_time_to"].acall(destination="Times Square")).raw_output

    assert payload["duration_minutes"] == 13
    assert payload["distance_km"] == 2.4
    assert any(p.title == "Times Square" for p in state.snapshot().places)


async def test_tool_errors_are_returned_not_raised(client_factory, session_state):
    """A tool that raises kills the turn; one that returns an error lets the model recover."""
    client = client_factory({"/v1/discover": httpx.Response(403, json={"title": "forbidden"})})
    registry = tools_by_name(client, session_state)

    payload = (await registry["find_nearby"].acall(query="coffee")).raw_output

    assert payload["results"] == []
    assert "403" in payload["error"]


async def test_describe_this_address_reports_the_anchor(tools):
    registry, _ = tools
    payload = (await registry["describe_this_address"].acall()).raw_output

    assert payload["resolved_address"] == "350 5th Ave, New York"
    assert payload["is_precise_match"] is True
    assert payload["caveat"] is None


async def test_every_tool_has_a_description_the_model_can_use(tools):
    registry, _ = tools
    assert len(registry) == 7
    for name, tool in registry.items():
        description = tool.metadata.description or ""
        assert len(description) > 80, f"{name} needs a usable description"


ISOLINE_RESPONSE = {"isolines": [{"polygons": [{"outer": "BFgz24HnmyjOwMwMwM_Y_YwM"}]}]}


async def test_reachable_area_records_a_polygon_for_the_map(client_factory, session_state):
    client = client_factory({"/v8/isolines": ISOLINE_RESPONSE})
    registry = tools_by_name(client, session_state)

    payload = (await registry["show_reachable_area"].acall(minutes=15, mode="walk")).raw_output

    assert payload["mode"] == "pedestrian"
    assert payload["boundary_points"] == 4
    assert session_state.snapshot().isoline is not None


async def test_reachable_area_rejects_an_impossible_mode(client_factory, session_state):
    registry = tools_by_name(client_factory({}), session_state)

    payload = (await registry["show_reachable_area"].acall(mode="teleport")).raw_output

    assert "Unsupported transport mode" in payload["error"]
    assert session_state.snapshot().isoline is None


async def test_resolve_address_returns_a_precision_flag(tools):
    registry, _ = tools
    payload = (await registry["resolve_address"].acall(address="350 5th Ave")).raw_output

    assert payload["is_precise_match"] is True
    assert payload["lat"] == pytest.approx(40.748444)


async def test_resolve_address_reports_a_failed_lookup(client_factory, session_state):
    client = client_factory({"/v1/geocode": {"items": []}})
    registry = tools_by_name(client, session_state)

    payload = (await registry["resolve_address"].acall(address="nowhere")).raw_output

    assert "could not resolve" in payload["error"]


async def test_travel_time_reports_an_unroutable_destination(client_factory, session_state):
    client = client_factory({"/v1/geocode": geocode_payload(), "/v8/routes": {"routes": []}})
    registry = tools_by_name(client, session_state)

    payload = (await registry["travel_time_to"].acall(destination="an island")).raw_output

    assert "No car route" in payload["error"]


async def test_category_search_reports_an_api_failure(client_factory, session_state):
    client = client_factory({"/v1/browse": httpx.Response(500, json={"title": "boom"})})
    registry = tools_by_name(client, session_state)

    payload = (await registry["find_by_category"].acall(category="park")).raw_output

    assert payload["results"] == []
    assert "500" in payload["error"]
