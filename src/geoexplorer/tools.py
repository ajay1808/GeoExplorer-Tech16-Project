"""The tools the agent can call.

Three things distinguish these from the prototype's two raw API wrappers:

1. **The anchor address is resolved once, up front.** The prototype pasted the address
   into every prompt and made the model re-geocode it on each turn — a wasted tool call
   per question, and an invitation for the model to drift onto a different location.
   Here every tool reads the already-resolved coordinate from session state.
2. **Return values are compact and typed.** The model sees a handful of named fields
   per place instead of a deeply nested HERE payload costing thousands of tokens.
3. **Tools record what they found.** Places and travel-time polygons are written into
   `SessionState`, which is what keeps the map beside the chat in sync with the
   conversation without a second round of API calls.

Docstrings here are load-bearing: LlamaIndex sends them to the model as the tool
descriptions, so they are written for that reader.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from llama_index.core.tools import BaseTool, FunctionTool

from .here import CATEGORY_IDS, HereAPIError, HereClient
from .models import Place
from .state import SessionState

logger = logging.getLogger(__name__)

# Categories worth counting when profiling a neighbourhood, and the label to report.
PROFILE_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("restaurant", "restaurants"),
    ("coffee", "coffee shops"),
    ("bar", "bars and pubs"),
    ("grocery", "grocery stores"),
    ("pharmacy", "pharmacies"),
    ("park", "parks"),
    ("public_transit", "transit stops"),
    ("bank", "banks"),
)

PROFILE_LIMIT = 30


ToolList = list[BaseTool | Callable[..., Any]]


def build_tools(client: HereClient, state: SessionState) -> ToolList:
    """Bind the HERE client and session state into a fresh set of tools."""

    async def find_nearby(
        query: str, radius_meters: int = 1500, limit: int = 6
    ) -> dict[str, Any]:
        """Search for places near the conversation's address by free-text description.

        Use this for anything specific or unusual: "vegan ramen", "climbing gym",
        "24 hour laundromat", a named business. For broad, common categories prefer
        find_by_category, which is more exhaustive.

        Args:
            query: What to look for, e.g. "bookstore" or "Thai food".
            radius_meters: How far to search. Defaults to 1500 (about a 20 minute walk).
            limit: Maximum number of results, 1-20.
        """
        center = state.anchor.coordinate
        try:
            places = await asyncio.to_thread(
                client.discover, center, query, radius_m=radius_meters, limit=min(limit, 20)
            )
        except HereAPIError as exc:
            return {"error": str(exc), "results": []}

        state.record_places(places)
        if not places:
            return {
                "results": [],
                "note": f"No matches for {query!r} within {radius_meters} m. "
                "Try a wider radius or a broader term before concluding none exist.",
            }
        return {
            "query": query,
            "count": len(places),
            "results": [p.as_tool_payload() for p in places],
        }

    async def find_by_category(
        category: str, radius_meters: int = 1500, limit: int = 8
    ) -> dict[str, Any]:
        """List every place of a standard category near the conversation's address.

        More reliable than free-text search for common amenities, because it queries
        HERE's structured category index rather than matching on names.

        Args:
            category: One of: restaurant, coffee, bar, cinema, museum, train_station,
                bus_stop, public_transit, hotel, park, convenience_store,
                shopping_mall, grocery, pharmacy, bank, atm.
            radius_meters: How far to search. Defaults to 1500.
            limit: Maximum number of results, 1-30.
        """
        if category not in CATEGORY_IDS:
            return {
                "error": f"{category!r} is not a supported category.",
                "supported_categories": sorted(CATEGORY_IDS),
                "hint": "For anything outside this list, use find_nearby instead.",
            }
        center = state.anchor.coordinate
        try:
            places = await asyncio.to_thread(
                client.browse,
                center,
                category=category,
                radius_m=radius_meters,
                limit=min(limit, PROFILE_LIMIT),
            )
        except HereAPIError as exc:
            return {"error": str(exc), "results": []}

        state.record_places(places)
        return {
            "category": category,
            "count": len(places),
            "results": [p.as_tool_payload() for p in places],
        }

    async def profile_neighbourhood(radius_meters: int = 1000) -> dict[str, Any]:
        """Count the common amenities around the address to characterise the neighbourhood.

        Use this for open-ended questions like "what is this area like?", "is it
        walkable?" or "what's the neighbourhood like for families?" — it returns an
        amenity census in one step instead of many separate searches.

        Args:
            radius_meters: Radius of the profile. Defaults to 1000 (about a 12 minute walk).
        """
        center = state.anchor.coordinate

        async def count(category: str) -> list[Place] | None:
            try:
                return await asyncio.to_thread(
                    client.browse,
                    center,
                    category=category,
                    radius_m=radius_meters,
                    limit=PROFILE_LIMIT,
                )
            except HereAPIError as exc:
                logger.warning("neighbourhood profile: %s failed: %s", category, exc)
                return None

        found = await asyncio.gather(*(count(cat) for cat, _ in PROFILE_CATEGORIES))

        census: dict[str, int] = {}
        unavailable: list[str] = []
        for (_, label), places in zip(PROFILE_CATEGORIES, found, strict=True):
            if places is None:
                unavailable.append(label)
                continue
            census[label] = len(places)
            state.record_places(places)

        payload: dict[str, Any] = {
            "radius_meters": radius_meters,
            "amenity_counts": census,
            "note": f"Counts are capped at {PROFILE_LIMIT} per category; a value of "
            f"{PROFILE_LIMIT} means 'at least {PROFILE_LIMIT}', not exactly that many.",
        }
        if unavailable:
            payload["categories_that_failed_to_load"] = unavailable
        return payload

    async def travel_time_to(destination: str, mode: str = "car") -> dict[str, Any]:
        """Find real travel distance and time from the address to somewhere else.

        This follows the road or footpath network, so it is the right tool for "how long
        does it take to get to X". Straight-line distance from a search result is not a
        substitute — in a city the two differ substantially.

        Args:
            destination: An address or place name to travel to.
            mode: One of car, pedestrian, bicycle. Defaults to car.
        """
        origin = state.anchor.coordinate
        try:
            target = await asyncio.to_thread(client.geocode, destination)
            summary = await asyncio.to_thread(
                client.route_summary, origin, target.coordinate, mode=mode
            )
        except (HereAPIError, ValueError) as exc:
            return {"error": str(exc)}

        state.record_places(
            [
                Place(
                    id=f"destination:{target.address.label}",
                    title=destination,
                    coordinate=target.coordinate,
                    address=target.address,
                    categories=["Destination"],
                )
            ]
        )
        return {"destination": target.address.label, **summary.as_tool_payload()}

    async def show_reachable_area(minutes: int = 15, mode: str = "pedestrian") -> dict[str, Any]:
        """Draw the area reachable from the address within a time budget, on the map.

        Use this for questions about what is "within a 15 minute walk" or "a 10 minute
        drive". It shades a real travel-time polygon on the map the user is looking at,
        which is a far better answer than a radius in metres.

        Args:
            minutes: The time budget, 1-120. Defaults to 15.
            mode: One of pedestrian, car, bicycle. Defaults to pedestrian.
        """
        center = state.anchor.coordinate
        try:
            isoline = await asyncio.to_thread(client.isoline, center, minutes=minutes, mode=mode)
        except (HereAPIError, ValueError) as exc:
            return {"error": str(exc)}

        state.record_isoline(isoline)
        return isoline.as_tool_payload()

    async def resolve_address(address: str) -> dict[str, Any]:
        """Look up the coordinates and canonical form of any address.

        Only needed for addresses other than the one this conversation is about — that
        one is already resolved, and every other tool uses it automatically.

        Args:
            address: The free-text address to resolve.
        """
        try:
            result = await asyncio.to_thread(client.geocode, address)
        except (HereAPIError, ValueError) as exc:
            return {"error": str(exc)}
        return result.as_tool_payload()

    async def describe_this_address() -> dict[str, Any]:
        """Report the address this conversation is anchored on, exactly as HERE resolved it.

        Use this when the user asks "where am I", "what address is this", or when a
        vague-looking match needs to be confirmed before you answer.
        """
        anchor = state.anchor
        return {
            "resolved_address": anchor.label,
            "lat": round(anchor.coordinate.lat, 6),
            "lng": round(anchor.coordinate.lng, 6),
            "is_precise_match": anchor.is_precise,
            "caveat": None
            if anchor.is_precise
            else "This matched a broad area rather than a specific building, so nearby "
            "results are relative to the area's centre point.",
        }

    return [
        FunctionTool.from_defaults(async_fn=find_nearby),
        FunctionTool.from_defaults(async_fn=find_by_category),
        FunctionTool.from_defaults(async_fn=profile_neighbourhood),
        FunctionTool.from_defaults(async_fn=travel_time_to),
        FunctionTool.from_defaults(async_fn=show_reachable_area),
        FunctionTool.from_defaults(async_fn=resolve_address),
        FunctionTool.from_defaults(async_fn=describe_this_address),
    ]
