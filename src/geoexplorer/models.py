"""Typed representations of the slices of HERE responses the app actually uses.

The raw HERE payloads are large and deeply nested. Handing them to an LLM verbatim
(as the original prototype did) burns thousands of tokens per tool call and buries the
few fields that matter. Everything crossing the tool boundary is normalised into these
models first, and `as_tool_payload` renders the compact form the agent sees.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field

EARTH_RADIUS_M = 6_371_000.0


class Coordinate(BaseModel):
    """A WGS84 latitude/longitude pair."""

    lat: float = Field(ge=-90.0, le=90.0)
    lng: float = Field(ge=-180.0, le=180.0)

    @classmethod
    def from_here(cls, position: dict[str, Any]) -> Coordinate:
        return cls(lat=position["lat"], lng=position["lng"])

    def as_here_param(self) -> str:
        """Render as the `at=lat,lng` form HERE expects."""
        return f"{self.lat},{self.lng}"

    def distance_to(self, other: Coordinate) -> float:
        """Great-circle distance in metres. Used for sanity-checking API distances."""
        p1, p2 = math.radians(self.lat), math.radians(other.lat)
        dp = p2 - p1
        dl = math.radians(other.lng - self.lng)
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


class Address(BaseModel):
    """A postal address, flattened from HERE's `address` object."""

    label: str = ""
    street: str | None = None
    city: str | None = None
    state: str | None = None
    postal_code: str | None = None
    country: str | None = None

    @classmethod
    def from_here(cls, address: dict[str, Any] | None) -> Address:
        address = address or {}
        return cls(
            label=address.get("label", ""),
            street=address.get("street"),
            city=address.get("city"),
            state=address.get("stateCode") or address.get("state"),
            postal_code=address.get("postalCode"),
            country=address.get("countryCode") or address.get("countryName"),
        )


class Place(BaseModel):
    """A point of interest returned by /discover or /browse."""

    id: str
    title: str
    coordinate: Coordinate
    address: Address
    categories: list[str] = Field(default_factory=list)
    distance_m: int | None = None
    phone: str | None = None
    website: str | None = None
    opening_hours: str | None = None

    @classmethod
    def from_here(cls, item: dict[str, Any]) -> Place:
        contacts = (item.get("contacts") or [{}])[0]
        hours = item.get("openingHours") or []
        return cls(
            id=item.get("id", ""),
            title=item.get("title", "Unnamed place"),
            coordinate=Coordinate.from_here(item["position"]),
            address=Address.from_here(item.get("address")),
            categories=[c["name"] for c in item.get("categories", []) if "name" in c],
            distance_m=item.get("distance"),
            phone=_first_contact_value(contacts, "phone"),
            website=_first_contact_value(contacts, "www"),
            opening_hours="; ".join(hours[0].get("text", [])) if hours else None,
        )

    def as_tool_payload(self) -> dict[str, Any]:
        """The compact dict handed back to the LLM — no nulls, no nesting."""
        payload: dict[str, Any] = {
            "name": self.title,
            "address": self.address.label,
            "lat": round(self.coordinate.lat, 6),
            "lng": round(self.coordinate.lng, 6),
        }
        if self.distance_m is not None:
            payload["distance_m"] = self.distance_m
        if self.categories:
            payload["category"] = self.categories[0]
        for key, value in (
            ("phone", self.phone),
            ("website", self.website),
            ("hours", self.opening_hours),
        ):
            if value:
                payload[key] = value
        return payload


class GeocodeResult(BaseModel):
    """A resolved address: what the user typed, and where HERE decided that is."""

    query: str
    coordinate: Coordinate
    address: Address
    result_type: str | None = None
    query_score: float | None = None

    @classmethod
    def from_here(cls, query: str, item: dict[str, Any]) -> GeocodeResult:
        scoring = item.get("scoring") or {}
        return cls(
            query=query,
            coordinate=Coordinate.from_here(item["position"]),
            address=Address.from_here(item.get("address")),
            result_type=item.get("resultType"),
            query_score=scoring.get("queryScore"),
        )

    @property
    def is_precise(self) -> bool:
        """True when HERE matched a specific address rather than a whole city or region.

        A vague match is not an error, but the agent should say so rather than quietly
        answering questions about the centroid of a metro area.
        """
        return self.result_type in {"houseNumber", "street", "place", "postalCodePoint"}

    def as_tool_payload(self) -> dict[str, Any]:
        return {
            "resolved_address": self.address.label,
            "lat": round(self.coordinate.lat, 6),
            "lng": round(self.coordinate.lng, 6),
            "match_type": self.result_type,
            "is_precise_match": self.is_precise,
        }


class RouteSummary(BaseModel):
    """Travel distance and time between two points for one transport mode."""

    origin: Coordinate
    destination: Coordinate
    mode: str
    distance_m: int
    duration_s: int

    def as_tool_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "distance_km": round(self.distance_m / 1000, 2),
            "duration_minutes": round(self.duration_s / 60),
        }


class Isoline(BaseModel):
    """A polygon enclosing everything reachable within a time budget."""

    center: Coordinate
    mode: str
    minutes: int
    boundary: list[Coordinate]

    def as_tool_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "minutes": self.minutes,
            "boundary_points": len(self.boundary),
            "note": "The reachable area is drawn on the map beside this conversation.",
        }


def _first_contact_value(contacts: dict[str, Any], key: str) -> str | None:
    entries = contacts.get(key) or []
    if entries and isinstance(entries, list):
        value = entries[0].get("value")
        return str(value) if value is not None else None
    return None
