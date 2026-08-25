"""Shared fixtures. Every HERE call in the suite is served by an in-process mock
transport, so the tests need no API key and no network."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from geoexplorer.here import HereClient
from geoexplorer.models import Coordinate
from geoexplorer.state import Anchor, SessionState

EMPIRE_STATE = Coordinate(lat=40.748444, lng=-73.985664)


def geocode_payload(label: str = "350 5th Ave, New York, NY 10118, United States") -> dict:
    return {
        "items": [
            {
                "title": label,
                "resultType": "houseNumber",
                "position": {"lat": EMPIRE_STATE.lat, "lng": EMPIRE_STATE.lng},
                "address": {
                    "label": label,
                    "street": "5th Ave",
                    "city": "New York",
                    "stateCode": "NY",
                    "postalCode": "10118",
                    "countryCode": "USA",
                },
                "scoring": {"queryScore": 0.99},
            }
        ]
    }


def places_payload(count: int = 3) -> dict:
    return {
        "items": [
            {
                "id": f"here:pds:place:{i}",
                "title": f"Test Cafe {i}",
                "position": {"lat": EMPIRE_STATE.lat + i * 0.001, "lng": EMPIRE_STATE.lng},
                "distance": 100 * (i + 1),
                "address": {"label": f"{i} Test Street, New York"},
                "categories": [{"name": "Coffee Shop"}],
                "contacts": [
                    {
                        "phone": [{"value": f"+1212555000{i}"}],
                        "www": [{"value": f"https://example.test/{i}"}],
                    }
                ],
                "openingHours": [{"text": ["Mon-Fri: 07:00 - 19:00"]}],
            }
            for i in range(count)
        ]
    }


ROUTE_PAYLOAD = {
    "routes": [{"sections": [{"summary": {"length": 2400, "duration": 780}}]}]
}


@pytest.fixture
def mock_transport() -> Callable[[dict[str, Any]], httpx.MockTransport]:
    """Build a transport that maps URL path fragments to canned JSON responses."""

    def factory(routes: dict[str, Any], status: int = 200) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            for fragment, payload in routes.items():
                if fragment in str(request.url):
                    if isinstance(payload, httpx.Response):
                        return payload
                    return httpx.Response(status, json=payload, request=request)
            return httpx.Response(
                404, json={"title": f"unmocked route {request.url}"}, request=request
            )

        return httpx.MockTransport(handler)

    return factory


@pytest.fixture
def client_factory(mock_transport):  # type: ignore[no-untyped-def]
    """A HereClient wired to canned responses."""

    def factory(routes: dict[str, Any], **kwargs: Any) -> HereClient:
        http = httpx.Client(transport=mock_transport(routes))
        return HereClient(api_key="test-key", client=http, **kwargs)

    return factory


@pytest.fixture
def session_state() -> SessionState:
    return SessionState(
        Anchor(label="350 5th Ave, New York", coordinate=EMPIRE_STATE, is_precise=True)
    )


__all__ = ["EMPIRE_STATE", "ROUTE_PAYLOAD", "geocode_payload", "json", "places_payload"]
