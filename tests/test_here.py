"""Client behaviour: encoding, retries, error typing, caching."""

from __future__ import annotations

import httpx
import pytest

from conftest import EMPIRE_STATE, ROUTE_PAYLOAD, geocode_payload, places_payload
from geoexplorer.here import (
    AddressNotFoundError,
    HereAPIError,
    HereClient,
    _validate_mode,
)


def test_geocode_returns_typed_result(client_factory):
    client = client_factory({"/v1/geocode": geocode_payload()})
    result = client.geocode("350 5th Ave")

    assert result.coordinate.lat == pytest.approx(EMPIRE_STATE.lat)
    assert result.address.city == "New York"
    assert result.is_precise


def test_geocode_url_encodes_special_characters():
    """The prototype built the URL with an f-string, so `&` truncated the query."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=geocode_payload())

    client = HereClient(api_key="k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    client.geocode("Marks & Spencer, High St #3")

    query = seen[0].url.params["q"]
    assert query == "Marks & Spencer, High St #3"
    assert "%26" in str(seen[0].url)  # the ampersand is escaped on the wire


def test_geocode_raises_when_no_match(client_factory):
    client = client_factory({"/v1/geocode": {"items": []}})
    with pytest.raises(AddressNotFoundError):
        client.geocode("nowhere at all")


def test_empty_address_rejected_before_any_request(client_factory):
    client = client_factory({})
    with pytest.raises(ValueError):
        client.geocode("   ")


def test_auth_failure_is_not_retried_and_explains_itself():
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(401, json={"error_description": "apiKey invalid"})

    client = HereClient(
        api_key="bad", client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=3
    )
    with pytest.raises(HereAPIError) as excinfo:
        client.geocode("anywhere")

    assert attempts == 1, "a rejected key will never succeed on retry"
    assert excinfo.value.status_code == 401
    assert "apiKey invalid" in str(excinfo.value)


def test_transient_failure_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr("geoexplorer.here.time.sleep", lambda _: None)
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(503, json={"title": "temporarily unavailable"})
        return httpx.Response(200, json=geocode_payload())

    client = HereClient(
        api_key="k", client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=2
    )
    assert client.geocode("350 5th Ave").address.city == "New York"
    assert attempts == 3


def test_repeated_identical_calls_hit_the_cache_not_the_network():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=geocode_payload())

    client = HereClient(api_key="k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    for _ in range(4):
        client.geocode("350 5th Ave")

    assert calls == 1
    assert client.cache_stats["hits"] == 3


def test_discover_uses_a_circle_filter_when_given_a_radius():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=places_payload())

    client = HereClient(api_key="k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    client.discover(EMPIRE_STATE, "coffee", radius_m=800)

    params = seen[0].url.params
    assert params["in"] == f"circle:{EMPIRE_STATE.as_here_param()};r=800"
    # HERE rejects `at` and `in=circle` together.
    assert "at" not in params


def test_discover_falls_back_to_at_without_a_radius():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=places_payload())

    client = HereClient(api_key="k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    client.discover(EMPIRE_STATE, "coffee")

    assert "in" not in seen[0].url.params
    assert seen[0].url.params["at"] == EMPIRE_STATE.as_here_param()


def test_browse_maps_a_friendly_category_to_a_here_id():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=places_payload())

    client = HereClient(api_key="k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    client.browse(EMPIRE_STATE, category="park")

    assert seen[0].url.params["categories"] == "550-5510-0202"


def test_route_summary_converts_units(client_factory):
    client = client_factory({"/v8/routes": ROUTE_PAYLOAD})
    summary = client.route_summary(EMPIRE_STATE, EMPIRE_STATE, mode="pedestrian")

    assert summary.as_tool_payload() == {
        "mode": "pedestrian",
        "distance_km": 2.4,
        "duration_minutes": 13,
    }


def test_route_with_no_result_raises(client_factory):
    client = client_factory({"/v8/routes": {"routes": []}})
    with pytest.raises(HereAPIError):
        client.route_summary(EMPIRE_STATE, EMPIRE_STATE)


@pytest.mark.parametrize(
    ("given", "expected"),
    [("walk", "pedestrian"), ("Driving", "car"), ("bike", "bicycle"), ("car", "car")],
)
def test_transport_mode_aliases(given, expected):
    assert _validate_mode(given) == expected


def test_unknown_transport_mode_rejected():
    with pytest.raises(ValueError, match="Unsupported transport mode"):
        _validate_mode("teleport")


def test_client_requires_an_api_key():
    with pytest.raises(ValueError):
        HereClient(api_key="")


# --- isolines and reverse geocoding -----------------------------------------------

# A real flexible-polyline encoding of a small closed ring near the Empire State Building.
ISOLINE_POLYLINE = "BFgz24HnmyjOwMwMwM_Y_YwM"


def test_isoline_decodes_the_flexible_polyline(client_factory):
    client = client_factory(
        {"/v8/isolines": {"isolines": [{"polygons": [{"outer": ISOLINE_POLYLINE}]}]}}
    )
    isoline = client.isoline(EMPIRE_STATE, minutes=15, mode="walk")

    assert isoline.mode == "pedestrian"
    assert isoline.minutes == 15
    assert len(isoline.boundary) == 4
    assert isoline.boundary[0].lat == pytest.approx(40.748, abs=1e-3)


def test_isoline_sends_time_range_in_seconds():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, json={"isolines": [{"polygons": [{"outer": ISOLINE_POLYLINE}]}]}
        )

    client = HereClient(api_key="k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    client.isoline(EMPIRE_STATE, minutes=20, mode="bicycle")

    params = seen[0].url.params
    assert params["range[values]"] == "1200"
    assert params["range[type]"] == "time"
    assert params["transportMode"] == "bicycle"


def test_isoline_minutes_are_clamped_to_a_sane_range():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, json={"isolines": [{"polygons": [{"outer": ISOLINE_POLYLINE}]}]}
        )

    client = HereClient(api_key="k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    client.isoline(EMPIRE_STATE, minutes=9999)

    assert seen[0].url.params["range[values]"] == "7200"  # capped at 120 minutes


def test_isoline_with_no_polygon_raises(client_factory):
    client = client_factory({"/v8/isolines": {"isolines": []}})
    with pytest.raises(HereAPIError, match="no 15-minute"):
        client.isoline(EMPIRE_STATE, minutes=15)


def test_reverse_geocode_returns_the_nearest_address(client_factory):
    client = client_factory({"/v1/revgeocode": geocode_payload()})
    result = client.reverse_geocode(EMPIRE_STATE)

    assert result.address.postal_code == "10118"


def test_reverse_geocode_with_no_match_raises(client_factory):
    client = client_factory({"/v1/revgeocode": {"items": []}})
    with pytest.raises(AddressNotFoundError):
        client.reverse_geocode(EMPIRE_STATE)


def test_discover_radius_is_clamped_to_the_api_maximum():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=places_payload(1))

    client = HereClient(api_key="k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    client.discover(EMPIRE_STATE, "coffee", radius_m=10_000_000)

    assert seen[0].url.params["in"].endswith(";r=100000")


def test_connection_errors_are_retried_then_reported(monkeypatch):
    monkeypatch.setattr("geoexplorer.here.time.sleep", lambda _: None)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network unreachable", request=request)

    client = HereClient(
        api_key="k", client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=1
    )
    with pytest.raises(HereAPIError, match="failed after retries"):
        client.geocode("anywhere")
