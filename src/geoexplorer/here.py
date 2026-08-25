"""A typed client for the HERE Location Services APIs.

Improvements over the prototype's two inline `requests.get` calls:

* Query values go through httpx's `params=` encoding. The prototype interpolated the
  raw address into an f-string URL, so any address containing `&`, `#` or `+`
  silently produced a wrong query.
* Every request has a timeout. Without one a hung connection freezes the whole
  Streamlit script run with no way out.
* Transient failures (429, 5xx, connection resets) are retried with exponential
  backoff instead of surfacing as a one-shot error string.
* Failures raise a typed exception carrying the HERE error message, rather than
  returning `{"error": ...}` dicts that the LLM would cheerfully summarise as fact.
* Responses are TTL-cached, so repeated questions about one address cost one call.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

import httpx

from .cache import TTLCache, make_key
from .models import Coordinate, GeocodeResult, Isoline, Place, RouteSummary

logger = logging.getLogger(__name__)

GEOCODE_URL = "https://geocode.search.hereapi.com/v1/geocode"
REVGEOCODE_URL = "https://revgeocode.search.hereapi.com/v1/revgeocode"
DISCOVER_URL = "https://discover.search.hereapi.com/v1/discover"
BROWSE_URL = "https://browse.search.hereapi.com/v1/browse"
ROUTES_URL = "https://router.hereapi.com/v8/routes"
ISOLINE_URL = "https://isoline.router.hereapi.com/v8/isolines"

TransportMode = Literal["car", "pedestrian", "bicycle"]
VALID_MODES: frozenset[str] = frozenset({"car", "pedestrian", "bicycle"})

# Category IDs verified against the HERE Places category system documentation.
# Anything not listed here is still reachable through free-text `discover`.
CATEGORY_IDS: dict[str, str] = {
    "restaurant": "100-1000-0000",
    "coffee": "100-1100-0010",
    "bar": "200-2000-0011",
    "cinema": "200-2100-0019",
    "museum": "300-3100-0000",
    "train_station": "400-4100-0035",
    "bus_stop": "400-4100-0042",
    "public_transit": "400-4100-0043",
    "hotel": "500-5000-0053",
    "park": "550-5510-0202",
    "convenience_store": "600-6000-0061",
    "shopping_mall": "600-6100-0062",
    "grocery": "600-6300-0066",
    "pharmacy": "600-6400-0070",
    "bank": "700-7000-0107",
    "atm": "700-7010-0108",
}

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class HereAPIError(RuntimeError):
    """A HERE request failed in a way the caller cannot paper over."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AddressNotFoundError(HereAPIError):
    """HERE returned a valid, empty result set for an address."""


class HereClient:
    """Synchronous HERE API client. One instance per session; safe to share across threads."""

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = 10.0,
        max_retries: int = 2,
        cache_ttl: float = 900.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("HereClient requires a non-empty API key.")
        self._api_key = api_key
        self._max_retries = max_retries
        self._cache = TTLCache(ttl_seconds=cache_ttl)
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(timeout),
            headers={"User-Agent": "GeoExplorer/2.0"},
            follow_redirects=True,
        )

    # ---------------------------------------------------------------- internals

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        """Issue a cached, retried GET and return the decoded JSON body."""
        cache_key = make_key(url, params)
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("HERE cache hit: %s", cache_key)
            return dict(cached)

        request_params = {**params, "apiKey": self._api_key}
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.get(url, params=request_params)
            except httpx.RequestError as exc:
                last_error = exc
            else:
                if response.status_code == 200:
                    body: dict[str, Any] = response.json()
                    self._cache.set(cache_key, body)
                    return body
                if response.status_code not in RETRYABLE_STATUS:
                    raise HereAPIError(
                        _describe_error(response), status_code=response.status_code
                    )
                last_error = HereAPIError(
                    _describe_error(response), status_code=response.status_code
                )

            if attempt < self._max_retries:
                backoff = 0.5 * (2**attempt)
                logger.warning("HERE request failed (%s), retrying in %.1fs", last_error, backoff)
                time.sleep(backoff)

        raise HereAPIError(f"HERE request to {url} failed after retries: {last_error}")

    # ------------------------------------------------------------------- search

    def geocode(self, address: str, *, lang: str = "en") -> GeocodeResult:
        """Resolve a free-text address to coordinates."""
        if not address.strip():
            raise ValueError("Address must not be empty.")
        body = self._get(GEOCODE_URL, {"q": address, "limit": 1, "lang": lang})
        items = body.get("items") or []
        if not items:
            raise AddressNotFoundError(f"HERE could not resolve the address {address!r}.")
        return GeocodeResult.from_here(address, items[0])

    def reverse_geocode(self, coordinate: Coordinate, *, lang: str = "en") -> GeocodeResult:
        """Find the nearest postal address to a coordinate."""
        body = self._get(
            REVGEOCODE_URL, {"at": coordinate.as_here_param(), "limit": 1, "lang": lang}
        )
        items = body.get("items") or []
        if not items:
            raise AddressNotFoundError(f"No address found near {coordinate.as_here_param()}.")
        return GeocodeResult.from_here(coordinate.as_here_param(), items[0])

    def discover(
        self,
        center: Coordinate,
        query: str,
        *,
        radius_m: int | None = None,
        limit: int = 8,
        lang: str = "en",
    ) -> list[Place]:
        """Free-text POI search around a point."""
        params: dict[str, Any] = {"q": query, "limit": _clamp(limit, 1, 50), "lang": lang}
        # HERE rejects `at` and `in=circle` together — the circle already implies a centre.
        if radius_m:
            params["in"] = f"circle:{center.as_here_param()};r={_clamp(radius_m, 1, 100_000)}"
        else:
            params["at"] = center.as_here_param()
        body = self._get(DISCOVER_URL, params)
        return [Place.from_here(item) for item in body.get("items", []) if "position" in item]

    def browse(
        self,
        center: Coordinate,
        *,
        category: str,
        radius_m: int | None = None,
        limit: int = 8,
        name: str | None = None,
        lang: str = "en",
    ) -> list[Place]:
        """Structured POI search by HERE category, which is exhaustive where discover is fuzzy."""
        category_id = CATEGORY_IDS.get(category, category)
        params: dict[str, Any] = {
            "at": center.as_here_param(),
            "categories": category_id,
            "limit": _clamp(limit, 1, 100),
            "lang": lang,
        }
        if radius_m:
            params["in"] = f"circle:{center.as_here_param()};r={_clamp(radius_m, 1, 100_000)}"
        if name:
            params["name"] = name
        body = self._get(BROWSE_URL, params)
        return [Place.from_here(item) for item in body.get("items", []) if "position" in item]

    # ------------------------------------------------------------------ routing

    def route_summary(
        self, origin: Coordinate, destination: Coordinate, *, mode: str = "car"
    ) -> RouteSummary:
        """Distance and duration along the real network, not straight-line distance."""
        mode = _validate_mode(mode)
        body = self._get(
            ROUTES_URL,
            {
                "transportMode": mode,
                "origin": origin.as_here_param(),
                "destination": destination.as_here_param(),
                "return": "summary",
            },
        )
        routes = body.get("routes") or []
        if not routes or not routes[0].get("sections"):
            raise HereAPIError(f"No {mode} route exists between those two points.")
        summary = routes[0]["sections"][0]["summary"]
        return RouteSummary(
            origin=origin,
            destination=destination,
            mode=mode,
            distance_m=int(summary["length"]),
            duration_s=int(summary["duration"]),
        )

    def isoline(self, center: Coordinate, *, minutes: int, mode: str = "pedestrian") -> Isoline:
        """The polygon of everything reachable from `center` within `minutes`."""
        mode = _validate_mode(mode)
        minutes = _clamp(minutes, 1, 120)
        body = self._get(
            ISOLINE_URL,
            {
                "transportMode": mode,
                "origin": center.as_here_param(),
                "range[type]": "time",
                "range[values]": minutes * 60,
            },
        )
        isolines = body.get("isolines") or []
        if not isolines or not isolines[0].get("polygons"):
            raise HereAPIError(f"HERE returned no {minutes}-minute {mode} isoline for that point.")
        outer = isolines[0]["polygons"][0]["outer"]
        return Isoline(
            center=center, mode=mode, minutes=minutes, boundary=_decode_polyline(outer)
        )

    # -------------------------------------------------------------------- admin

    @property
    def cache_stats(self) -> dict[str, int]:
        return {"entries": len(self._cache), "hits": self._cache.hits, "misses": self._cache.misses}

    def close(self) -> None:
        self._client.close()


def _decode_polyline(encoded: str) -> list[Coordinate]:
    """Decode a HERE flexible polyline into coordinates."""
    import flexpolyline

    return [Coordinate(lat=lat, lng=lng) for lat, lng in flexpolyline.decode(encoded)]


def _describe_error(response: httpx.Response) -> str:
    """Pull HERE's own error text out of the body when it provides one."""
    try:
        body = response.json()
    except ValueError:
        return f"HERE returned HTTP {response.status_code}."
    detail = body.get("error_description") or body.get("cause") or body.get("title")
    if response.status_code in (401, 403):
        detail = detail or "The HERE API key was rejected. Check that it is valid and enabled."
    return f"HERE returned HTTP {response.status_code}: {detail or 'no detail provided'}"


def _validate_mode(mode: str) -> str:
    normalised = mode.strip().lower()
    aliases = {"walk": "pedestrian", "walking": "pedestrian", "foot": "pedestrian",
               "drive": "car", "driving": "car", "bike": "bicycle", "cycling": "bicycle"}
    normalised = aliases.get(normalised, normalised)
    if normalised not in VALID_MODES:
        raise ValueError(f"Unsupported transport mode {mode!r}. Use one of: {sorted(VALID_MODES)}.")
    return normalised


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))
