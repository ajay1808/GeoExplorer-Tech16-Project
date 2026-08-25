"""Normalisation of HERE payloads, and the compaction that saves the agent tokens."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from conftest import EMPIRE_STATE, geocode_payload, places_payload
from geoexplorer.models import Coordinate, GeocodeResult, Place


def test_coordinate_rejects_out_of_range_values():
    with pytest.raises(ValidationError):
        Coordinate(lat=91.0, lng=0.0)
    with pytest.raises(ValidationError):
        Coordinate(lat=0.0, lng=181.0)


def test_haversine_distance_is_plausible():
    # Empire State Building to Times Square is roughly 1 km.
    times_square = Coordinate(lat=40.758, lng=-73.9855)
    assert 900 < EMPIRE_STATE.distance_to(times_square) < 1200


def test_place_flattens_contacts_and_hours():
    place = Place.from_here(places_payload(1)["items"][0])

    assert place.phone == "+12125550000"
    assert place.website == "https://example.test/0"
    assert place.opening_hours == "Mon-Fri: 07:00 - 19:00"


def test_place_survives_a_payload_with_no_optional_fields():
    place = Place.from_here(
        {"id": "x", "title": "Bare", "position": {"lat": 1.0, "lng": 2.0}}
    )

    assert place.categories == []
    assert place.phone is None
    assert place.as_tool_payload() == {"name": "Bare", "address": "", "lat": 1.0, "lng": 2.0}


def test_tool_payload_drops_nulls_and_is_far_smaller_than_the_raw_item():
    raw = places_payload(1)["items"][0]
    payload = Place.from_here(raw).as_tool_payload()

    assert None not in payload.values()
    assert len(json.dumps(payload)) < len(json.dumps(raw))


def test_precision_flag_distinguishes_a_building_from_a_region():
    precise = GeocodeResult.from_here("q", geocode_payload()["items"][0])
    assert precise.is_precise

    vague_item = geocode_payload()["items"][0] | {"resultType": "locality"}
    assert not GeocodeResult.from_here("q", vague_item).is_precise
