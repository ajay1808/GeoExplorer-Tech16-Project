"""The deck specification handed to the browser.

The prototype's map was dead code — commented out, referencing an undefined variable,
never registered as a tool. These assertions keep the replacement honest.
"""

from __future__ import annotations

import json

from conftest import EMPIRE_STATE, places_payload
from geoexplorer.models import Coordinate, Isoline, Place
from geoexplorer.state import Anchor, MapSnapshot
from geoexplorer.ui.map import MAX_ZOOM, MIN_ZOOM, build_deck


def snapshot(places=(), isoline=None) -> MapSnapshot:
    return MapSnapshot(
        anchor=Anchor(label="350 5th Ave", coordinate=EMPIRE_STATE, is_precise=True),
        places=list(places),
        isoline=isoline,
    )


def spec(snap: MapSnapshot) -> dict:
    return json.loads(build_deck(snap).to_json())


def test_anchor_alone_still_renders_one_layer():
    layers = spec(snapshot())["layers"]
    assert len(layers) == 1
    assert layers[0]["@@type"] == "ScatterplotLayer"


def test_places_and_isoline_each_add_a_layer():
    places = [Place.from_here(i) for i in places_payload(3)["items"]]
    isoline = Isoline(
        center=EMPIRE_STATE,
        mode="pedestrian",
        minutes=15,
        boundary=[EMPIRE_STATE, Coordinate(lat=40.75, lng=-73.98)],
    )
    layers = spec(snapshot(places, isoline))["layers"]

    # Polygon first so it paints beneath the markers; anchor last so it sits on top.
    assert [layer["@@type"] for layer in layers] == [
        "PolygonLayer",
        "ScatterplotLayer",
        "ScatterplotLayer",
    ]


def test_the_anchor_is_drawn_last_and_in_a_distinct_colour():
    places = [Place.from_here(i) for i in places_payload(1)["items"]]
    layers = spec(snapshot(places))["layers"]

    anchor_layer, place_layer = layers[-1], layers[-2]
    assert anchor_layer["getFillColor"] != place_layer["getFillColor"]
    assert anchor_layer["data"][0]["name"] == "350 5th Ave"


def test_tooltips_report_distance_from_the_anchor():
    places = [Place.from_here(i) for i in places_payload(1)["items"]]
    place_layer = spec(snapshot(places))["layers"][0]

    assert "m from the address" in place_layer["data"][0]["detail"]


def test_a_basemap_is_requested():
    """`map_style=None` renders on a blank void — easy to ship by accident."""
    rendered = spec(snapshot())
    assert rendered["mapStyle"], "the deck must name a basemap style"
    assert rendered["mapProvider"] == "carto", "carto needs no API token"


def test_zoom_stays_within_sane_bounds_for_a_lone_anchor():
    view = spec(snapshot())["initialViewState"]
    assert MIN_ZOOM <= view["zoom"] <= MAX_ZOOM


def test_distant_results_zoom_further_out_than_close_ones():
    def place_at(offset: float) -> Place:
        return Place.from_here(
            {
                "id": f"p{offset}",
                "title": "P",
                "position": {"lat": EMPIRE_STATE.lat + offset, "lng": EMPIRE_STATE.lng},
            }
        )

    close = spec(snapshot([place_at(0.001)]))["initialViewState"]["zoom"]
    far = spec(snapshot([place_at(0.05)]))["initialViewState"]["zoom"]

    assert far < close


def test_view_is_centred_on_the_anchor():
    view = spec(snapshot())["initialViewState"]
    assert view["latitude"] == EMPIRE_STATE.lat
    assert view["longitude"] == EMPIRE_STATE.lng
