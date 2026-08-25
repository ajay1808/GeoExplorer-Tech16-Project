"""The map beside the chat.

The prototype had a Folium map, but it was commented out and would not have run: the
helper referenced an undefined `map_object`, never returned, and was never registered
as a tool. This version is wired to real agent output — every place a tool finds and
every travel-time polygon it computes appears here — and uses pydeck, which ships with
Streamlit, so it adds no dependency.
"""

from __future__ import annotations

import math
from typing import Any

import pydeck as pdk

from ..state import MapSnapshot

ANCHOR_COLOUR = [214, 69, 65]
PLACE_COLOUR = [11, 110, 79]
ISOLINE_FILL = [11, 110, 79, 45]
ISOLINE_LINE = [11, 110, 79, 180]

BASEMAP_STYLE = "light"
MIN_ZOOM, MAX_ZOOM = 10.0, 17.0
VIEW_HEIGHT_PX = 520


def build_deck(snapshot: MapSnapshot) -> pdk.Deck:
    """Render the anchor, discovered places, and any reachable-area polygon."""
    layers: list[pdk.Layer] = []

    if snapshot.isoline is not None:
        layers.append(
            pdk.Layer(
                "PolygonLayer",
                data=[{"polygon": [[c.lng, c.lat] for c in snapshot.isoline.boundary]}],
                get_polygon="polygon",
                get_fill_color=ISOLINE_FILL,
                get_line_color=ISOLINE_LINE,
                line_width_min_pixels=2,
                stroked=True,
                filled=True,
                pickable=False,
            )
        )

    if snapshot.places:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=[_place_row(p, snapshot) for p in snapshot.places],
                get_position=["lng", "lat"],
                get_fill_color=PLACE_COLOUR,
                get_radius=40,
                radius_min_pixels=6,
                radius_max_pixels=14,
                stroked=True,
                get_line_color=[255, 255, 255],
                line_width_min_pixels=1,
                pickable=True,
            )
        )

    anchor = snapshot.anchor
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=[
                {
                    "lat": anchor.coordinate.lat,
                    "lng": anchor.coordinate.lng,
                    "name": anchor.label,
                    "detail": "The address this conversation is about",
                }
            ],
            get_position=["lng", "lat"],
            get_fill_color=ANCHOR_COLOUR,
            get_radius=60,
            radius_min_pixels=9,
            radius_max_pixels=18,
            stroked=True,
            get_line_color=[255, 255, 255],
            line_width_min_pixels=2,
            pickable=True,
        )
    )

    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=anchor.coordinate.lat,
            longitude=anchor.coordinate.lng,
            zoom=_zoom_for(snapshot),
            pitch=0,
        ),
        # Carto's Positron tiles: a free basemap that needs no Mapbox token, and whose
        # muted palette keeps the markers legible.
        map_style=BASEMAP_STYLE,
        tooltip={
            "html": "<b>{name}</b><br/>{detail}",
            "style": {"backgroundColor": "#1f2933", "color": "white", "fontSize": "12px"},
        },
    )


def _place_row(place: Any, snapshot: MapSnapshot) -> dict[str, Any]:
    distance = round(snapshot.anchor.coordinate.distance_to(place.coordinate))
    detail = place.address.label or ""
    parts = [p for p in (place.categories[0] if place.categories else "", detail) if p]
    return {
        "lat": place.coordinate.lat,
        "lng": place.coordinate.lng,
        "name": place.title,
        "detail": " · ".join(parts) + f"<br/>{distance} m from the address",
    }


def _zoom_for(snapshot: MapSnapshot) -> float:
    """Pick a zoom that fits the furthest discovered place into the viewport."""
    radius_m = max(snapshot.bounds_padding(), 200.0)
    # Show a little beyond the furthest point, across a ~600px tall viewport.
    metres_per_pixel = (radius_m * 2.4) / VIEW_HEIGHT_PX
    ground_resolution = 156_543.03392 * math.cos(math.radians(snapshot.anchor.coordinate.lat))
    zoom = math.log2(ground_resolution / metres_per_pixel)
    return max(MIN_ZOOM, min(MAX_ZOOM, zoom))
