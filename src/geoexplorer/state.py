"""Per-conversation state shared between the agent's tools and the UI.

The tools need somewhere to record what they found so the map can draw it, and the UI
reads that from Streamlit's synchronous script thread while the agent writes to it from
its own event-loop thread. A plain lock-guarded object is the right tool: workflow
`Context` state is reachable only from async code on the loop that owns it, which makes
it awkward — and, across `asyncio.run` boundaries, unsafe — to read from the UI side.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from .models import Coordinate, GeocodeResult, Isoline, Place


@dataclass
class Anchor:
    """The address the whole conversation is about."""

    label: str
    coordinate: Coordinate
    is_precise: bool

    @classmethod
    def from_geocode(cls, result: GeocodeResult) -> Anchor:
        return cls(
            label=result.address.label,
            coordinate=result.coordinate,
            is_precise=result.is_precise,
        )


class SessionState:
    """Thread-safe record of the anchor, the places found so far, and any isoline."""

    def __init__(self, anchor: Anchor) -> None:
        self._lock = threading.Lock()
        self._anchor = anchor
        self._places: dict[str, Place] = {}
        self._isoline: Isoline | None = None

    @property
    def anchor(self) -> Anchor:
        with self._lock:
            return self._anchor

    def record_places(self, places: list[Place]) -> None:
        """Add places to the map, keeping the first sighting of each HERE place id."""
        with self._lock:
            for place in places:
                self._places.setdefault(place.id, place)

    def record_isoline(self, isoline: Isoline) -> None:
        with self._lock:
            self._isoline = isoline

    def snapshot(self) -> MapSnapshot:
        """An immutable copy for the renderer, taken without holding the lock during draw."""
        with self._lock:
            return MapSnapshot(
                anchor=self._anchor,
                places=list(self._places.values()),
                isoline=self._isoline,
            )

    def clear_overlays(self) -> None:
        """Drop discovered places and polygons, keeping the anchor."""
        with self._lock:
            self._places.clear()
            self._isoline = None


@dataclass(frozen=True)
class MapSnapshot:
    """What the map draws right now."""

    anchor: Anchor
    places: list[Place] = field(default_factory=list)
    isoline: Isoline | None = None

    def bounds_padding(self) -> float:
        """A zoom-ish hint: how far the furthest place sits from the anchor, in metres."""
        if not self.places:
            return 500.0
        return max(
            self.anchor.coordinate.distance_to(p.coordinate) for p in self.places
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "anchor": self.anchor.label,
            "place_count": len(self.places),
            "has_isoline": self.isoline is not None,
        }
