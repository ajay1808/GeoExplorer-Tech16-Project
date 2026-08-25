"""Session state: de-duplication, isolation between sessions, thread safety."""

from __future__ import annotations

import threading

from conftest import EMPIRE_STATE, places_payload
from geoexplorer.models import Address, Coordinate, Isoline, Place
from geoexplorer.state import Anchor, SessionState


def _places(count: int) -> list[Place]:
    return [Place.from_here(item) for item in places_payload(count)["items"]]


def test_places_deduplicate_by_id(session_state: SessionState):
    session_state.record_places(_places(3))
    session_state.record_places(_places(3))

    assert len(session_state.snapshot().places) == 3


def test_snapshot_is_detached_from_later_writes(session_state: SessionState):
    session_state.record_places(_places(1))
    snapshot = session_state.snapshot()
    session_state.record_places(_places(3))

    assert len(snapshot.places) == 1


def test_clear_overlays_keeps_the_anchor(session_state: SessionState):
    session_state.record_places(_places(2))
    session_state.record_isoline(
        Isoline(center=EMPIRE_STATE, mode="pedestrian", minutes=15, boundary=[EMPIRE_STATE])
    )
    session_state.clear_overlays()
    snapshot = session_state.snapshot()

    assert snapshot.places == []
    assert snapshot.isoline is None
    assert snapshot.anchor.label == "350 5th Ave, New York"


def test_concurrent_writes_do_not_lose_places(session_state: SessionState):
    def writer(offset: int) -> None:
        session_state.record_places(
            [
                Place(
                    id=f"p{offset}-{i}",
                    title=f"Place {offset}-{i}",
                    coordinate=Coordinate(lat=40.0, lng=-73.0),
                    address=Address(),
                )
                for i in range(20)
            ]
        )

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(session_state.snapshot().places) == 8 * 20


def test_bounds_padding_reflects_the_furthest_place(session_state: SessionState):
    far = Place(
        id="far",
        title="Far",
        coordinate=Coordinate(lat=EMPIRE_STATE.lat + 0.02, lng=EMPIRE_STATE.lng),
        address=Address(),
    )
    session_state.record_places([far])

    assert session_state.snapshot().bounds_padding() > 1500


def test_two_sessions_do_not_share_places():
    a = SessionState(Anchor(label="A", coordinate=EMPIRE_STATE, is_precise=True))
    b = SessionState(Anchor(label="B", coordinate=EMPIRE_STATE, is_precise=True))
    a.record_places(_places(2))

    assert len(b.snapshot().places) == 0
