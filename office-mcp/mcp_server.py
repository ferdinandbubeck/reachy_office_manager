"""FastMCP server exposing office availability/booking tools.

Run with:
    uv run python mcp_server.py

Serves streamable-HTTP MCP on port 8002 (so a remote agent, eg the NAT
router-agent, can connect to it over the network rather than via stdio).
"""

from __future__ import annotations

import functools
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

import store
from office_layout import ROOMS, ROOMS_BY_ID

mcp = FastMCP("office-availability", host="0.0.0.0", port=8002, log_level="WARNING")


def _logged(fn):
    """Records every call to a tool (name, args, success/error) in the
    shared activity log, so the frontend can show a live feed of what the
    agent is actually doing via MCP."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            result = fn(*args, **kwargs)
            store.log_activity(fn.__name__, kwargs, ok=True)
            return result
        except Exception as e:
            store.log_activity(fn.__name__, kwargs, ok=False, error=str(e))
            raise

    return wrapper


@mcp.tool()
@_logged
def get_current_time() -> str:
    """Returns the current date/time in ISO 8601 (UTC), for reasoning about
    'now', 'in 30 minutes', etc. before calling book_room."""
    return datetime.now(timezone.utc).isoformat()


@mcp.tool()
@_logged
def list_rooms() -> list[dict]:
    """Lists every room/area in the office: id, name, kind
    (desk_area | meeting_room | kitchen), capacity, and short spoken-style
    walking directions to it (landmark-based) - use `directions` whenever
    you tell someone where to go, not just that a spot is free."""
    return [
        {"room_id": r.id, "name": r.name, "kind": r.kind, "capacity": r.capacity, "directions": r.directions}
        for r in ROOMS
    ]


@mcp.tool()
@_logged
def get_office_overview() -> list[dict]:
    """Returns live occupancy for every room: for desk areas, how many desks
    are free/occupied; for meeting rooms, whether they're booked right now
    and any upcoming bookings."""
    return store.office_overview()


@mcp.tool()
@_logged
def get_room_status(room_id: str) -> dict:
    """Returns live status for a single room by id (see list_rooms for
    valid ids). For desk areas this includes the individual desks; for
    meeting rooms it includes the current and upcoming bookings."""
    return store.room_occupancy(room_id)


@mcp.tool()
@_logged
def list_free_desks(room_id: str | None = None) -> list[dict]:
    """Lists currently unoccupied desks. Each desk includes its room's
    name, capacity and `directions` right here (no follow-up call needed to
    identify what kind of room it is or how to get there).

    Leave room_id unset to see EVERY free desk across the whole office in
    one call - always do this for a general question like "is there a
    free single office/desk anywhere", so a single occupied room doesn't
    make you wrongly conclude nothing is free. Only pass room_id when the
    user named that specific room."""
    desks = store.list_desks(room_id)
    free = [d for d in desks if not d["occupied"]]
    for d in free:
        room = ROOMS_BY_ID.get(d["room_id"])
        d["room_name"] = room.name if room else ""
        d["capacity"] = room.capacity if room else None
        d["directions"] = room.directions if room else ""
    return free


@mcp.tool()
@_logged
def book_desk(room_id: str, occupied_by: str) -> dict:
    """Occupies the first free desk in the given room for `occupied_by`
    (a person's name). Always pass the exact room_id from your most recent
    list_free_desks/get_room_status result - don't rely on memory of which
    room you mentioned a turn ago, since state can change between checking
    and booking. Fails if the room has no free desks; if it does, the error
    lists any equivalent room (same kind/capacity) that currently has a
    free desk - try one of those immediately rather than telling the user
    nothing is free. The response includes `directions` - use it verbatim
    when telling the person how to get there, never guess/invent
    directions."""
    free = [d for d in store.list_desks(room_id) if not d["occupied"]]
    if not free:
        room = ROOMS_BY_ID.get(room_id)
        alternatives = []
        if room is not None:
            for other in ROOMS:
                if other.id == room_id or other.kind != room.kind or other.capacity != room.capacity:
                    continue
                if any(not d["occupied"] for d in store.list_desks(other.id)):
                    alternatives.append(other.id)
        hint = f" Equivalent rooms with a free desk right now: {', '.join(alternatives)}." if alternatives else ""
        raise ValueError(f"No free desks in room '{room_id}'.{hint}")
    desk = free[0]
    store.set_desk_occupied(desk["id"], True, occupied_by)
    room = ROOMS_BY_ID.get(room_id)
    return {
        "desk_id": desk["id"], "room_id": room_id,
        "room_name": room.name if room else "", "capacity": room.capacity if room else None,
        "occupied_by": occupied_by, "directions": room.directions if room else "",
    }


@mcp.tool()
@_logged
def release_desk(desk_id: str) -> dict:
    """Marks a desk as free again (eg the person left for the day)."""
    if not store.set_desk_occupied(desk_id, False):
        raise ValueError(f"Unknown desk_id: {desk_id}")
    return {"desk_id": desk_id, "occupied": False}


@mcp.tool()
@_logged
def list_meeting_rooms() -> list[dict]:
    """Lists just the meeting rooms (id, name, capacity) - use this to know
    which room_ids are valid for book_room."""
    return [
        {"room_id": r.id, "name": r.name, "capacity": r.capacity}
        for r in ROOMS if r.kind == "meeting_room"
    ]


@mcp.tool()
@_logged
def check_room_availability(room_id: str, start_time: str, end_time: str) -> dict:
    """Checks whether a meeting room is free for a given ISO 8601 time
    range (start_time/end_time). Returns available=False and the
    conflicting booking if it's already taken."""
    conflict = store.find_conflicting_booking(room_id, start_time, end_time)
    return {
        "room_id": room_id,
        "available": conflict is None,
        "conflicting_booking": conflict.__dict__ if conflict else None,
    }


@mcp.tool()
@_logged
def book_room(room_id: str, title: str, start_time: str, end_time: str, booked_by: str | None = None) -> dict:
    """Books a meeting room for a time range (ISO 8601 start_time/end_time).
    Raises an error if the room is already booked for an overlapping time -
    call check_room_availability first if you want to confirm beforehand.
    The response includes `directions` - use it verbatim when telling the
    person how to get there, never guess/invent directions."""
    if room_id not in ROOMS_BY_ID or ROOMS_BY_ID[room_id].kind != "meeting_room":
        raise ValueError(f"'{room_id}' is not a valid meeting room id (see list_meeting_rooms)")
    conflict = store.find_conflicting_booking(room_id, start_time, end_time)
    if conflict is not None:
        raise ValueError(
            f"Room '{room_id}' is already booked '{conflict.title}' from "
            f"{conflict.start_time} to {conflict.end_time}"
        )
    booking = store.create_booking(room_id, title, start_time, end_time, booked_by)
    return {**booking.__dict__, "directions": ROOMS_BY_ID[room_id].directions}


@mcp.tool()
@_logged
def cancel_booking(booking_id: int) -> dict:
    """Cancels a booking by id (see get_room_status for booking ids)."""
    if not store.cancel_booking(booking_id):
        raise ValueError(f"Unknown booking_id: {booking_id}")
    return {"booking_id": booking_id, "cancelled": True}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
