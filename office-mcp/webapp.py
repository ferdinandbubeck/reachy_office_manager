"""Small FastAPI app: serves the floor-plan frontend and a REST API the
frontend uses to read/write the same SQLite store the MCP server uses.

Run with:
    uv run uvicorn webapp:app --port 8003 --reload
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import store
from office_layout import ROOMS, ROOMS_BY_ID

app = FastAPI(title="Office Availability")


class DeskToggleRequest(BaseModel):
    occupied: bool
    occupied_by: str | None = None


class BookingRequest(BaseModel):
    title: str
    booked_by: str | None = None
    start_time: str | None = None  # ISO 8601; defaults to now
    duration_minutes: int = 30


@app.get("/api/layout")
def get_layout():
    """Static room/desk geometry for rendering the floor plan."""
    return [
        {
            "id": r.id,
            "name": r.name,
            "kind": r.kind,
            "x": r.x,
            "y": r.y,
            "width": r.width,
            "height": r.height,
            "capacity": r.capacity,
            "desks": [{"id": d.id, "x": d.x, "y": d.y} for d in r.desks],
            "seats": [{"id": s.id, "x": s.x, "y": s.y} for s in r.seats],
        }
        for r in ROOMS
    ]


@app.get("/api/state")
def get_state():
    """Live occupancy/booking state for every room."""
    return store.office_overview()


@app.get("/api/logs")
def get_logs(limit: int = 30):
    """Recent MCP tool-call activity, for the live activity feed."""
    return store.get_recent_activity(limit)


@app.post("/api/desks/{desk_id}/toggle")
def toggle_desk(desk_id: str, body: DeskToggleRequest):
    if not store.set_desk_occupied(desk_id, body.occupied, body.occupied_by):
        raise HTTPException(404, f"Unknown desk_id: {desk_id}")
    return {"desk_id": desk_id, "occupied": body.occupied}


@app.post("/api/rooms/{room_id}/book")
def book_room(room_id: str, body: BookingRequest):
    if room_id not in ROOMS_BY_ID or ROOMS_BY_ID[room_id].kind != "meeting_room":
        raise HTTPException(400, f"'{room_id}' is not a meeting room")
    start = datetime.fromisoformat(body.start_time) if body.start_time else datetime.now(timezone.utc)
    end = start + timedelta(minutes=body.duration_minutes)
    start_iso, end_iso = start.isoformat(), end.isoformat()

    conflict = store.find_conflicting_booking(room_id, start_iso, end_iso)
    if conflict is not None:
        raise HTTPException(409, f"Room already booked: '{conflict.title}'")

    booking = store.create_booking(room_id, body.title, start_iso, end_iso, body.booked_by)
    return booking.__dict__


@app.post("/api/bookings/{booking_id}/cancel")
def cancel_booking(booking_id: int):
    if not store.cancel_booking(booking_id):
        raise HTTPException(404, f"Unknown booking_id: {booking_id}")
    return {"booking_id": booking_id, "cancelled": True}


@app.get("/")
def index():
    return FileResponse("frontend/index.html")


app.mount("/static", StaticFiles(directory="frontend"), name="static")
