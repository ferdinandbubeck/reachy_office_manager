"""SQLite-backed state for desks and room bookings.

Shared by the MCP server and the frontend's API - both are separate
processes, so SQLite (not an in-memory dict) is the source of truth. Simple
on purpose: this backs a showcase demo, not a production booking system.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from office_layout import ROOMS, ROOMS_BY_ID

DB_PATH = Path(__file__).parent / "office.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS desks (
                id TEXT PRIMARY KEY,
                room_id TEXT NOT NULL,
                occupied INTEGER NOT NULL DEFAULT 0,
                occupied_by TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bookings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room_id TEXT NOT NULL,
                title TEXT NOT NULL,
                booked_by TEXT,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                tool TEXT NOT NULL,
                params TEXT,
                ok INTEGER NOT NULL,
                error TEXT
            )
            """
        )
        # Seed desks for every desk_area room, ignoring ones that already exist.
        for room in ROOMS:
            for desk in room.desks:
                conn.execute(
                    "INSERT OR IGNORE INTO desks (id, room_id, occupied, occupied_by) VALUES (?, ?, 0, NULL)",
                    (desk.id, room.id),
                )


@dataclass
class Booking:
    id: int
    room_id: str
    title: str
    booked_by: str | None
    start_time: str
    end_time: str


def _row_to_booking(row: sqlite3.Row) -> Booking:
    return Booking(
        id=row["id"],
        room_id=row["room_id"],
        title=row["title"],
        booked_by=row["booked_by"],
        start_time=row["start_time"],
        end_time=row["end_time"],
    )


def list_desks(room_id: str | None = None) -> list[dict]:
    with _connect() as conn:
        if room_id:
            rows = conn.execute("SELECT * FROM desks WHERE room_id = ? ORDER BY id", (room_id,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM desks ORDER BY id").fetchall()
        return [dict(row) for row in rows]


def set_desk_occupied(desk_id: str, occupied: bool, occupied_by: str | None = None) -> bool:
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE desks SET occupied = ?, occupied_by = ? WHERE id = ?",
            (1 if occupied else 0, occupied_by if occupied else None, desk_id),
        )
        return cur.rowcount > 0


def list_bookings(room_id: str | None = None, active_only: bool = False) -> list[Booking]:
    with _connect() as conn:
        query = "SELECT * FROM bookings"
        params: list = []
        clauses = []
        if room_id:
            clauses.append("room_id = ?")
            params.append(room_id)
        if active_only:
            now = _now()
            clauses.append("start_time <= ? AND end_time >= ?")
            params.extend([now, now])
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY start_time"
        rows = conn.execute(query, params).fetchall()
        return [_row_to_booking(row) for row in rows]


def find_conflicting_booking(room_id: str, start_time: str, end_time: str) -> Booking | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT * FROM bookings
            WHERE room_id = ? AND start_time < ? AND end_time > ?
            LIMIT 1
            """,
            (room_id, end_time, start_time),
        ).fetchone()
        return _row_to_booking(row) if row else None


def create_booking(room_id: str, title: str, start_time: str, end_time: str, booked_by: str | None = None) -> Booking:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO bookings (room_id, title, booked_by, start_time, end_time, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (room_id, title, booked_by, start_time, end_time, _now()),
        )
        booking_id = cur.lastrowid
        row = conn.execute("SELECT * FROM bookings WHERE id = ?", (booking_id,)).fetchone()
        return _row_to_booking(row)


def cancel_booking(booking_id: int) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM bookings WHERE id = ?", (booking_id,))
        return cur.rowcount > 0


def room_occupancy(room_id: str) -> dict:
    """Computed status for a single room: desk occupancy for desk areas,
    current/next booking for meeting rooms."""
    room = ROOMS_BY_ID.get(room_id)
    if room is None:
        raise KeyError(f"Unknown room_id: {room_id}")

    if room.kind == "desk_area":
        desks = list_desks(room_id)
        occupied = sum(1 for d in desks if d["occupied"])
        return {
            "room_id": room.id,
            "name": room.name,
            "kind": room.kind,
            "capacity": room.capacity,
            "directions": room.directions,
            "occupied_desks": occupied,
            "free_desks": len(desks) - occupied,
            "desks": desks,
        }

    if room.kind == "meeting_room":
        now_booking = list_bookings(room_id, active_only=True)
        upcoming = [b for b in list_bookings(room_id) if b.end_time > _now()]
        return {
            "room_id": room.id,
            "name": room.name,
            "kind": room.kind,
            "capacity": room.capacity,
            "directions": room.directions,
            "booked_now": bool(now_booking),
            "current_booking": now_booking[0].__dict__ if now_booking else None,
            "upcoming_bookings": [b.__dict__ for b in upcoming],
        }

    return {
        "room_id": room.id, "name": room.name, "kind": room.kind,
        "capacity": room.capacity, "directions": room.directions,
    }


def office_overview() -> list[dict]:
    return [room_occupancy(room.id) for room in ROOMS]


def log_activity(tool: str, params: dict, ok: bool, error: str | None = None) -> None:
    """Records one MCP tool invocation, so the frontend can show a live feed
    of what the agent is actually calling."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO activity_log (ts, tool, params, ok, error) VALUES (?, ?, ?, ?, ?)",
            (_now(), tool, json.dumps(params, default=str), 1 if ok else 0, error),
        )


def get_recent_activity(limit: int = 30) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM activity_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            try:
                d["params"] = json.loads(d["params"]) if d["params"] else {}
            except (TypeError, ValueError):
                pass
            d["ok"] = bool(d["ok"])
            out.append(d)
        return out


init_db()
