"""Office layout matching frontend/floorplan.jpg.

Coordinates are percentages of the floor plan image's width/height, in a
viewBox of "0 0 87 100" (the photo is 2112x2428px, ie width:height ~ 0.87,
so this keeps the SVG overlay pixel-aligned with the background photo
without distortion).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Desk:
    id: str
    x: float
    y: float


@dataclass
class Room:
    id: str
    name: str
    kind: str  # "desk_area" | "meeting_room" | "kitchen"
    x: float
    y: float
    width: float
    height: float
    capacity: int
    desks: list[Desk] = field(default_factory=list)
    # Purely-visual chair positions for meeting rooms (all share the room's
    # booked/free status - unlike `desks`, these are never individually
    # occupied, so they're not seeded into the desks/occupancy table).
    seats: list[Desk] = field(default_factory=list)
    # Short spoken-style walking directions, landmark-based, so the agent
    # can actually tell someone how to get there (not just that it's free).
    directions: str = ""


ROOMS: list[Room] = [
    Room(
        id="team-a",
        name="Team Room A",
        kind="desk_area",
        x=3, y=9.5, width=19, height=6.5,
        capacity=2,
        desks=[Desk("team-a-d1", 8.44, 13.59), Desk("team-a-d2", 19.57, 13.59)],
        directions="Ganz vorne links im ersten Raum.",
    ),
    Room(
        id="open-space-left",
        name="Open Space",
        kind="desk_area",
        x=3, y=28, width=19, height=19,
        capacity=6,
        desks=[
            Desk("open-space-left-d1", 9.89, 29.65), Desk("open-space-left-d2", 20.6, 29.65),
            Desk("open-space-left-d3", 9.89, 36.45), Desk("open-space-left-d4", 20.6, 36.45),
            Desk("open-space-left-d5", 9.89, 43.25), Desk("open-space-left-d6", 20.6, 43.25),
        ],
        directions="Links, direkt hinter Team Room A.",
    ),
    Room(
        id="team-b",
        name="Team Room B",
        kind="desk_area",
        x=3, y=52.5, width=22, height=15,
        capacity=4,
        desks=[
            Desk("team-b-d1", 7.11, 54.68), Desk("team-b-d2", 7.11, 66.22),
            Desk("team-b-d3", 22.45, 54.68), Desk("team-b-d4", 22.45, 66.22),
        ],
        directions="Links im Gang, unter dem Monitor-Schild, hinter dem Open Space.",
    ),
    # Note: there is no separate "Small Office 1" - that space (y ~68-80%)
    # is corridor leading to the stairwell, not a room with a desk.
    Room(
        id="small-office-2",
        name="Small Office 2",
        kind="desk_area",
        x=1.75, y=84, width=15.75, height=8,
        capacity=1,
        desks=[Desk("small-office-2-d1", 11.4, 87.6)],
        directions="Ganz hinten links, am Ende des Gangs kurz vor der Treppe.",
    ),
    Room(
        id="small-office-4",
        name="Small Office 4",
        kind="desk_area",
        x=24, y=87.5, width=14, height=7,
        capacity=1,
        desks=[Desk("small-office-4-d1", 30.68, 91.1)],
        directions="Hinten in der Mitte, gegenüber der Treppe.",
    ),
    Room(
        id="kitchen",
        name="Küche",
        kind="kitchen",
        x=26, y=58.5, width=18, height=15,
        capacity=0,
        directions="In der Mitte des Büros, zwischen Open Space und den Meetingräumen.",
    ),
    Room(
        id="meeting-large",
        name="Meeting Room (Large)",
        kind="meeting_room",
        x=50.5, y=36.5, width=36.5, height=28.5,
        capacity=6,
        seats=[
            Desk("meeting-large-c1", 56.44, 53.06), Desk("meeting-large-c2", 65.62, 53.06),
            Desk("meeting-large-c3", 56.44, 57.33), Desk("meeting-large-c4", 65.62, 57.33),
            Desk("meeting-large-c5", 56.44, 61.49), Desk("meeting-large-c6", 65.62, 61.49),
        ],
        directions="Rechts im Büro, hinter der Küche, der große Raum mit dem ovalen Tisch.",
    ),
    Room(
        id="team-c",
        name="Team Room C",
        kind="desk_area",
        x=50.5, y=66.5, width=19.5, height=11,
        capacity=2,
        desks=[Desk("team-c-d1", 55.0, 67.3), Desk("team-c-d2", 64.48, 67.3)],
        directions="Rechts, direkt unter dem großen Meetingraum.",
    ),
    Room(
        id="small-office-3",
        name="Small Office 3",
        kind="desk_area",
        x=73, y=86, width=14, height=7,
        capacity=1,
        desks=[Desk("small-office-3-d1", 66.53, 90.19)],
        directions="Rechts hinten, in der Ecke neben Team Room C.",
    ),
]

ROOMS_BY_ID: dict[str, Room] = {room.id: room for room in ROOMS}
