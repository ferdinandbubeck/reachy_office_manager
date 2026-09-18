"""Tiny local HTTP server exposing robot capabilities to the NAT agent.

The NAT service (react_agent tools) runs in a separate process from the bot,
so it can't call ReachyService directly. This lightweight server (no extra
framework) lets it trigger head/body movement and request a fresh
description of what the camera currently sees over plain HTTP POSTs.

POST /look expects `{"direction": "left"|"right"|"up"|"down"|"front"|"around"}`
and blocks until the physical movement has finished, so the agent's next
tool call (typically reachy_see) sees the robot already in its new pose
instead of the pre-turn view.

POST /see is synchronous: it captures a fresh camera frame and describes it
with a vision model, returning `{"description": "..."}`. The agent uses this
as a tool it can decide to call itself (e.g. after turning to look
somewhere), instead of images being force-attached to every message.

POST /look_around takes no body: turns through left/front/right (reusing
/look's natural head movement at each stop) and describes what's visible in
each direction, returning one combined `{"summary": "..."}` - a snapshot
the agent can remember/reason about, unlike /see's single always-fresh look.

POST /gesture expects `{"name": "<dance move name>"}` and POST /emotion
expects `{"name": "<emotion name>"}`; both block until the move finishes
playing.
"""

import asyncio
import base64
import io
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .reachy_service import ReachyService

logger = logging.getLogger(__name__)

VALID_DIRECTIONS = ("left", "right", "up", "down", "front")
SETTLE_BUFFER = 0.3  # extra margin so the frame we then grab isn't mid-motion

# Set once run_bot() is up, so the (thread-based) HTTP handler can schedule
# coroutines onto the bot's actual asyncio event loop.
_main_loop: asyncio.AbstractEventLoop | None = None


def set_main_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _main_loop
    _main_loop = loop


def _do_look(direction: str) -> bool:
    """Runs the move and blocks until it's actually done settling. Returns
    whether the robot was actually connected/moved (a 0.0 duration means
    look_at()/goto_body_yaw() silently no-opped, eg because Reachy isn't
    connected right now)."""
    service = ReachyService.get_instance()
    succeeded = False
    if direction == "around":
        for body_yaw_deg, head_yaw_deg in ((-90, -90), (90, 90), (0, 0)):
            duration = service.goto_body_yaw(body_yaw_deg, head_yaw_deg)
            if duration > 0:
                succeeded = True
                time.sleep(duration + SETTLE_BUFFER)
    else:
        duration = service.look_at(direction if direction in VALID_DIRECTIONS else "front")
        if duration > 0:
            succeeded = True
            time.sleep(duration + SETTLE_BUFFER)
    return succeeded


# A curated, general-purpose subset of the full emotions-library (see
# reachy_emotion.py for all 85) used to ground the vision model's reaction
# suggestion to something the agent can reliably act on.
REACTION_EMOTIONS = (
    "surprised1", "amazed1", "curious1", "cheerful1", "welcoming1",
    "confused1", "thoughtful1", "laughing1", "uncertain1", "indifferent1",
    "disgusted1", "boredom1", "attentive1", "shy1", "loving1",
)


async def _describe_scene(question: str = "") -> str:
    frame = await ReachyService.get_instance().capture_fresh_frame()
    if frame is None:
        return "I couldn't get a clear view from my camera just now."

    from PIL import Image
    from openai import AsyncOpenAI

    image = Image.fromarray(frame, mode="RGB")
    max_dim = 512
    w, h = image.size
    if w > max_dim or h > max_dim:
        if w > h:
            image = image.resize((max_dim, int(max_dim / w * h)), Image.Resampling.BILINEAR)
        else:
            image = image.resize((int(max_dim / h * w), max_dim), Image.Resampling.BILINEAR)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=70)
    data_url = f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are the Reachy Mini robot looking through your own camera "
                            "right now. In 1-2 short sentences, describe what you currently "
                            "see in direct first-person terms, focusing on: "
                            f"{question or 'a general description of the scene'}. "
                            "Also pick the single emotion from this list that best matches "
                            "how a robot would genuinely react to seeing this: "
                            f"{', '.join(REACTION_EMOTIONS)}."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "auto"}},
                ],
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "scene_description",
                "schema": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "reaction_emotion": {"type": "string", "enum": list(REACTION_EMOTIONS)},
                    },
                    "required": ["description", "reaction_emotion"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
        max_tokens=200,
    )

    import json as _json
    try:
        parsed = _json.loads(response.choices[0].message.content or "{}")
        description = parsed.get("description") or "I'm not sure what I'm looking at."
        emotion = parsed.get("reaction_emotion")
    except (_json.JSONDecodeError, AttributeError):
        return response.choices[0].message.content or "I'm not sure what I'm looking at."

    if emotion:
        return (
            f"{description} "
            f"(Suggested physical reaction: call reachy_emotion with '{emotion}' if it fits.)"
        )
    return description


SCAN_DIRECTIONS = ("left", "front", "right")


def _do_scan_room() -> str:
    """Turns through SCAN_DIRECTIONS, pausing at each (look_at's existing
    settle time + antenna flick already gives this a natural, curious-glance
    feel - no extra choreography needed here) to describe what's visible.
    Runs in the HTTP handler thread, same as _do_look; dispatches each
    description onto the bot's asyncio loop and blocks for the result,
    mirroring how /look and /see each already work individually."""
    observations = []
    for direction in SCAN_DIRECTIONS:
        succeeded = _do_look(direction)
        if not succeeded:
            observations.append(f"{direction}: Reachy ist gerade nicht verbunden.")
            continue
        if _main_loop is None:
            observations.append(f"{direction}: Kamera ist nicht bereit.")
            continue
        try:
            future = asyncio.run_coroutine_threadsafe(
                _describe_scene("welche Objekte und Personen sind zu sehen, und wo befinden sie sich ungefähr"),
                _main_loop,
            )
            description = future.result(timeout=20.0)
        except Exception as e:
            logger.warning(f"Scan description failed for direction '{direction}': {e}")
            description = "(konnte nichts erkennen)"
        # Strip any appended emotion-reaction suggestion - noise in a combined summary.
        description = description.split(" (Suggested physical reaction:")[0].strip()
        observations.append(f"{direction}: {description}")
    return "\n".join(observations)


async def _count_people() -> dict:
    """Captures a fresh frame and asks a vision model specifically for a
    person count - a narrower, more reliable task than free-form
    description, meant for the agent to call itself instead of asking the
    user whether they're alone."""
    frame = await ReachyService.get_instance().capture_fresh_frame()
    if frame is None:
        return {"person_count": 0, "confident": False, "notes": "no camera frame available"}

    from PIL import Image
    from openai import AsyncOpenAI

    image = Image.fromarray(frame, mode="RGB")
    max_dim = 512
    w, h = image.size
    if w > max_dim or h > max_dim:
        if w > h:
            image = image.resize((max_dim, int(max_dim / w * h)), Image.Resampling.BILINEAR)
        else:
            image = image.resize((int(max_dim / h * w), max_dim), Image.Resampling.BILINEAR)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=70)
    data_url = f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Count the distinct people clearly visible in this image "
                            "from a robot's camera. Only count people you can actually "
                            "see, not people implied off-frame."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "auto"}},
                ],
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "people_count",
                "schema": {
                    "type": "object",
                    "properties": {
                        "person_count": {"type": "integer"},
                        "confident": {"type": "boolean"},
                        "notes": {"type": "string"},
                    },
                    "required": ["person_count", "confident", "notes"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
        max_tokens=100,
    )

    try:
        parsed = json.loads(response.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return {"person_count": 0, "confident": False, "notes": "vision model returned invalid data"}
    return {
        "person_count": int(parsed.get("person_count", 0)),
        "confident": bool(parsed.get("confident", False)),
        "notes": str(parsed.get("notes", "")),
    }


class ReachyControlHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silence default request logging

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b""
        try:
            return json.loads(body) if body else {}
        except json.JSONDecodeError:
            return {}

    def _respond_json(self, status: int, payload: dict) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def do_POST(self):
        if self.path == "/look":
            data = self._read_json_body()
            direction = str(data.get("direction", "front")).strip().lower()
            succeeded = _do_look(direction)  # blocks until the move has settled
            self._respond_json(200, {"status": "ok" if succeeded else "failed", "direction": direction})
            return

        if self.path == "/see":
            if _main_loop is None:
                self._respond_json(503, {"error": "bot event loop not ready"})
                return
            data = self._read_json_body()
            question = str(data.get("question", "") or "")
            try:
                future = asyncio.run_coroutine_threadsafe(_describe_scene(question), _main_loop)
                description = future.result(timeout=20.0)
                self._respond_json(200, {"description": description})
            except Exception as e:
                logger.warning(f"Failed to describe scene: {e}")
                self._respond_json(200, {"description": "I couldn't get a clear look just now."})
            return

        if self.path == "/look_around":
            self._read_json_body()  # drain the request body even though we ignore it -
            # every other handler does this; skipping it here left an
            # unread body sitting in the socket while _do_scan_room()
            # blocked for ~10s, which is a plausible cause of the
            # connection resets/ReadErrors NAT's httpx client was hitting.
            try:
                summary = _do_scan_room()
                self._respond_json(200, {"summary": summary})
            except Exception as e:
                logger.warning(f"Look-around scan failed: {e}")
                self._respond_json(200, {"summary": "Der Rundumblick hat gerade nicht funktioniert."})
            return

        if self.path == "/count_people":
            if _main_loop is None:
                self._respond_json(503, {"error": "bot event loop not ready"})
                return
            try:
                future = asyncio.run_coroutine_threadsafe(_count_people(), _main_loop)
                result = future.result(timeout=20.0)
                self._respond_json(200, result)
            except Exception as e:
                logger.warning(f"Failed to count people: {e}")
                self._respond_json(200, {"person_count": 0, "confident": False, "notes": "error"})
            return

        if self.path == "/gesture":
            data = self._read_json_body()
            name = str(data.get("name", "")).strip()
            service = ReachyService.get_instance()
            duration = service.play_gesture(name)
            status = "ok" if duration > 0 else ("cooldown" if duration < 0 else "failed")
            if duration > 0:
                time.sleep(duration + SETTLE_BUFFER)
            self._respond_json(200, {"status": status, "name": name})
            return

        if self.path == "/emotion":
            data = self._read_json_body()
            name = str(data.get("name", "")).strip()
            service = ReachyService.get_instance()
            duration = service.play_emotion(name)
            status = "ok" if duration > 0 else ("cooldown" if duration < 0 else "failed")
            if duration > 0:
                time.sleep(duration + SETTLE_BUFFER)
            self._respond_json(200, {"status": status, "name": name})
            return

        if self.path == "/face_tracking":
            data = self._read_json_body()
            enabled = bool(data.get("enabled", False))
            service = ReachyService.get_instance()
            ok = service.set_face_tracking(enabled)
            self._respond_json(200, {"status": "ok" if ok else "failed", "enabled": enabled})
            return

        self.send_response(404)
        self.end_headers()


def start_control_api(port: int = 7861) -> ThreadingHTTPServer:
    # Threading, not plain HTTPServer: /look_around alone blocks for
    # ~10-15s (3 physical turns + 3 vision calls), and a single-threaded
    # server handling only one connection at a time is a plausible source
    # of the intermittent connection drops seen from NAT's httpx client
    # during long requests. Each request gets its own thread here instead.
    server = ThreadingHTTPServer(("127.0.0.1", port), ReachyControlHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Reachy control API listening on http://127.0.0.1:{port}")
    return server
