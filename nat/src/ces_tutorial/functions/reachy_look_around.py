"""Tool that lets the agent scan the room by turning through several fixed
positions and describing what's visible at each, so it can reason about
where things/people are relative to it."""

import httpx
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class ReachyLookAroundConfig(FunctionBaseConfig, name="reachy_look_around"):
    """Tool that turns Reachy through left/front/right, describing each."""
    bot_url: str = "http://localhost:7861"


@register_function(config_type=ReachyLookAroundConfig)
async def reachy_look_around(tool_config: ReachyLookAroundConfig, builder: Builder):

    async def _reachy_look_around(reason: str) -> str:
        # The bot's control API is a plain single-threaded
        # http.server.HTTPServer blocking for ~10-15s on this endpoint
        # (3 physical turns + 3 vision calls) - that occasionally drops the
        # connection with an empty httpx.ReadError even though the request
        # actually completes fine server-side (confirmed by calling it
        # directly). One retry papers over that flakiness cheaply.
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(75.0, connect=10.0)) as client:
                    response = await client.post(f"{tool_config.bot_url}/look_around", json={})
                    response.raise_for_status()
                    data = response.json()
                return data.get("summary", "Der Rundumblick hat nicht funktioniert.")
            except httpx.HTTPError as e:
                last_error = e
        return f"Der Rundumblick ist gerade zweimal fehlgeschlagen ({last_error}). Bitte kurz nochmal versuchen."

    yield FunctionInfo.from_fn(
        _reachy_look_around,
        description=(
            "Turns Reachy Mini through several fixed positions (left, front, "
            "right) with natural head movement, pausing briefly at each to look "
            "around, and returns what's visible in every direction as one "
            "combined summary. Use this for a broader 'look around'/'what's in "
            "this room' request, instead of calling reachy_look + reachy_see "
            "repeatedly yourself. Unlike reachy_see (always a single fresh look "
            "you must re-take every time), you MAY remember this scan's result "
            "for the rest of the conversation and reason about where things or "
            "people are relative to you - re-scan only if the room could "
            "plausibly have changed (time has passed, someone moved, etc.).\n\n"
            "Args:\n"
            "    reason (str): why you're scanning, eg 'user asked what's in the room'."
        ),
    )
