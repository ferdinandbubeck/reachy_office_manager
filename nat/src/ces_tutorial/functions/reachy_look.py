"""Tool that lets the agent make Reachy Mini physically turn its head."""

import httpx
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

VALID_DIRECTIONS = ("left", "right", "up", "down", "front", "around")


class ReachyLookConfig(FunctionBaseConfig, name="reachy_look"):
    """Tool that makes the Reachy Mini robot turn its head to look in a direction."""
    bot_url: str = "http://localhost:7861"


@register_function(config_type=ReachyLookConfig)
async def reachy_look(tool_config: ReachyLookConfig, builder: Builder):

    async def _reachy_look(direction: str) -> str:
        direction = (direction or "front").strip().lower()
        if direction not in VALID_DIRECTIONS:
            direction = "front"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{tool_config.bot_url}/look", json={"direction": direction}, timeout=15.0
            )
            response.raise_for_status()
            data = response.json()

        if data.get("status") != "ok":
            return (
                "Could not physically move - Reachy doesn't seem to be connected right "
                "now. Tell the user honestly that you can't turn/move at the moment, "
                "don't pretend you did."
            )

        if direction == "around":
            return "Reachy is looking around the room."
        return f"Reachy turned its head to look {direction}."

    yield FunctionInfo.from_fn(
        _reachy_look,
        description=(
            "Makes the Reachy Mini robot physically turn its head to look in a "
            "direction. Use this whenever the user asks the robot to look around, "
            "turn its head, or look left/right/up/down.\n\n"
            "Args:\n"
            "    direction (str): one of 'left', 'right', 'up', 'down', 'front', "
            "or 'around' (scans left then right then returns to front)."
        ),
    )
