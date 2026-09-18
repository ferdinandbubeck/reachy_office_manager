"""Tool that lets the agent play a Reachy Mini dance/gesture move."""

import httpx
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

GESTURE_NAMES = (
    "chicken_peck, chin_lead, dizzy_spin, grid_snap, groovy_sway_and_roll, "
    "head_tilt_roll, headbanger_combo, interwoven_spirals, jackson_square, "
    "neck_recoil, pendulum_swing, polyrhythm_combo, sharp_side_tilt, "
    "side_glance_flick, side_peekaboo, side_to_side_sway, simple_nod, "
    "stumble_and_recover, uh_huh_tilt, yeah_nod"
)


class ReachyGestureConfig(FunctionBaseConfig, name="reachy_gesture"):
    """Tool that plays a Reachy Mini dance/gesture move by name."""
    bot_url: str = "http://localhost:7861"


@register_function(config_type=ReachyGestureConfig)
async def reachy_gesture(tool_config: ReachyGestureConfig, builder: Builder):

    async def _reachy_gesture(move_name: str) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{tool_config.bot_url}/gesture", json={"name": move_name}, timeout=15.0
            )
            response.raise_for_status()
            data = response.json()
        status = data.get("status")
        if status == "ok":
            return f"Played the '{move_name}' gesture."
        if status == "cooldown":
            return (
                "Skipped: an emotion or gesture played too recently, still "
                "on cooldown. Just answer normally without one this time."
            )
        return f"Could not play gesture '{move_name}' - it may not be a valid name."

    yield FunctionInfo.from_fn(
        _reachy_gesture,
        description=(
            "Makes the Reachy Mini robot perform a fun physical dance/gesture "
            "move. Use this for playful moments, when asked to dance, or to "
            "add personality to a response.\n\n"
            f"Args:\n    move_name (str): one of: {GESTURE_NAMES}."
        ),
    )
