"""Tool that toggles Reachy Mini's visual head-tracking (following a face)."""

import httpx
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class ReachyFaceTrackingConfig(FunctionBaseConfig, name="reachy_face_tracking"):
    """Tool that enables/disables Reachy Mini's head tracking of a face."""
    bot_url: str = "http://localhost:7861"


@register_function(config_type=ReachyFaceTrackingConfig)
async def reachy_face_tracking(tool_config: ReachyFaceTrackingConfig, builder: Builder):

    async def _reachy_face_tracking(enabled: bool) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{tool_config.bot_url}/face_tracking", json={"enabled": enabled}, timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
        if data.get("status") == "ok":
            return "Face tracking is now on - my head will follow the detected face." if enabled \
                else "Face tracking is now off."
        return "Could not change face tracking right now."

    yield FunctionInfo.from_fn(
        _reachy_face_tracking,
        description=(
            "Turns Reachy Mini's automatic visual face-tracking on or off. "
            "While on, the robot's head continuously follows a detected "
            "face instead of staying still. Enable it when the user asks "
            "you to look at them, follow them, or keep watching them; "
            "disable it again if they ask you to stop, or before doing a "
            "deliberate reachy_look turn (tracking would otherwise fight "
            "the manual movement).\n\n"
            "Args:\n    enabled (bool): true to start tracking, false to stop."
        ),
    )
