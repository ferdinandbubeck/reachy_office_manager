"""Tool that lets the agent take a fresh look through Reachy Mini's camera."""

import httpx
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class ReachySeeConfig(FunctionBaseConfig, name="reachy_see"):
    """Tool that captures a fresh camera frame and describes what's visible."""
    bot_url: str = "http://localhost:7861"


@register_function(config_type=ReachySeeConfig)
async def reachy_see(tool_config: ReachySeeConfig, builder: Builder):

    async def _reachy_see(question: str) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{tool_config.bot_url}/see", json={"question": question}, timeout=25.0
            )
            response.raise_for_status()
            data = response.json()
        return data.get("description", "I couldn't get a clear look just now.")

    yield FunctionInfo.from_fn(
        _reachy_see,
        description=(
            "Takes a fresh look through the Reachy Mini robot's own camera right "
            "now and returns a short description of what is currently visible. "
            "Use this whenever you need to answer what you (the robot) can see, "
            "including after turning to look in a new direction with reachy_look.\n\n"
            "Args:\n"
            "    question (str): what to look for or describe, eg 'describe the "
            "surroundings' or 'what color shirt is the person wearing'."
        ),
    )
