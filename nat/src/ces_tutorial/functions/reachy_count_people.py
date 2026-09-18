"""Tool that lets the agent count how many people are currently in view."""

import httpx
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class ReachyCountPeopleConfig(FunctionBaseConfig, name="reachy_count_people"):
    """Tool that captures a fresh camera frame and counts visible people."""
    bot_url: str = "http://localhost:7861"


@register_function(config_type=ReachyCountPeopleConfig)
async def reachy_count_people(tool_config: ReachyCountPeopleConfig, builder: Builder):

    async def _reachy_count_people(reason: str) -> str:
        # `reason` isn't used server-side - NAT's single_fn wrapper requires
        # exactly one parameter, and having the model state *why* it's
        # checking (eg "deciding between a private office and open desk")
        # costs nothing and keeps the tool call self-documenting in logs.
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{tool_config.bot_url}/count_people", json={}, timeout=25.0)
            response.raise_for_status()
            data = response.json()
        count = data.get("person_count", 0)
        confident = data.get("confident", False)
        notes = data.get("notes", "")
        hedge = "" if confident else " (not fully certain)"
        return f"person_count={count}{hedge}. {notes}".strip()

    yield FunctionInfo.from_fn(
        _reachy_count_people,
        description=(
            "Takes a fresh look through Reachy Mini's own camera and counts how "
            "many people are currently visible. Use this whenever you need to know "
            "whether someone is alone or with others (eg to decide between a "
            "private single office and a shared/meeting space) - call this "
            "yourself to find out, never ask the user whether they're alone.\n\n"
            "Args:\n"
            "    reason (str): why you're checking, eg 'deciding between a "
            "private office and open desk'."
        ),
    )
