"""Tool that lets the agent recall everything saved via remember_fact,
across all past conversations."""

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from ces_tutorial import memory_store


class RecallFactsConfig(FunctionBaseConfig, name="recall_facts"):
    """Tool that returns every fact saved in persistent memory."""
    pass


@register_function(config_type=RecallFactsConfig)
async def recall_facts(tool_config: RecallFactsConfig, builder: Builder):

    async def _recall_facts(reason: str) -> str:
        # `reason` isn't used server-side - NAT's single_fn wrapper requires
        # exactly one parameter; stating why you're recalling costs nothing
        # and keeps the tool call self-documenting in logs.
        facts = memory_store.recall_all()
        if not facts:
            return "Keine gespeicherten Erinnerungen vorhanden."
        return "\n".join(f"- {f['fact']}" for f in facts)

    yield FunctionInfo.from_fn(
        _recall_facts,
        description=(
            "Returns every fact previously saved via remember_fact, from any "
            "past conversation. Call this whenever a stored preference or "
            "detail might be relevant (eg before recommending food, a desk, "
            "or anything personal) rather than asking the user to repeat "
            "themselves.\n\n"
            "Args:\n"
            "    reason (str): why you're recalling, eg 'checking for food preferences'."
        ),
    )
