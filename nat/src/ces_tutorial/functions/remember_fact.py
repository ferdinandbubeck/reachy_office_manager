"""Tool that lets the agent save something worth remembering across
conversations (not just within the current one)."""

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from ces_tutorial import memory_store


class RememberFactConfig(FunctionBaseConfig, name="remember_fact"):
    """Tool that stores a short fact in persistent memory."""
    pass


@register_function(config_type=RememberFactConfig)
async def remember_fact(tool_config: RememberFactConfig, builder: Builder):

    async def _remember_fact(fact: str) -> str:
        memory_store.remember(fact)
        return "Gemerkt."

    yield FunctionInfo.from_fn(
        _remember_fact,
        description=(
            "Saves a short fact to persistent memory, so you can recall it in "
            "a completely different, later conversation (not just this one - "
            "you already have this conversation's own history for free). Use "
            "this for things worth remembering long-term: a stated preference "
            "('mag kein Fleisch'), a recurring detail about the user or "
            "office, something they explicitly asked you to remember. Keep "
            "each fact short and self-contained (it will be read back out of "
            "context later).\n\n"
            "Args:\n"
            "    fact (str): the fact to remember, written as a short standalone sentence."
        ),
    )
