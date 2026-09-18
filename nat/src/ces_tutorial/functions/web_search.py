"""Tool that lets the agent search the live web (DuckDuckGo, no API key
needed) - for anything that changes over time or isn't in the LLM's
training data, eg today's cafeteria menu or nearby restaurants."""

import asyncio

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class WebSearchConfig(FunctionBaseConfig, name="web_search"):
    """Tool that searches the web via DuckDuckGo and returns top results."""
    max_results: int = 4


@register_function(config_type=WebSearchConfig)
async def web_search(tool_config: WebSearchConfig, builder: Builder):

    async def _web_search(query: str) -> str:
        from ddgs import DDGS

        def _run_search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=tool_config.max_results))

        try:
            results = await asyncio.to_thread(_run_search)
        except Exception as e:
            return f"Web search failed: {e}"

        if not results:
            return "No web results found for that query."

        lines = []
        for r in results:
            title = r.get("title", "")
            snippet = r.get("body", "")
            lines.append(f"- {title}: {snippet}")
        return "\n".join(lines)

    yield FunctionInfo.from_fn(
        _web_search,
        description=(
            "Searches the live web (DuckDuckGo) and returns short result "
            "snippets. Use this for anything that changes over time or that "
            "you can't know from training alone - eg today's cafeteria menu "
            "('Speiseplan'), nearby restaurants, current events, opening "
            "hours. Summarize the results in your own words when answering, "
            "don't just read snippets verbatim.\n\n"
            "Args:\n"
            "    query (str): the search query, eg 'Speiseplan Kantine [Stadt] heute'."
        ),
    )
