"""Tool that lets the agent express a recorded Reachy Mini emotion."""

import httpx
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

EMOTION_NAMES = (
    "amazed1, anxiety1, attentive1, attentive2, boredom1, boredom2, calming1, "
    "cheerful1, come1, confused1, contempt1, curious1, disgusted1, "
    "displeased1, displeased2, downcast1, dying1, electric1, enthusiastic1, "
    "enthusiastic2, exhausted1, fear1, frustrated1, furious1, go_away1, "
    "grateful1, helpful1, helpful2, impatient1, impatient2, "
    "incomprehensible2, indifferent1, inquiring1, inquiring2, inquiring3, "
    "irritated1, irritated2, laughing1, laughing2, lonely1, lost1, loving1, "
    "no1, no_excited1, no_sad1, oops1, oops2, proud1, proud2, proud3, rage1, "
    "relief1, relief2, reprimand1, reprimand2, reprimand3, resigned1, sad1, "
    "sad2, scared1, serenity1, shy1, success1, success2, surprised1, "
    "surprised2, thoughtful1, thoughtful2, tired1, toc-toc-toc, uncertain1, "
    "uncomfortable1, understanding1, understanding2, waiting, welcoming1, "
    "welcoming2, yes1, yes_sad1"
)


class ReachyEmotionConfig(FunctionBaseConfig, name="reachy_emotion"):
    """Tool that plays a recorded Reachy Mini emotion by name."""
    bot_url: str = "http://localhost:7861"


@register_function(config_type=ReachyEmotionConfig)
async def reachy_emotion(tool_config: ReachyEmotionConfig, builder: Builder):

    async def _reachy_emotion(emotion_name: str) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{tool_config.bot_url}/emotion", json={"name": emotion_name}, timeout=15.0
            )
            response.raise_for_status()
            data = response.json()
        status = data.get("status")
        if status == "ok":
            return f"Expressed the '{emotion_name}' emotion physically."
        if status == "cooldown":
            return (
                "Skipped: an emotion or gesture played too recently, still "
                "on cooldown. Just answer normally without one this time."
            )
        return f"Could not play emotion '{emotion_name}' - it may not be a valid name."

    yield FunctionInfo.from_fn(
        _reachy_emotion,
        description=(
            "Makes the Reachy Mini robot physically express an emotion "
            "(head/antenna motion, no sound) matching the mood of the "
            "conversation - eg curiosity, surprise, being pleased, confusion, "
            "sadness. Use this whenever it would add genuine expressiveness "
            "to your response, not on every single reply.\n\n"
            f"Args:\n    emotion_name (str): one of: {EMOTION_NAMES}."
        ),
    )
