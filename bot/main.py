#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import asyncio
import base64
import io
import json
import logging
import os
import sys
import warnings

from dotenv import load_dotenv
from loguru import logger

# Loguru's default sink has no level filter, so every logger.debug(...) call
# anywhere in pipecat/our own code (there are many, per-frame) was flooding
# the console. Cap the console sink at INFO; nothing below that is lost,
# it's just not printed - use LOG_LEVEL=DEBUG env var to see everything again.
def _configure_console_log_level() -> None:
    """(Re-)apply our console log level.

    Called once at import time AND again at the top of run_bot(): pipecat's
    own CLI runner (pipecat.runner.run.main(), invoked below via
    `if __name__ == "__main__"`) does its own `logger.remove()` +
    `logger.add(sys.stderr, level="DEBUG")` internally, silently undoing
    whatever we set here before it ever calls back into run_bot(). Re-running
    this after that point is what actually makes the filter stick.
    """
    logger.remove()
    logger.add(sys.stderr, level=os.environ.get("LOG_LEVEL", "INFO"))


_configure_console_log_level()

# Silence the Pillow "'mode' parameter is deprecated" DeprecationWarning -
# it's noise, not something actionable (the actual fix is to stop passing
# mode= to Image.fromarray, done below where that call happens).
warnings.filterwarnings("ignore", category=DeprecationWarning, module="PIL")


class _InterceptHandler(logging.Handler):
    """Routes stdlib `logging` records (used by reachy_service.py and the
    reachy_mini SDK) into loguru, so they actually show up in bot output -
    without this, those logger.info/warning calls are silently dropped since
    nothing configures a stdlib logging handler."""

    def emit(self, record: logging.LogRecord) -> None:
        level = logger.level(record.levelname).name if record.levelname in logger._core.levels else record.levelno
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())


logging.basicConfig(handlers=[_InterceptHandler()], level=logging.INFO, force=True)

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame, OutputImageRawFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIObserver
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import (
    create_transport,
    get_transport_client_id,
    maybe_capture_participant_camera,
)
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
import aiohttp

from nat_vision_llm import NATVisionLLMService
from services.reachy_service import ReachyService
from services.processor import ReachyWobblerProcessor
from services.control_api import start_control_api, set_main_loop


load_dotenv(override=True)

# Started once per process: lets the NAT agent (separate process) trigger
# Reachy head movements and request a scene description over HTTP.
start_control_api()


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=True,
        video_out_enabled=True,
        video_out_is_live=True,
        video_out_width=640,
        video_out_height=480,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    ),
}

REACHY_VIDEO_WIDTH = 640
REACHY_VIDEO_HEIGHT = 480
REACHY_VIDEO_FPS = 8


async def _stream_reachy_video(task: PipelineTask):
    """Continuously pushes Reachy's camera frames to the client's video
    display, resized/letterboxed to the transport's fixed output size."""
    from PIL import Image
    import numpy as np

    interval = 1.0 / REACHY_VIDEO_FPS
    service = ReachyService.get_instance()
    try:
        while True:
            await asyncio.sleep(interval)
            if service.camera_consumer is None:
                continue
            result = service.camera_consumer.latest_frame()
            if result is None:
                continue
            _frame_id, frame = result

            image = Image.fromarray(frame)
            image.thumbnail((REACHY_VIDEO_WIDTH, REACHY_VIDEO_HEIGHT), Image.Resampling.BILINEAR)
            canvas = Image.new("RGB", (REACHY_VIDEO_WIDTH, REACHY_VIDEO_HEIGHT), (0, 0, 0))
            offset = ((REACHY_VIDEO_WIDTH - image.width) // 2, (REACHY_VIDEO_HEIGHT - image.height) // 2)
            canvas.paste(image, offset)

            await task.queue_frames([
                OutputImageRawFrame(
                    image=np.asarray(canvas).tobytes(),
                    size=(REACHY_VIDEO_WIDTH, REACHY_VIDEO_HEIGHT),
                    format="RGB",
                )
            ])
    except asyncio.CancelledError:
        pass


GESTURE_POLL_INTERVAL = 2.0  # how often to sample the camera, in seconds
GESTURE_REACTION_COOLDOWN = 30.0  # min gap between two gesture-triggered reactions
GESTURE_LABELS = ("wave", "thumbs_up", "thumbs_down", "peace_sign", "ok_sign")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "moondream")

GESTURE_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "gesture_detected": {"type": "boolean"},
        "gesture": {"type": "string", "enum": list(GESTURE_LABELS) + ["none"]},
    },
    "required": ["gesture_detected", "gesture"],
}


async def _watch_for_gestures(task: PipelineTask, messages: list):
    """Sparingly samples Reachy's camera for a small set of unambiguous
    gestures and, if one is seen, has the agent react - but only while
    nobody is talking, and at most once per GESTURE_REACTION_COOLDOWN, so it
    stays a rare, deliberate moment rather than constant background chatter.

    Runs against a local Ollama vision model (not the cloud LLM) since this
    is a cheap, narrow, high-frequency check - no point burning API tokens
    polling every couple seconds just to answer "is anyone waving right now".
    """
    from PIL import Image

    service = ReachyService.get_instance()
    last_reaction_time = 0.0

    async with aiohttp.ClientSession() as ollama_session:
        try:
            while True:
                await asyncio.sleep(GESTURE_POLL_INTERVAL)

                if service.is_busy_talking:
                    continue
                if asyncio.get_event_loop().time() - last_reaction_time < GESTURE_REACTION_COOLDOWN:
                    continue
                if service.camera_consumer is None:
                    continue
                result = service.camera_consumer.latest_frame()
                if result is None:
                    continue
                _frame_id, frame = result

                # Moondream is a small local model - it does much better with
                # a bigger/sharper crop (its own encoder expects ~378x378,
                # so downscaling to 320x240 was throwing detail away) and
                # with visually-described cues instead of bare category
                # names, which small VLMs are notably worse at grounding.
                image = Image.fromarray(frame)
                image.thumbnail((384, 384), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=80)
                image_b64 = base64.b64encode(buffer.getvalue()).decode()

                try:
                    async with ollama_session.post(
                        f"{OLLAMA_URL}/api/chat",
                        json={
                            "model": OLLAMA_VISION_MODEL,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": (
                                        "Look closely at this person's hand(s). Is one of "
                                        "these clearly happening, described visually:\n"
                                        "- wave: an open hand raised near shoulder height or "
                                        "above, fingers spread or loosely together, palm "
                                        "facing the camera\n"
                                        "- thumbs_up: a closed fist with only the thumb "
                                        "extended pointing upward\n"
                                        "- thumbs_down: a closed fist with only the thumb "
                                        "extended pointing downward\n"
                                        "- peace_sign: index and middle finger extended "
                                        "upward in a V shape, other fingers curled down\n"
                                        "- ok_sign: thumb and index finger touching to form "
                                        "a circle, other fingers extended\n"
                                        "Only answer gesture_detected=true if the hand shape "
                                        "clearly and unmistakably matches one of these "
                                        "descriptions - a hand that is merely resting, "
                                        "typing, holding something, or only partly visible "
                                        "does not count. If unsure, set gesture_detected=false "
                                        "and gesture='none'."
                                    ),
                                    "images": [image_b64],
                                }
                            ],
                            "format": GESTURE_CHECK_SCHEMA,
                            "stream": False,
                        },
                        timeout=aiohttp.ClientTimeout(total=10.0),
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                    parsed = json.loads(data["message"]["content"])
                except Exception as e:
                    logger.warning(f"Gesture check failed (Ollama): {e}")
                    continue

                if not parsed.get("gesture_detected") or parsed.get("gesture") == "none" or service.is_busy_talking:
                    continue

                gesture = parsed.get("gesture", "a gesture")
                logger.info(f"Detected unprompted gesture: {gesture}")
                last_reaction_time = asyncio.get_event_loop().time()

                messages.append(
                    {
                        "role": "system",
                        "content": (
                            f"You just noticed the user doing a '{gesture}' gesture through "
                            "your camera, completely unprompted. Briefly and naturally react "
                            "to it out loud, as if you just saw it happen live."
                        ),
                    }
                )
                await task.queue_frames([LLMRunFrame()])
        except asyncio.CancelledError:
            pass


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    set_main_loop(asyncio.get_running_loop())

    async with aiohttp.ClientSession() as session:

        stt = OpenAISTTService(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        tts = OpenAITTSService(
            api_key=os.getenv("OPENAI_API_KEY"),
            voice="alloy",
        )

        llm = NATVisionLLMService(
            api_key=os.getenv("NVIDIA_API_KEY"),
            base_url="http://localhost:8001/v1",
        )

        messages = [
            {
                "role": "system",
                "content": "Du bist der valantic Office Manager, verkörpert durch den Reachy Mini Roboter in einem WebRTC-Anruf. Antworte auf Deutsch, kurz und hilfsbereit. Deine Ausgabe wird laut vorgelesen, verwende also keine Emojis, Aufzählungszeichen oder andere Zeichen, die man nicht sprechen kann. Du hast keine Bilder direkt in dieser Nachricht - deine Kamera nutzt du gezielt über eigene Werkzeuge (reachy_see, reachy_count_people, reachy_look_around), nicht automatisch bei jeder Nachricht. Behaupte niemals, ein Bild erhalten zu haben, wenn du keines über eines dieser Werkzeuge abgerufen hast.",
            },
        ]

        context = LLMContext(messages)
        context_aggregator = LLMContextAggregatorPair(context)
        transcript = TranscriptProcessor()
        rtvi = RTVIProcessor()

        wobbler = ReachyWobblerProcessor()
        wobbler.set_tts_service(tts)

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                rtvi,  # RTVI protocol processor
                stt,  # STT
                transcript.user(),  # Capture user transcripts
                context_aggregator.user(),  # User responses
                llm,  # LLM
                tts,  # TTS
                wobbler,
                transport.output(),  # Transport bot output
                transcript.assistant(),  # Capture assistant transcripts
                context_aggregator.assistant(),  # Assistant spoken responses
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[RTVIObserver(rtvi)],
            idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        )

        video_task = None
        gesture_task = None

        @transcript.event_handler("on_transcript_update")
        async def handle_transcript_update(processor, frame):
            """Handle transcript updates and send them to the web UI"""
            for message in frame.messages:
                logger.info(f"Transcript [{message.role}]: {message.content}")

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            nonlocal video_task, gesture_task
            logger.info(f"Client connected")

            await ReachyService.get_instance().start_camera_consumer()
            video_task = asyncio.create_task(_stream_reachy_video(task))
            if os.getenv("REACHY_GESTURE_WATCH", "false").lower() == "true":
                gesture_task = asyncio.create_task(_watch_for_gestures(task, messages))

            await maybe_capture_participant_camera(transport, client)

            client_id = get_transport_client_id(transport, client)
            
            # Set the user_id for automatic image fetching
            llm.set_user_id(client_id)

            # Kick off the conversation with a fixed greeting, spoken
            # directly via TTS - bypassing the LLM entirely. A pure
            # system-message conversation (no real user turn yet) reliably
            # makes chat models "acknowledge" a kickoff instruction (eg
            # "Verstanden! ...") instead of actually saying the requested
            # line, no matter how the instruction is phrased. A fixed
            # opening line doesn't need LLM creativity anyway.
            #
            # Wait for Reachy's audio/video peer connection to actually
            # finish negotiating before speaking - start_camera_consumer()
            # returns as soon as the consumer's background tasks are
            # kicked off, not once ICE/DTLS is actually up (that takes
            # ~1-2s). Speaking immediately means this first utterance's
            # audio gets pushed to the speaker track before there's a live
            # connection to send it over, so it's silently lost.
            service = ReachyService.get_instance()
            for _ in range(30):  # up to ~3s
                if service.camera_consumer is not None and service.camera_consumer.is_connected():
                    break
                await asyncio.sleep(0.1)

            greeting = "Hallo! Ich bin Vally - der Office Manager. Wie kann ich dir helfen?"
            messages.append({"role": "assistant", "content": greeting})
            await task.queue_frames([TTSSpeakFrame(text=greeting)])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"Client disconnected")
            if video_task is not None:
                video_task.cancel()
            if gesture_task is not None:
                gesture_task.cancel()
            await ReachyService.get_instance().stop_camera_consumer()
            await task.cancel()

        runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

        await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    _configure_console_log_level()  # pipecat's runner reset this to DEBUG on startup
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()