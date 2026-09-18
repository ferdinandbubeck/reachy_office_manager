import asyncio
import base64
import hashlib
import random
from pipecat.frames.frames import TTSSpeakFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from .reachy_service import ReachyService

# Spoken while the agent's own tool-calling is slow (eg reachy_look_around,
# an office/MCP lookup, or a vision call) - a technical fix, not a prompt
# one: NAT hides all tool-call timing behind one blocking HTTP response, so
# there's no frame-level signal for "a tool is running" to hook into.
# Instead we just race a timer against the real reply: if nothing has
# started coming back within FILLER_DELAY_SECONDS of the user finishing
# speaking, say one of these; if the real reply wins the race, the filler
# never fires.
FILLER_PHRASES = (
    "Einen Moment, ich schaue nach.",
    "Kurz, ich sehe mal nach.",
    "Moment, ich prüfe das kurz.",
)
FILLER_DELAY_SECONDS = 2.5

# Only arm the filler timer for requests that plausibly hit a slow tool
# (office/MCP lookups or vision calls) - a plain "turn left"/"dance" is
# fast enough on its own that the filler was firing needlessly and just
# sounded like unnecessary chatter. Simple keyword allowlist on the
# transcribed text, not the system prompt - deliberately conservative
# (opt-in), so anything not matching stays filler-free by default.
FILLER_TRIGGER_KEYWORDS = (
    # office / MCP
    "büro", "buero", "platz", "schreibtisch", "meeting", "raum", "räume",
    "buchen", "gebucht", "termin", "frei", "besetzt", "office", "desk",
    # vision / look-around
    "sieh", "siehst", "schau dich um", "umsehen", "umschauen", "umschau",
    "beschreib", "erkenn", "person", "leute", "kamera",
)


class ReachyWobblerProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.service = ReachyService.get_instance()
        # Attempt connection on initialization
        if not self.service.connected:
            self.service.connect()

        # Track bot speaking state
        self.bot_is_speaking = False
        # Track seen audio frames to avoid duplicates
        self.seen_audio_hashes = set()
        # Clear hash set periodically to prevent memory growth
        self.frame_count = 0
        self.hash_clear_interval = 1000
        self._filler_task: asyncio.Task | None = None
        self._tts = None  # set via set_tts_service() once created
        self._last_user_text = ""

    def set_tts_service(self, tts):
        """The TTS service instance, needed to inject a filler
        TTSSpeakFrame directly into its own queue via queue_frame().

        Deliberately NOT done via task.queue_frames() (pipeline source) or
        push_frame from this processor's position (after TTS, which would
        skip synthesis entirely) - task.queue_frames() was tried first and
        made the filler arrive *after* the real (slow) reply instead of
        before it: a frame queued at the pipeline source still has to pass
        through every upstream processor in order, including the LLM stage,
        which is exactly the one busy/blocked on the slow NAT call. Queuing
        directly on the TTS service's own queue skips that queue entirely."""
        self._tts = tts

    def reset_state(self):
        """Reset processor state (called on disconnect)."""
        self.bot_is_speaking = False
        self.seen_audio_hashes.clear()
        self.frame_count = 0
        self._cancel_filler_timer()

    def _cancel_filler_timer(self):
        if self._filler_task is not None:
            self._filler_task.cancel()
            self._filler_task = None

    async def _speak_filler_after_delay(self):
        try:
            await asyncio.sleep(FILLER_DELAY_SECONDS)
            if self._tts is not None:
                await self._tts.queue_frame(TTSSpeakFrame(text=random.choice(FILLER_PHRASES)), FrameDirection.DOWNSTREAM)
        except asyncio.CancelledError:
            pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Track bot speaking state. Keyed off TTSStartedFrame/TTSStoppedFrame
        # (emitted by the TTS service itself) rather than
        # BotStartedSpeakingFrame/BotStoppedSpeakingFrame (emitted by the
        # transport's own audio-send loop) - the latter never fire at all
        # when audio_out_enabled=False (browser speaker off, Reachy speaker
        # only), which silently broke wobbling and the Reachy-speaker feed
        # entirely.
        if isinstance(frame, TranscriptionFrame):
            self._last_user_text = (frame.text or "").lower()

        elif isinstance(frame, TTSStartedFrame):
            # The real (or filler) reply has started - either way, the race
            # is over, so stop the filler timer if it's still pending.
            self._cancel_filler_timer()
            self.bot_is_speaking = True
            self.service.is_busy_talking = True
            self.seen_audio_hashes.clear()  # Clear hashes when new speech starts
            self.service.start_wobbling()

        elif isinstance(frame, TTSStoppedFrame):
            self.bot_is_speaking = False
            self.service.is_busy_talking = False
            self.service.stop_wobbling()
            self.seen_audio_hashes.clear()

        elif isinstance(frame, UserStartedSpeakingFrame):
            self._cancel_filler_timer()
            self.bot_is_speaking = False
            self.service.is_busy_talking = True
            self.service.stop_wobbling()
            self.service.freeze_antennas_for_listening()
            self.seen_audio_hashes.clear()

        elif isinstance(frame, UserStoppedSpeakingFrame):
            self.service.is_busy_talking = False
            self.service.unfreeze_antennas()
            # Only start the filler race if this utterance plausibly needs
            # a slow tool (office/MCP or vision) - a plain movement/gesture
            # command resolves fast enough on its own that a filler here
            # would just be unnecessary chatter.
            self._cancel_filler_timer()
            if any(kw in self._last_user_text for kw in FILLER_TRIGGER_KEYWORDS):
                self._filler_task = asyncio.create_task(self._speak_filler_after_delay())
        
        # Only feed audio if bot is actively speaking
        elif isinstance(frame, AudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            if self.bot_is_speaking:
                # Create hash of audio data to detect duplicates
                audio_hash = hashlib.md5(frame.audio).hexdigest()
                
                if audio_hash not in self.seen_audio_hashes:
                    # Mark as seen
                    self.seen_audio_hashes.add(audio_hash)
                    
                    # Feed to wobbler
                    b64_audio = base64.b64encode(frame.audio).decode('utf-8')

                    self.service.feed_audio(b64_audio)
                    
                    # Periodically clear hash set to prevent unbounded growth
                    self.frame_count += 1
                    if self.frame_count >= self.hash_clear_interval:
                        # Keep only the most recent hashes
                        if len(self.seen_audio_hashes) > 100:
                            self.seen_audio_hashes = set(list(self.seen_audio_hashes)[-100:])
                        self.frame_count = 0

        await self.push_frame(frame, direction)
