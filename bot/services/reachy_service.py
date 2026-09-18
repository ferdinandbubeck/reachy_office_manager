import threading
import logging
import math
import time
from reachy_mini import ReachyMini
from .moves import MovementManager
from .wobbler import HeadWobbler
from .dance_emotion_moves import GotoQueueMove, DanceQueueMove, EmotionQueueMove
from reachy_mini.utils import create_head_pose

EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"

logger = logging.getLogger(__name__)

MOVE_DURATION = 1.0  # seconds for a 90 degree look_at/goto_body_yaw turn
MIN_MOVE_DURATION = 0.5  # floor, so tiny moves (eg pure head tilt) aren't instant


def _duration_for_turn(delta_deg: float) -> float:
    """Scales move duration with how far the body actually turns, so a big
    left-to-right sweep (~180 deg) doesn't snap at the same speed as a small
    90 deg turn."""
    return max(MIN_MOVE_DURATION, MOVE_DURATION * abs(delta_deg) / 90.0)


# Hard, code-level rate limit for emotions/gestures. The LLM agent decides
# *whether* to express something, but it can't be trusted to reliably pace
# itself via prompt instructions alone - this guarantees a minimum quiet gap
# between expressive moves regardless of what the agent asks for, so they
# read as distinct, deliberate moments rather than constant motion.
EXPRESSIVE_MOVE_COOLDOWN = 45.0  # seconds

class ReachyService:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, host='localhost'):
        self.robot = None
        self.motion_manager = None
        self.wobbler = None
        self.host = host
        self.connected = False
        self.camera_consumer = None
        self.current_body_yaw_rad = 0.0
        self.current_head_yaw_rad = 0.0
        self.current_head_pitch_rad = 0.0
        self._recorded_moves = None  # lazy-loaded RecordedMoves (emotions)
        self._last_expressive_move_time = 0.0  # monotonic clock, for the cooldown gate
        self.face_tracking_enabled = False
        self.is_busy_talking = False  # True while either the bot or the user is speaking
        self._last_frame_id = None
        self._last_frame_seen_at = 0.0

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = ReachyService()
        return cls._instance

    def connect(self):
        # If already connected, return
        if self.connected:
            logger.debug("Reachy already connected")
            return
            
        # If previously disconnected, clean up any leftover state
        if self.robot or self.motion_manager or self.wobbler:
            logger.info("Cleaning up previous Reachy connection...")
            self.disconnect()
            
        try:
            import os

            use_sim = os.getenv('REACHY_USE_SIM', 'false').lower() == 'true'
            robot_name = os.getenv('REACHY_ROBOT_NAME', 'reachy_mini')
            host = os.getenv('REACHY_HOST', 'reachy-mini.local')

            if use_sim:
                import time

                # Verify DISPLAY is set
                display = os.getenv('DISPLAY')
                logger.info(f"DISPLAY environment variable: {display}")

                # Give Xvfb extra time to stabilize
                logger.info("Waiting for display to be ready...")
                time.sleep(3)

            logger.info(f"Connecting to Reachy Mini (use_sim={use_sim}, robot_name={robot_name}, host={host})...")

            # 'no_media' by default: camera/mic come from the separate
            # ReachyCentralConsumer (via HF's cloud relay - see
            # start_camera_consumer), not from this SDK connection. The
            # 'default'/'webrtc' media backend tries a *direct LAN* WebRTC
            # signalling connection (port 8443) first, which reliably times
            # out on networks that don't forward it - and since that happens
            # inside the same ReachyMini() constructor, it was taking the
            # whole connection (including motor control) down with it.
            media_backend = os.getenv('REACHY_MEDIA_BACKEND', 'no_media')

            self.robot = ReachyMini(
                robot_name=robot_name,
                host=host,
                connection_mode='localhost_only' if use_sim else 'network',
                use_sim=use_sim,
                spawn_daemon=False,
                timeout=15.0,          # Increased timeout
                log_level='DEBUG',
                media_backend=media_backend,
            )
            logger.info("Successfully connected to Reachy Mini daemon")

            # media_backend='no_media' above releases the daemon's local media
            # hardware so our own SDK connection doesn't try to open a WebRTC
            # video session over the network (that's what was hanging the
            # whole connect() including motor control). But the daemon's
            # camera->HF-central-relay bridge (used by ReachyCentralConsumer
            # for the live camera feed) also depends on that local media
            # pipeline being acquired - if we leave it released, the robot's
            # own relay client can't reach its local media server and the
            # central feed silently never gets frames. Re-acquire it directly
            # via the daemon HTTP API (bypassing our SDK's media_backend,
            # which would just release it again since it's still 'no_media').
            if media_backend == 'no_media':
                try:
                    import httpx

                    httpx.post(f"http://{host}:8000/api/media/acquire", timeout=10.0).raise_for_status()
                    logger.info("Re-acquired media on daemon (for central camera relay).")
                except Exception as e:
                    logger.warning(f"Failed to re-acquire media on daemon: {e}")

            # Wake the robot up: enable motors and play the wake-up animation.
            # The platform puts the robot to sleep/limp whenever no app is
            # holding a session, so we must explicitly wake it on connect.
            try:
                self.robot.enable_motors()
                self.robot.wake_up()
                logger.info("Reachy woken up and motors enabled.")
            except Exception as e:
                logger.warning(f"Failed to wake up Reachy: {e}")

            # 1. Initialize Motor Cortex (Background Thread)
            self.motion_manager = MovementManager(self.robot)
            self.motion_manager.start() 
            
            # 2. Initialize Auditory Cortex (Links Audio -> Motion)
            self.wobbler = HeadWobbler(self.motion_manager.set_speech_offsets)
            self.wobbler.start()
            
            self.connected = True
            logger.info("Reachy Service Started: Breathing & Sway active.")
        except Exception as e:
            import traceback
            logger.warning(f"Reachy Mini daemon not available: {e}")
            logger.warning(f"Full traceback: {traceback.format_exc()}")
            logger.warning("Pipeline will continue without Reachy robot control.")
            logger.warning("To enable Reachy: start daemon with 'mjpython -m reachy_mini.daemon.app.main --sim --no-localhost-only'")
            
            # Clean up partial robot object to avoid destructor errors
            self.robot = None
            # Don't raise - allow pipeline to run without Reachy

    async def start_camera_consumer(self):
        """Starts a ReachyCentralConsumer to stream Reachy's camera over
        Hugging Face's central WebRTC relay. This works even though the bot runs
        on a different machine than the robot's daemon (no direct LAN WebRTC needed).

        Video-only: audio in/out goes through the PC's own mic/speaker
        (the browser tab), not Reachy's - see main.py's transport_params.
        """
        if self.camera_consumer is not None:
            return

        import os
        from reachy_mini.media.central_consumer import ReachyCentralConsumer

        hf_token = os.getenv("HF_TOKEN")
        robot_name = os.getenv("REACHY_ROBOT_NAME", "reachy_mini")

        if not hf_token:
            logger.warning("HF_TOKEN not set, cannot start Reachy camera consumer.")
            return

        try:
            consumer = ReachyCentralConsumer(robot_name=robot_name, hf_token=hf_token)
            await consumer.start()
            self.camera_consumer = consumer
            logger.info("Reachy camera consumer started (central WebRTC relay).")
        except Exception as e:
            logger.warning(f"Failed to start Reachy camera consumer: {e}")
            self.camera_consumer = None

    async def stop_camera_consumer(self):
        """Stops the ReachyCentralConsumer, if running."""
        if self.camera_consumer is not None:
            try:
                await self.camera_consumer.stop()
            except Exception as e:
                logger.warning(f"Error stopping Reachy camera consumer: {e}")
            self.camera_consumer = None

    async def capture_fresh_frame(self, timeout: float = 6.0):
        """Returns a recent RGB frame from Reachy's camera, using the single
        persistent central-relay session for the whole connection.

        Opening/closing a session per request would repeatedly release the
        daemon's remote-session lock, which makes it reset the robot to a
        "clean idle" state (motors off, head down) after every vision
        question - so we deliberately keep ONE session alive for as long as
        the bot is connected, and only restart it if it has actually gone
        stale (no new frame for a while).
        """
        import asyncio
        import time

        if self.camera_consumer is None:
            await self.start_camera_consumer()
        if self.camera_consumer is None:
            return None

        now = time.monotonic()
        result = self.camera_consumer.latest_frame()

        stale = (
            result is not None
            and result[0] == self._last_frame_id
            and (now - self._last_frame_seen_at) > 8.0
        )
        if result is None or stale:
            logger.info("Reachy camera frame missing/stale, restarting session...")
            await self.stop_camera_consumer()
            await self.start_camera_consumer()
            if self.camera_consumer is None:
                return None
            deadline = time.monotonic() + timeout
            result = None
            while time.monotonic() < deadline:
                result = self.camera_consumer.latest_frame()
                if result is not None:
                    break
                await asyncio.sleep(0.2)

        if result is None:
            logger.warning("Timed out waiting for a frame from Reachy's camera.")
            return None

        frame_id, frame = result
        if frame_id != self._last_frame_id:
            self._last_frame_id = frame_id
            self._last_frame_seen_at = time.monotonic()
        return frame

    def get_camera_frame(self):
        """Synchronous fallback: grabs a frame from a LAN-local media manager,
        if connected that way. Prefer capture_fresh_frame() when possible.
        """
        if self.connected and self.robot:
            try:
                bgr_frame = self.robot.media.get_frame()
                if bgr_frame is not None:
                    return bgr_frame[:, :, ::-1]  # BGR -> RGB
            except Exception as e:
                logger.warning(f"Failed to get camera frame from Reachy: {e}")

        return None

    def feed_audio(self, audio_chunk_base64):
        """Feeds audio from TTS to the wobble engine."""
        if self.wobbler:
            logger.debug("Feeding audio to Reachy")  # fires per audio chunk, ~every 10-20ms
            self.wobbler.feed(audio_chunk_base64)
    
    def set_listening_pose(self):
        """Sets robot back to listening/idle pose."""
        if self.motion_manager:
            self.motion_manager.set_listening(True)

    def start_wobbling(self):
        """Marks the start of a clean speech-driven wobble: exits listening
        mode and resets the wobbler so no state from a previous utterance
        carries over."""
        if self.motion_manager:
            self.motion_manager.set_listening(False)
        if self.wobbler:
            self.wobbler.reset()

    def stop_wobbling(self):
        """Cleanly stops the speech-driven head wobble: zeroes the offset
        (instead of freezing at whatever it was on the last audio chunk) and
        resets the wobbler so start_wobbling()'s next utterance begins fresh.

        Does NOT freeze antennas (see freeze_antennas_for_listening) - the
        bot finishing a sentence shouldn't stop idle breathing sway or an
        in-flight/upcoming turn's antenna flick from animating.
        """
        if self.wobbler:
            self.wobbler.reset()
        if self.motion_manager:
            self.motion_manager.set_speech_offsets((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    def freeze_antennas_for_listening(self):
        """Freezes antennas while actively capturing the user's voice, so
        antenna servo motion/noise doesn't interfere with the mic. Call
        unfreeze_antennas() as soon as the user stops talking."""
        self.set_listening_pose()

    def unfreeze_antennas(self):
        """Releases the antenna freeze set by freeze_antennas_for_listening(),
        letting idle breathing sway / turn flicks / etc animate again."""
        if self.motion_manager:
            self.motion_manager.set_listening(False)

    def set_face_tracking(self, enabled: bool) -> bool:
        """Enables/disables the daemon-side visual head tracking (the robot's
        head follows a detected face). Returns True if the call succeeded."""
        if not self.connected or not self.robot:
            logger.debug(f"Reachy not connected - ignoring set_face_tracking({enabled})")
            return False
        try:
            if enabled:
                # Stop our own 100Hz breathing/idle loop from fighting the
                # daemon's tracking control loop for the same motors before
                # handing control over - see MovementManager.set_external_control.
                if self.motion_manager:
                    self.motion_manager.set_external_control(True)
                try:
                    self.robot.start_head_tracking()
                except Exception:
                    if self.motion_manager:
                        self.motion_manager.set_external_control(False)
                    raise
            else:
                self.robot.stop_head_tracking()
                if self.motion_manager:
                    self.motion_manager.set_external_control(False)
            self.face_tracking_enabled = enabled
            logger.info(f"Reachy face tracking {'enabled' if enabled else 'disabled'}")
            return True
        except Exception as e:
            logger.error(f"set_face_tracking({enabled}) failed: {e}")
            return False

    def _yield_face_tracking(self) -> None:
        """Hand control back from daemon-side face tracking before issuing a
        manual move - while tracking owns the head (see set_face_tracking),
        our own move queue is paused and would silently no-op otherwise."""
        if self.face_tracking_enabled:
            self.set_face_tracking(False)

    def look_at(self, direction: str) -> float:
        """Maps semantic direction to robot pose. Returns the move's duration
        in seconds (0 if the move couldn't be issued), so callers know how
        long to wait before the robot has actually settled."""
        if not self.connected or not self.motion_manager or not self.robot:
            logger.debug(f"Reachy not connected - ignoring look_at({direction})")
            return 0.0
        self._yield_face_tracking()

        # Fixed, known positions rather than incremental stepping: each
        # direction always jumps straight to the same absolute pose, so the
        # robot's "positions" are predictable and repeatable. Head yaw is
        # commanded in world-frame absolute terms, so it must match the body
        # yaw target 1:1 or the head visibly lags behind the body.
        POSITIONS_DEG = {
            "front": {"body_yaw": 0, "pitch": 0},
            "left": {"body_yaw": 90, "pitch": 0},
            "right": {"body_yaw": -90, "pitch": 0},
            "up": {"body_yaw": None, "pitch": -15},    # None: keep current yaw
            "down": {"body_yaw": None, "pitch": 15},
        }

        pos = POSITIONS_DEG.get(direction, POSITIONS_DEG["front"])
        current_body_yaw_deg = math.degrees(self.current_body_yaw_rad)
        target_body_yaw_deg = current_body_yaw_deg if pos["body_yaw"] is None else pos["body_yaw"]
        target_head_pitch_deg = pos["pitch"]

        target_head_yaw_deg = target_body_yaw_deg
        target_body_yaw = math.radians(target_body_yaw_deg)
        duration = _duration_for_turn(target_body_yaw_deg - current_body_yaw_deg)

        try:
            target_pose = create_head_pose(0, 0, 0, 0, target_head_pitch_deg, target_head_yaw_deg, degrees=True)
            current_head_pose = self.robot.get_current_head_pose()
            _, current_antennas = self.robot.get_current_joint_positions()

            goto_move = GotoQueueMove(
                target_head_pose=target_pose,
                start_head_pose=current_head_pose,
                target_antennas=(0, 0),
                start_antennas=(current_antennas[0], current_antennas[1]),
                target_body_yaw=target_body_yaw,
                start_body_yaw=self.current_body_yaw_rad,
                duration=duration
            )
            self.current_body_yaw_rad = target_body_yaw
            self.current_head_yaw_rad = math.radians(target_head_yaw_deg)
            self.current_head_pitch_rad = math.radians(target_head_pitch_deg)
            self.motion_manager.queue_move(goto_move)
            self.motion_manager.set_moving_state(1.0)
            logger.info(f"Reachy looking {direction}")
            return duration
        except Exception as e:
            logger.error(f"Look at failed: {e}")
            return 0.0

    def goto_body_yaw(self, body_yaw_deg: float, head_yaw_deg: float = 0.0, duration: float | None = None) -> float:
        """Turns to an absolute body yaw angle (degrees). Used for the 'around'
        scan sequence, which needs explicit targets rather than look_at's
        incremental left/right stepping. Returns the duration used.
        """
        if not self.connected or not self.motion_manager or not self.robot:
            logger.debug(f"Reachy not connected - ignoring goto_body_yaw({body_yaw_deg})")
            return 0.0
        self._yield_face_tracking()

        if duration is None:
            duration = _duration_for_turn(body_yaw_deg - math.degrees(self.current_body_yaw_rad))

        target_body_yaw = math.radians(body_yaw_deg)
        try:
            target_pose = create_head_pose(0, 0, 0, 0, 0, head_yaw_deg, degrees=True)
            current_head_pose = self.robot.get_current_head_pose()
            _, current_antennas = self.robot.get_current_joint_positions()

            goto_move = GotoQueueMove(
                target_head_pose=target_pose,
                start_head_pose=current_head_pose,
                target_antennas=(0, 0),
                start_antennas=(current_antennas[0], current_antennas[1]),
                target_body_yaw=target_body_yaw,
                start_body_yaw=self.current_body_yaw_rad,
                duration=duration,
            )
            self.current_body_yaw_rad = target_body_yaw
            self.motion_manager.queue_move(goto_move)
            self.motion_manager.set_moving_state(1.0)
            logger.info(f"Reachy turning to body_yaw={body_yaw_deg}deg")
            return duration
        except Exception as e:
            logger.error(f"goto_body_yaw failed: {e}")
            return 0.0

    RESTORE_DURATION = 0.6

    def _queue_restore_pose(self) -> float:
        """Queues a quick move back to the tracked look_at position (always
        level - roll=0).

        Emotions/gestures play their own recorded head+body trajectory
        (usually assuming a forward-facing start, and some - eg
        groovy_sway_and_roll - deliberately tilt/roll the head as part of
        the choreography), so without this the robot would silently "forget"
        it was turned left/right/etc, or stay tilted, as soon as one finishes
        playing. Always queued (even when already tracking the front/neutral
        look direction) since roll is never tracked/checked here - only this
        move actually levels it back out. Called right after queuing an
        emotion/gesture so it runs immediately once that move ends.
        """
        try:
            target_pose = create_head_pose(
                0, 0, 0, 0,
                math.degrees(self.current_head_pitch_rad),
                math.degrees(self.current_head_yaw_rad),
                degrees=True,
            )
            current_head_pose = self.robot.get_current_head_pose()
            _, current_antennas = self.robot.get_current_joint_positions()
            restore_move = GotoQueueMove(
                target_head_pose=target_pose,
                start_head_pose=current_head_pose,
                target_antennas=(0, 0),
                start_antennas=(current_antennas[0], current_antennas[1]),
                target_body_yaw=self.current_body_yaw_rad,
                start_body_yaw=self.current_body_yaw_rad,
                duration=self.RESTORE_DURATION,
            )
            self.motion_manager.queue_move(restore_move)
            return self.RESTORE_DURATION
        except Exception as e:
            logger.warning(f"Failed to queue pose restore: {e}")
            return 0.0

    def _check_expressive_cooldown(self) -> bool:
        """Returns True if an emotion/gesture may play right now. Does not
        consume the cooldown by itself - call _mark_expressive_move() once
        the move is actually queued."""
        return (time.monotonic() - self._last_expressive_move_time) >= EXPRESSIVE_MOVE_COOLDOWN

    def _mark_expressive_move(self) -> None:
        self._last_expressive_move_time = time.monotonic()

    def play_gesture(self, move_name: str) -> float:
        """Plays a dance/gesture move from reachy_mini_dances_library by name,
        then restores the previously held look_at position. Returns the total
        duration in seconds, 0.0 if it couldn't be played, or -1.0 if skipped
        because another expressive move played too recently (cooldown)."""
        if not self.connected or not self.motion_manager:
            logger.debug(f"Reachy not connected - ignoring play_gesture({move_name})")
            return 0.0
        if not self._check_expressive_cooldown():
            logger.info(f"play_gesture({move_name!r}) skipped - still on cooldown")
            return -1.0
        self._yield_face_tracking()
        try:
            move = DanceQueueMove(move_name)
            self.motion_manager.queue_move(move)
            duration = move.duration
            duration += self._queue_restore_pose()
            self.motion_manager.set_moving_state(duration)
            self._mark_expressive_move()
            logger.info(f"Reachy playing gesture '{move_name}' ({duration:.1f}s)")
            return duration
        except Exception as e:
            logger.error(f"play_gesture({move_name!r}) failed: {e}")
            return 0.0

    def play_emotion(self, emotion_name: str) -> float:
        """Plays a recorded emotion move (from the Pollen emotions-library HF
        dataset) by name, then restores the previously held look_at position.
        Returns the total duration in seconds, 0.0 if it couldn't be played,
        or -1.0 if skipped because another expressive move played too
        recently (cooldown). The dataset is downloaded and cached on first
        use, which can take a while."""
        if not self.connected or not self.motion_manager:
            logger.debug(f"Reachy not connected - ignoring play_emotion({emotion_name})")
            return 0.0
        if not self._check_expressive_cooldown():
            logger.info(f"play_emotion({emotion_name!r}) skipped - still on cooldown")
            return -1.0
        self._yield_face_tracking()
        try:
            if self._recorded_moves is None:
                from reachy_mini.motion.recorded_move import RecordedMoves
                self._recorded_moves = RecordedMoves(EMOTIONS_DATASET)

            move = EmotionQueueMove(emotion_name, self._recorded_moves)
            self.motion_manager.queue_move(move)
            duration = move.duration
            duration += self._queue_restore_pose()
            self.motion_manager.set_moving_state(duration)
            self._mark_expressive_move()
            logger.info(f"Reachy playing emotion '{emotion_name}' ({duration:.1f}s)")
            return duration
        except Exception as e:
            logger.error(f"play_emotion({emotion_name!r}) failed: {e}")
            return 0.0

    def disconnect(self):
        """Disconnect and cleanup Reachy resources."""
        if not self.connected:
            return
            
        logger.info("Disconnecting Reachy service...")
        
        # Stop background threads
        if self.motion_manager:
            self.motion_manager.stop()
        if self.wobbler:
            self.wobbler.stop()
        
        # Disconnect robot
        if self.robot:
            try:
                # The robot client should disconnect gracefully
                if hasattr(self.robot, 'client') and self.robot.client:
                    self.robot.client.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting robot: {e}")
        
        # Reset state
        self.robot = None
        self.motion_manager = None
        self.wobbler = None
        self.connected = False
        
        logger.info("Reachy service disconnected")
    
    def stop(self):
        """Alias for disconnect for backwards compatibility."""
        self.disconnect()
