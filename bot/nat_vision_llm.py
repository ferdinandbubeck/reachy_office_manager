"""
Custom LLM service that automatically fetches user images for vision queries.

This integrates with the NAT router to automatically capture images when
the router determines a query should go to the vision model.
"""

import asyncio
import base64
import io

import numpy as np
from typing import Optional

from loguru import logger
from PIL import Image
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    UserImageRequestFrame,
    UserImageRawFrame,
    StartInterruptionFrame,
    InputAudioRawFrame,
    UserSpeakingFrame,
    BotSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.nvidia.llm import NvidiaLLMService

from services.reachy_service import ReachyService

# Whether to force-attach a fresh camera frame to *every* user message. This
# was the original design (so the agent always felt "aware"), but now that
# the agent has explicit on-demand vision tools (reachy_see,
# reachy_count_people, reachy_look_around) it actively causes hallucinations
# in unrelated conversations - eg a restaurant/web-search question getting a
# random office camera frame attached, with the model then confidently
# describing it as "an uploaded menu photo" that never existed. Off by
# default; the agent calls a vision tool itself when it actually needs to see.
AUTO_ATTACH_IMAGE = False


class NATVisionLLMService(NvidiaLLMService):
    """
    Custom NVIDIA LLM service that automatically fetches user images.
    
    When it receives an LLMMessagesFrame, it:
    1. Automatically requests the user's camera image (once per turn)
    2. Waits for the image to arrive
    3. Manually adds the image to the context in OpenAI's multimodal format
    4. Sends the messages + image to the NAT router
    
    The NAT router will then decide whether to use the vision model or not.
    """

    def __init__(self, *args, user_id: Optional[str] = None, max_image_dimension: int = 512, image_quality: int = 70, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("NATVisionLLMService: Initialized with max_dimension=%d, quality=%d (streaming disabled for NAT)", max_image_dimension, image_quality)
        self._user_id = user_id
        self._pending_image_future: Optional[asyncio.Future] = None
        self._last_image: Optional[UserImageRawFrame] = None
        self._current_turn_has_image = False  # Track if we've fetched image for current turn
        self._max_image_dimension = max_image_dimension
        self._image_quality = image_quality
        self._pending_messages_frame: Optional[Frame] = None  # Store the frame to get context

    def set_user_id(self, user_id: str):
        """Set the user ID for image requests."""
        self._user_id = user_id
        logger.debug(f"NATVisionLLMService: User ID set to {user_id}")

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to stay within max dimension while preserving aspect ratio."""
        width, height = image.size
        max_dim = self._max_image_dimension

        if width <= max_dim and height <= max_dim:
            return image

        if width > height:
            new_width = max_dim
            new_height = int((max_dim / width) * height)
        else:
            new_height = max_dim
            new_width = int((max_dim / height) * width)

        resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        return resized

    def _encode_image_to_base64(self, frame) -> Optional[str]:
        """
        Encode a captured image to base64 JPEG data URL.

        Accepts either a UserImageRawFrame (browser camera) or a raw BGR
        numpy array (Reachy's onboard camera via reachy_mini's media manager).

        Returns:
            Data URL string like "data:image/jpeg;base64,..." or None on error
        """
        try:
            if isinstance(frame, np.ndarray):
                # RGB numpy array from Reachy's camera
                image = Image.fromarray(frame, mode='RGB')
            else:
                # Convert frame to PIL Image
                image_format = getattr(frame, 'format', 'RGB')
                image_size = getattr(frame, 'size', (640, 360))

                image = Image.frombytes(image_format, image_size, frame.image)

            # Resize if needed
            image = self._resize_image(image)

            # Convert to RGB if needed
            if image.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(
                    image,
                    mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None
                )
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')

            # Compress to JPEG
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=self._image_quality)
            image_bytes = buffer.getvalue()

            # Base64 encode
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            return f"data:image/jpeg;base64,{image_b64}"

        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return None

    def _add_image_to_context(self, context, image_data_url: str, user_message: str):
        """
        Manually add image to the last user message in context using OpenAI's multimodal format.
        
        Args:
            context: LLMContext object from the LLMMessagesFrame
            image_data_url: Data URL string like "data:image/jpeg;base64,..."
            user_message: The user's text question about the image
        """
        if not context:
            logger.warning("No context provided, cannot add image")
            return

        messages = context.messages
        if not messages:
            logger.warning("No messages in context, cannot add image")
            return

        # Find the last user message
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                # Convert content to multimodal format
                current_content = messages[i].get("content", "")
                
                # If it's already a list (multimodal), append to it
                if isinstance(current_content, list):
                    messages[i]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": image_data_url, "detail": "auto"}
                    })
                    logger.info("Appended image to existing multimodal user message")
                else:
                    # Convert string to multimodal format
                    messages[i]["content"] = [
                        {"type": "text", "text": current_content or user_message},
                        {"type": "image_url", "image_url": {"url": image_data_url, "detail": "auto"}}
                    ]
                    logger.info("Converted user message to multimodal format with image")
                
                logger.info("Image successfully added to context!")
                return

        logger.warning("No user message found in context to add image to")

    def _strip_old_images(self, messages: list):
        """Removes image_url items from every message except the last one.

        Images get baked into the persistent conversation history (the same
        message dicts are reused across turns), so without this, every old
        "what do you see" question keeps its photo attached forever. The
        model then has multiple images from different points in time in
        context and can end up describing a stale one instead of the fresh
        capture for the current question.
        """
        for msg in messages[:-1]:
            content = msg.get("content")
            if isinstance(content, list):
                text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                msg["content"] = " ".join(p for p in text_parts if p)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """
        Intercept frames to handle automatic image fetching.
        """
        # Only log significant frames to reduce noise (exclude audio, image, and speaking frames)
        # Debug, not info: this fires for every frame in the pipeline
        # (dozens per second, mostly output video/audio plumbing) - useful
        # when actively debugging frame flow, pure noise otherwise.
        logger.debug(f"NATVisionLLMService.process_frame called with {type(frame).__name__}, direction={direction}")
        
        # Reset turn state on interruption (new user input)
        if isinstance(frame, StartInterruptionFrame):
            self._current_turn_has_image = False
        
        # Capture incoming images for later use
        if isinstance(frame, UserImageRawFrame):
            self._last_image = frame
            
            # If we're waiting for an image, resolve the future
            if self._pending_image_future and not self._pending_image_future.done():
                logger.info("NATVisionLLMService: Image captured for pending request")
                self._pending_image_future.set_result(frame)
            
            # Don't pass the raw image frame downstream
            return

        # For LLMContextFrame (not LLMMessagesFrame!), fetch image first (only once per turn)
        if isinstance(frame, LLMContextFrame):
            # Check if there's a user message in the context
            has_user_message = any(msg.get("role") == "user" for msg in frame.context.messages)
            
            if not has_user_message:
                # No user message yet, skip image fetch
                logger.debug("NATVisionLLMService: No user message in context, skipping image fetch")
                await super().process_frame(frame, direction)
                return
            
            if AUTO_ATTACH_IMAGE and not self._current_turn_has_image:
                logger.info("NATVisionLLMService: Intercepting LLMMessagesFrame to add image")

                # Drop images from earlier turns so only the freshest capture
                # is ever in context (see _strip_old_images docstring).
                self._strip_old_images(frame.context.messages)

                # Fetch the image first
                await self._fetch_and_wait_for_image(frame)
                
                # Now add it to this frame's context
                if self._last_image is not None:
                    logger.info("NATVisionLLMService: Encoding and adding image to context")
                    image_data_url = self._encode_image_to_base64(self._last_image)
                    
                    if image_data_url:
                        context = getattr(frame, 'context', None)
                        messages = frame.context.messages if context else []
                        
                        # Extract user message text
                        user_message = ""
                        for msg in reversed(messages):
                            if msg.get("role") == "user":
                                content = msg.get("content", "")
                                user_message = content if isinstance(content, str) else ""
                                break
                        
                        self._add_image_to_context(context, image_data_url, user_message)
                        self._current_turn_has_image = True
                        
                        # Log what we're sending to NAT
                        logger.info(f"NATVisionLLMService: Frame has {len(messages)} messages after adding image")
                        for i, msg in enumerate(messages):
                            role = msg.get("role", "unknown")
                            content = msg.get("content", "")
                            if isinstance(content, list):
                                logger.info(f"  Message {i} ({role}): multimodal with {len(content)} items")
                                for j, item in enumerate(content):
                                    item_type = item.get("type", "unknown")
                                    if item_type == "image_url":
                                        url = item.get("image_url", {}).get("url", "")
                                        logger.info(f"    Item {j}: {item_type}, URL length: {len(url)}")
                                    else:
                                        logger.info(f"    Item {j}: {item_type}")
                            else:
                                logger.info(f"  Message {i} ({role}): text only, length {len(content) if content else 0}")
                else:
                    logger.warning("NATVisionLLMService: No image received, sending frame without image")
            elif not AUTO_ATTACH_IMAGE:
                logger.debug("NATVisionLLMService: Auto-attach disabled, agent must call reachy_see/reachy_count_people/reachy_look_around itself")
            else:
                logger.debug("NATVisionLLMService: Already have image this turn, skipping fetch")

        # Continue with normal processing
        await super().process_frame(frame, direction)

    async def _fetch_and_wait_for_image(self, context_frame: LLMContextFrame):
        """
        Request an image and wait for it to arrive.

        Prefers Reachy's onboard camera when connected; falls back to
        requesting a frame from the browser participant otherwise.
        """
        reachy_frame = await ReachyService.get_instance().capture_fresh_frame()
        if reachy_frame is not None:
            logger.info("NATVisionLLMService: Using image from Reachy's onboard camera")
            self._last_image = reachy_frame
            return

        if not self._user_id:
            logger.warning("NATVisionLLMService: No user_id set, cannot fetch image")
            return

        # Extract the user's question from the messages
        user_message = ""
        for msg in reversed(context_frame.context.messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_message = content
                    break

        # Create a future to wait for the image
        self._pending_image_future = asyncio.Future()

        # Request the image (don't use append_to_context since we're adding manually)
        await self.push_frame(
            UserImageRequestFrame(
                user_id=self._user_id,
                text=user_message,
                append_to_context=False,  # We'll add it manually
            ),
            FrameDirection.UPSTREAM,
        )

        # Wait for the image with a timeout
        try:
            await asyncio.wait_for(self._pending_image_future, timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning("NATVisionLLMService: Timeout waiting for user image")
        finally:
            self._pending_image_future = None

