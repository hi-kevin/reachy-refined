"""Bidirectional local audio stream using Reachy Mini SDK.

Adapted from reachy_mini_conversation_app/console.py to be backend-agnostic.
Connects Reachy Mini's audio I/O (via SDK) to an async handler (Cognitive Brain).
"""

import time
import asyncio
import logging
import numpy as np
import collections
from typing import Any, List, Optional, Tuple, Protocol, TYPE_CHECKING
from scipy.signal import resample

from reachy_mini import ReachyMini
from reachy_mini.media.media_manager import MediaBackend

logger = logging.getLogger(__name__)

# Type definitions for audio data
AudioFrame = Tuple[int, np.ndarray]  # (sample_rate, audio_data)

class AudioHandler(Protocol):
    """Protocol for the brain component that handles audio streams."""
    async def receive(self, frame: AudioFrame) -> None: ...
    async def emit(self) -> Optional[AudioFrame]: ...
    async def shutdown(self) -> None: ...

def audio_to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert audio to float32."""
    if audio.dtype == np.float32:
        return audio
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32)

class LocalStream:
    """LocalStream using Reachy Mini's recorder/player."""

    def __init__(
        self,
        handler: AudioHandler,
        robot: ReachyMini,
        face_watcher: Optional[Any] = None,
    ):
        """Initialize the stream with a handler and robot instance.

        Args:
            handler: The brain component that consumes/produces audio.
            robot: ReachyMini SDK instance.
            face_watcher: Optional FaceWatcher whose face_present property gates
                          the mic → Gemini audio path. When None, audio is always
                          forwarded (original behaviour).
        """
        self.handler = handler
        self._robot = robot
        self._face_watcher = face_watcher
        self._playback_end_time = 0.0
        self._stop_event = asyncio.Event()
        self._tasks: List[asyncio.Task[None]] = []

    def launch(self) -> None:
        """Start the recorder/player and run the async processing loops."""
        self._stop_event.clear()

        # Start media pipelines
        logger.info("Starting Reachy Mini media...")
        self._robot.media.start_recording()
        self._robot.media.start_playing()
        
        # Create async tasks
        loop = asyncio.get_event_loop()
        self._tasks = [
            loop.create_task(self.record_loop(), name="stream-record-loop"),
            loop.create_task(self.play_loop(), name="stream-play-loop"),
        ]
        
        logger.info("LocalStream launched.")

    def close(self) -> None:
        """Stop the stream and underlying media pipelines."""
        logger.info("Stopping LocalStream...")
        
        # Stop media first
        try:
            self._robot.media.stop_recording()
        except Exception:
            pass
        try:
            self._robot.media.stop_playing()
        except Exception:
            pass

        # Signal loops to stop
        self._stop_event.set()
        
        # Cancel tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

    async def record_loop(self) -> None:
        """Read mic frames from the recorder and forward them to the handler.

        When a face_watcher is configured, audio is only forwarded to Gemini
        while face_present is True. The mic hardware keeps running regardless
        so we don't miss the first frames when a face appears.
        """
        input_sample_rate = self._robot.media.get_input_audio_samplerate()
        logger.debug(f"Audio recording started at {input_sample_rate} Hz")

        has_logged_rate = False
        _mic_active_logged = False

        # Periodic audio stats (once per second instead of per-frame)
        _stats_time = time.time()
        _frame_count = 0
        _rms_sum = 0.0
        _rms_max = 0.0

        while not self._stop_event.is_set():
            if not has_logged_rate:
                logger.info(f"Actual Input Sample Rate: {input_sample_rate}")
                has_logged_rate = True

            # Gate: only send audio to Gemini when a face is present
            face_active = (
                self._face_watcher is None or self._face_watcher.face_present
            )

            if face_active and not _mic_active_logged:
                logger.info("[MIC START] Face present — begin forwarding audio to Gemini.")
                _mic_active_logged = True
            elif not face_active and _mic_active_logged:
                logger.info("[MIC PAUSED] No face — audio gated (mic hardware still running).")
                _mic_active_logged = False

            # Get audio sample from SDK (always drain the buffer)
            audio_frame = self._robot.media.get_audio_sample()

            if audio_frame is not None and len(audio_frame) > 0 and face_active:
                rms = np.sqrt(np.mean(audio_frame**2))
                _frame_count += 1
                _rms_sum += rms
                _rms_max = max(_rms_max, rms)
                await self.handler.receive((input_sample_rate, audio_frame))

            # Log summary once per 5 seconds (only when active)
            now = time.time()
            if now - _stats_time >= 5.0 and _frame_count > 0:
                avg_rms = _rms_sum / _frame_count
                logger.info(f"[MIC ACTIVE] frames={_frame_count}, avg_rms={avg_rms:.4f}, max_rms={_rms_max:.4f}")
                _frame_count = 0
                _rms_sum = 0.0
                _rms_max = 0.0
                _stats_time = now

            await asyncio.sleep(0.01)  # Yield to avoid busy loop

    async def play_loop(self) -> None:
        """Fetch outputs from the handler and play audio frames."""
        output_sample_rate = self._robot.media.get_output_audio_samplerate()
        logger.debug(f"Audio playback started at {output_sample_rate} Hz")

        while not self._stop_event.is_set():
            # Get audio from brain
            handler_output = await self.handler.emit()

            if handler_output:
                sr, audio_data = handler_output
                
                # Reshape if needed (ensure mono/stereo match)
                if audio_data.ndim == 2:
                     # Channels last
                    if audio_data.shape[1] > audio_data.shape[0]:
                        audio_data = audio_data.T
                    # Mixdown to mono if needed (simple average)
                    if audio_data.shape[1] > 1:
                        audio_data = np.mean(audio_data, axis=1)

                # Ensure float32 for Reachy SDK
                audio_frame = audio_to_float32(audio_data)

                # Resample if needed
                if sr != output_sample_rate:
                    audio_frame = resample(
                        audio_frame,
                        int(len(audio_frame) * output_sample_rate / sr),
                    )

                # Calculate duration to block mic
                duration = len(audio_frame) / output_sample_rate
                # Add a small buffer (e.g. 0.1s) for system latency
                self._playback_end_time = time.time() + duration + 0.1

                # Push to SDK
                logger.debug(f"Pushing {len(audio_frame)} samples to speaker")
                self._robot.media.push_audio_sample(audio_frame)

            await asyncio.sleep(0.01)
