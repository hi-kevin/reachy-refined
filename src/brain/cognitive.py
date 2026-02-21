import asyncio
import logging
import os
import time
import traceback
from typing import Optional, Tuple, AsyncGenerator, List, Dict, Any, TYPE_CHECKING
import numpy as np
from scipy.signal import resample

from google import genai
from google.genai import types

# Import RoboticsBrain protocol or class
from .robotics import RoboticsBrain
from ..memory.server import MemoryServer
from ..robot_mcp_server import RobotMCPServer, TOOL_DECLARATIONS as ROBOT_TOOL_DECLARATIONS

logger = logging.getLogger(__name__)

# Constants
MODEL_ID = "gemini-2.5-flash-native-audio-preview-12-2025"
AUDIO_SAMPLE_RATE = 16000
AUDIO_OUT_SAMPLE_RATE = 24000

# Sentinel placed in output_queue to signal end of a turn's audio
TURN_END = object()


class CognitiveBrain:
    """
    Handles conversation and high-level reasoning using Gemini Live API.
    Manages the Bidi (Bidirectional) stream for low-latency voice interaction.

    Session lifecycle is tied to FaceWatcher AWAKE/SLEEPING transitions:
      wake()  → opens a fresh Gemini Live session + send/receive loops
      sleep() → tears down the session (Gemini goes stale after long silence)
    """

    def __init__(self, robotics_brain: Optional[RoboticsBrain] = None, memory_server: Optional[MemoryServer] = None, api_key: Optional[str] = None, robot_mcp: Optional[RobotMCPServer] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            logger.error("GOOGLE_API_KEY not found.")

        # Match reference implementation using v1beta
        self.client = genai.Client(
            api_key=self.api_key,
            http_options={"api_version": "v1beta"}
        )
        self.output_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self._session = None
        self._send_queue: asyncio.Queue = asyncio.Queue()
        self._receive_task = None
        self._send_task = None

        # Session lifecycle events
        self._wake_event: asyncio.Event = asyncio.Event()
        self._session_active = False
        self._connection_lost: asyncio.Event = asyncio.Event()

        # Injected brains
        self.robotics_brain = robotics_brain
        self.memory_server = memory_server
        self.robot_mcp = robot_mcp

    # ------------------------------------------------------------------
    # Public wake/sleep API (called from FaceWatcher transitions)
    # ------------------------------------------------------------------

    def wake(self) -> None:
        """Signal that a face is present — start a Gemini session."""
        logger.info("CognitiveBrain: wake signal received.")
        self._wake_event.set()

    def sleep(self) -> None:
        """Signal that no face is present — tear down the Gemini session."""
        logger.info("CognitiveBrain: sleep signal received — closing session.")
        self._session_active = False
        self._connection_lost.clear()  # prevent stale reconnect signal
        # Cancel send/receive tasks so session context manager exits
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
        if self._send_task and not self._send_task.done():
            self._send_task.cancel()
        # Drain queued audio so the next session starts clean
        while not self._send_queue.empty():
            try:
                self._send_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def start_up(self):
        """Main loop: waits for wake signals, opens a fresh Gemini session each time."""
        self.running = True
        logger.info(f"Cognitive Brain ready (Model: {MODEL_ID}). Waiting for first face...")

        while self.running:
            # Block until FaceWatcher calls wake()
            self._wake_event.clear()
            await self._wake_event.wait()
            if not self.running:
                break

            logger.info("Cognitive Brain: opening new Gemini Live session...")
            await self._run_session()
            logger.info("Cognitive Brain: session ended. Waiting for next wake...")

    async def _run_session(self):
        """Open one Gemini Live session and run until sleep() is called.
        Auto-reconnects on transient connection errors (e.g. keepalive timeout)."""
        MAX_RETRIES = 5
        retry_count = 0

        # Tool definitions — built dynamically so new robot tools are
        # picked up automatically from ROBOT_TOOL_DECLARATIONS.
        _TYPE_MAP = {"STRING": "STRING", "NUMBER": "NUMBER", "INTEGER": "INTEGER", "BOOLEAN": "BOOLEAN"}

        def _build_schema(params: Dict[str, Any], required: List[str]) -> types.Schema:
            """Convert a ROBOT_TOOL_DECLARATIONS parameter dict into a types.Schema."""
            if not params:
                return types.Schema(type="OBJECT", properties={})
            props = {}
            for name, spec in params.items():
                props[name] = types.Schema(
                    type=_TYPE_MAP.get(spec.get("type", "STRING"), "STRING"),
                    description=spec.get("description", ""),
                )
            return types.Schema(type="OBJECT", properties=props, required=required)

        builtin_declarations = [
            types.FunctionDeclaration(
                name="analyze_scene",
                description="Analyze the current visual scene to identify objects, people, or layout.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "question": types.Schema(
                            type="STRING",
                            description="Specific question about what to look for."
                        )
                    },
                    required=["question"]
                )
            ),
            types.FunctionDeclaration(
                name="remember",
                description="Store a piece of information in long-term memory.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "content": types.Schema(
                            type="STRING",
                            description="The information to remember."
                        )
                    },
                    required=["content"]
                )
            ),
            types.FunctionDeclaration(
                name="recall",
                description="Search long-term memory for relevant information.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "query": types.Schema(
                            type="STRING",
                            description="Search query."
                        )
                    },
                    required=["query"]
                )
            ),
        ]

        robot_declarations = [
            types.FunctionDeclaration(
                name=decl["name"],
                description=decl["description"],
                parameters=_build_schema(
                    decl.get("parameters", {}),
                    decl.get("required", []),
                ),
            )
            for decl in ROBOT_TOOL_DECLARATIONS
        ]

        all_tool = types.Tool(function_declarations=builtin_declarations + robot_declarations)
        logger.info(
            "Registered %d built-in + %d robot tools with Gemini.",
            len(builtin_declarations),
            len(robot_declarations),
        )

        system_instr = """You are Reachy, a robot companion.
        You are cheerful, curious.
        You can see the world by calling the analyze_scene tool.
        You have a long-term memory. Use 'remember' to save important details about the user.
        Use 'recall' if you need to find past information.
        Be concise in your spoken responses."""

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Zephyr",
                    )
                )
            ),
            tools=[all_tool],
            system_instruction=types.Content(
                parts=[types.Part(text=system_instr)]
            ),
        )

        # Retry loop — reconnects on transient WebSocket failures
        while self._session_active or retry_count == 0:
            self._connection_lost.clear()

            try:
                async with self.client.aio.live.connect(model=MODEL_ID, config=config) as session:
                    self._session = session
                    self._session_active = True
                    retry_count = 0  # reset on successful connect
                    logger.info("Gemini Live session established.")

                    self._receive_task = asyncio.create_task(self._receive_loop())
                    self._send_task = asyncio.create_task(self._send_loop())

                    # Block until sleep() clears _session_active, or connection drops
                    while self._session_active and self.running:
                        # Check for connection loss every 100ms
                        try:
                            await asyncio.wait_for(self._connection_lost.wait(), timeout=0.1)
                            # connection_lost was set — break to reconnect
                            logger.warning("Connection lost detected — will attempt reconnect.")
                            break
                        except asyncio.TimeoutError:
                            pass

                    # Cancel send/receive tasks first
                    for task in [self._receive_task, self._send_task]:
                        if task and not task.done():
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass

                    # If the connection was lost (not a clean sleep), forcibly close the
                    # underlying WebSocket before the async-with __aexit__ tries to — otherwise
                    # it will hang indefinitely on a dead socket.
                    if self._connection_lost.is_set():
                        try:
                            ws = getattr(session, "_ws", None)
                            if ws is not None:
                                await asyncio.wait_for(ws.close(), timeout=2.0)
                        except Exception:
                            pass  # already closed or timed out — that's fine

            except Exception as e:
                logger.error(f"Gemini Live session error: {e}")
                traceback.print_exc()
            finally:
                self._session = None
                logger.info("Gemini Live session closed.")

            # If sleep() was called, don't reconnect
            if not self._session_active or not self.running:
                break

            # Exponential backoff before retry
            retry_count += 1
            if retry_count > MAX_RETRIES:
                logger.error(f"Max reconnect retries ({MAX_RETRIES}) exceeded. Giving up.")
                self._session_active = False
                break

            delay = min(2 ** retry_count, 30)  # 2, 4, 8, 16, 30 seconds
            logger.info(f"Reconnecting in {delay}s (attempt {retry_count}/{MAX_RETRIES})...")
            await asyncio.sleep(delay)

        self._session_active = False

    async def _send_loop(self):
        """Consume audio from local stream and send to Gemini."""
        logger.info("[SEND] Send loop started — forwarding audio to Gemini.")

        buffer = bytearray()
        BUFFER_LIMIT = 2048  # 1024 int16 samples
        MIC_GAIN = 1.0  # Reverted to 1.0 per user request

        # Diagnostic counters
        _frames_received = 0
        _chunks_sent = 0
        _last_stats_time = time.time()
        _logged_sample_rate = False

        while self._session_active and self._session:
            try:
                frame = await self._send_queue.get()
                sr, data = frame
                _frames_received += 1

                # Handling stereo input if present (N, 2)
                if len(data.shape) == 2 and data.shape[1] == 2:
                    data = np.mean(data, axis=1)

                # Log the incoming sample rate once
                if not _logged_sample_rate:
                    logger.info(f"[SEND] First audio frame: sample_rate={sr}, samples={len(data)}, dtype={data.dtype}, shape={data.shape}")
                    if sr != AUDIO_SAMPLE_RATE:
                        logger.warning(f"[SEND] Sample rate mismatch! Robot={sr}Hz, Gemini expects={AUDIO_SAMPLE_RATE}Hz. Will resample.")
                    _logged_sample_rate = True

                # Resample if needed (robot mic rate != 16kHz)
                if sr != AUDIO_SAMPLE_RATE:
                    num_samples = int(len(data) * AUDIO_SAMPLE_RATE / sr)
                    data = resample(data, num_samples).astype(np.float32)

                # Robust floating point normalization and clipping
                # Apply gain
                data = data * MIC_GAIN
                # Clip to valid range BEFORE conversion
                data = np.clip(data, -1.0, 1.0)

                # Convert float32 audio from Reachy SDK to int16 PCM for Gemini
                # Reference uses 32767.0
                audio_int16 = (data * 32767.0).astype(np.int16)

                # Buffer and send in chunks
                buffer.extend(audio_int16.tobytes())

                if len(buffer) >= BUFFER_LIMIT:
                    # Send utilizing the preferred method for the model
                    await self._session.send(input={"data": bytes(buffer), "mime_type": "audio/pcm"})
                    buffer.clear()
                    _chunks_sent += 1

                # Periodic stats (every 5 seconds)
                now = time.time()
                if now - _last_stats_time >= 5.0:
                    qsize = self._send_queue.qsize()
                    logger.info(f"[SEND STATS] frames_in={_frames_received}, chunks_sent={_chunks_sent}, queue_size={qsize}, buffer_bytes={len(buffer)}")
                    _frames_received = 0
                    _chunks_sent = 0
                    _last_stats_time = now

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error sending to Gemini: {e}")
                traceback.print_exc()
                # Signal connection loss so _run_session can reconnect
                self._connection_lost.set()
                break

        logger.info("[SEND] Send loop exited.")

    async def _receive_loop(self):
        """Receive responses (audio/text) from Gemini."""
        logger.info("[RECV] Receive loop started.")
        while self._session_active:
            try:
                if not self._session:
                    break

                async for response in self._session.receive():
                    # --- Log ALL response fields for diagnostics ---
                    if response.voice_activity_detection_signal:
                        logger.info(f"[RECV] VAD SIGNAL: {response.voice_activity_detection_signal}")
                    if response.voice_activity:
                        logger.info(f"[RECV] VOICE ACTIVITY: {response.voice_activity}")
                    if response.go_away:
                        logger.warning(f"[RECV] GO_AWAY: {response.go_away}")
                    if response.session_resumption_update:
                        logger.info(f"[RECV] SESSION RESUMPTION: {response.session_resumption_update}")
                    if response.setup_complete:
                        logger.info(f"[RECV] SETUP COMPLETE")
                    if response.usage_metadata:
                        logger.info(f"[RECV] USAGE: prompt={response.usage_metadata.prompt_token_count}, response={response.usage_metadata.response_token_count}")

                    if response.server_content:
                        # Handle Audio
                        model_turn = response.server_content.model_turn
                        if model_turn:
                            for part in model_turn.parts:
                                if part.inline_data:
                                    # Audio detected
                                    mime_type = part.inline_data.mime_type
                                    if "audio" in mime_type:
                                        # Decode raw PCM from Gemini (24kHz int16)
                                        audio_int16 = np.frombuffer(part.inline_data.data, dtype=np.int16)

                                        logger.info(f"[RECV] Audio chunk: {len(audio_int16)} samples at {AUDIO_OUT_SAMPLE_RATE}Hz")

                                        await self.output_queue.put((AUDIO_OUT_SAMPLE_RATE, audio_int16))
                                    else:
                                        logger.info(f"[RECV] Non-audio inline_data: {mime_type}")
                                elif part.text:
                                    logger.info(f"[RECV] Text: {part.text}")
                                else:
                                    logger.info(f"[RECV] Unknown part type: {part}")

                        # Signal end-of-turn so play_loop flushes the accumulated buffer.
                        # generation_complete fires as soon as the model finishes generating
                        # (turn_complete is delayed waiting for realtime playback to finish).
                        if response.server_content.generation_complete:
                            logger.info(f"[RECV] GENERATION COMPLETE — flushing audio")
                            await self.output_queue.put(TURN_END)
                        if response.server_content.turn_complete:
                            logger.info(f"[RECV] TURN COMPLETE")
                        if response.server_content.interrupted:
                            logger.info(f"[RECV] INTERRUPTED")
                            await self.output_queue.put(TURN_END)

                    if response.tool_call:
                         for fc in response.tool_call.function_calls:
                             logger.info(f"Executing tool: {fc.name}")
                             result_text = "Tool execution failed."

                             if fc.name == "analyze_scene":
                                 args = fc.args
                                 question = args.get("question", "What do you see?")
                                 if self.robotics_brain:
                                     result_text = await self.robotics_brain.capture_and_analyze(question)
                                 else:
                                     result_text = "Vision unavailable."

                             elif fc.name == "remember":
                                 args = fc.args
                                 content = args.get("content", "")
                                 if self.memory_server:
                                     # Sync call in async loop - technically blocking but sqlite is fast
                                     result_text = self.memory_server.remember(content)
                                 else:
                                     result_text = "Memory unavailable."

                             elif fc.name == "recall":
                                 args = fc.args
                                 query = args.get("query", "")
                                 if self.memory_server:
                                     result_text = self.memory_server.recall(query)
                                 else:
                                     result_text = "Memory unavailable."

                             elif self.robot_mcp is not None:
                                 # Route all other tool calls through the robot MCP server
                                 result_text = await self.robot_mcp.dispatch(fc.name, dict(fc.args or {}))

                             else:
                                 result_text = f"No handler for tool: {fc.name}"

                             # Send result back
                             await self._session.send(input=types.ToolResponse(
                                 function_responses=[
                                     types.FunctionResponse(
                                         name=fc.name,
                                         response={"result": result_text}
                                     )
                                 ]
                             ))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error receiving from Gemini: {e}")
                # Signal connection loss so _run_session can reconnect
                self._connection_lost.set()
                break

        logger.info("[RECV] Receive loop exited.")

    async def receive(self, frame: Tuple[int, np.ndarray]):
        """Receive audio frame from LocalStream (External API)."""
        if self._session_active:
            await self._send_queue.put(frame)

    async def emit(self) -> Optional[Tuple[int, np.ndarray]]:
        """Return audio frame to LocalStream (External API)."""
        try:
            return self.output_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def shutdown(self):
        """Close connection."""
        self.running = False
        self._session_active = False
        self._wake_event.set()  # unblock start_up loop so it can exit
        if self._receive_task:
            self._receive_task.cancel()
        if self._send_task:
            self._send_task.cancel()
        logger.info("Cognitive Brain stopped.")
