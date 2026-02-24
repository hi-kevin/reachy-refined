import asyncio
import base64
import logging
import os
import threading
import time
import traceback
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
from scipy.signal import resample

from google import genai
from google.genai import types

from .robotics import RoboticsBrain
from ..memory.server import MemoryServer
from ..robot_mcp_server import RobotMCPServer, TOOL_DECLARATIONS as ROBOT_TOOL_DECLARATIONS

logger = logging.getLogger(__name__)

MODEL_ID = "gemini-2.5-flash-native-audio-preview-12-2025"
AUDIO_SAMPLE_RATE = 16000
AUDIO_OUT_SAMPLE_RATE = 24000

TURN_END = object()


class CognitiveBrain:
    """
    Handles conversation and high-level reasoning using Gemini Live API.

    Session lifecycle is tied to FaceWatcher AWAKE/SLEEPING transitions:
      wake()  → opens a fresh Gemini Live session + send/receive loops
      sleep() → tears down the session

    Person context is injected at session start (from long-term memory summary)
    and can be updated mid-session via set_active_person().
    """

    def __init__(
        self,
        robotics_brain: Optional[RoboticsBrain] = None,
        memory_server: Optional[MemoryServer] = None,
        api_key: Optional[str] = None,
        robot_mcp: Optional[RobotMCPServer] = None,
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            logger.error("GOOGLE_API_KEY not found.")

        self.client = genai.Client(
            api_key=self.api_key,
            http_options={"api_version": "v1beta"},
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

        # Injected dependencies
        self.robotics_brain = robotics_brain
        self.memory_server = memory_server
        self.robot_mcp = robot_mcp

        # Active person context (written by FaceWatcher thread, read in async loop)
        self._person_lock = threading.Lock()
        self._active_person_id: Optional[int] = None
        self._active_person_name: Optional[str] = None
        self._active_person_context: str = ""
        self._active_session_id: Optional[int] = None

        # Event loop reference — set in start_up() so threaded callers can
        # schedule coroutines via asyncio.run_coroutine_threadsafe.
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

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
        self._connection_lost.clear()
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
        if self._send_task and not self._send_task.done():
            self._send_task.cancel()
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
        # Clear active person so the next session gets a fresh start
        with self._person_lock:
            self._active_person_id = None
            self._active_person_name = None
            self._active_person_context = ""
            self._active_session_id = None

    # ------------------------------------------------------------------
    # Person context API (called from FaceWatcher thread)
    # ------------------------------------------------------------------

    def set_active_person(self, person_id: int, name: str, context_str: str) -> None:
        """Store the identified person's context.

        Called from FaceWatcher's background thread.  Thread-safe.
        If a session is already open, injects context as a text turn immediately.
        """
        with self._person_lock:
            self._active_person_id = person_id
            self._active_person_name = name
            self._active_person_context = context_str

        logger.info("CognitiveBrain: active person → '%s' (id=%d)", name, person_id)

        # If session already running, inject context mid-session
        if (
            self._session is not None
            and self._session_active
            and self._event_loop is not None
        ):
            asyncio.run_coroutine_threadsafe(
                self._inject_person_context_message(context_str, name),
                self._event_loop,
            )

    @property
    def active_session_id(self) -> Optional[int]:
        with self._person_lock:
            return self._active_session_id

    @active_session_id.setter
    def active_session_id(self, value: Optional[int]) -> None:
        with self._person_lock:
            self._active_session_id = value

    # ------------------------------------------------------------------
    # Gemini fallback face identification (called from FaceWatcher thread)
    # ------------------------------------------------------------------

    async def identify_unknown_face(self, face_crop_b64: str) -> Optional[str]:
        """Use Gemini vision to describe an unknown face and match to memory.

        Fired asynchronously from FaceWatcher when LBPH returns Unknown.
        Rate-limited by the caller (FALLBACK_COOLDOWN_S).
        """
        if not self.robotics_brain or not self.memory_server:
            return None
        try:
            image_bytes = base64.b64decode(face_crop_b64)
            description = await self.robotics_brain.analyze_scene(
                image_bytes,
                "Describe this person's appearance briefly: hair colour, approximate age, "
                "glasses, distinctive features. Be concise (1-2 sentences).",
            )
            if not description:
                return None

            candidates = self.memory_server.find_person_by_description(description)
            if candidates:
                best = candidates[0]
                guessed_name = best.get("display_name") or best.get("face_label")
                logger.info("Gemini fallback guessed: %s", guessed_name)
                return guessed_name

            logger.info("Gemini fallback: unknown face — %s", descriptio)
            return None
        except Exception as e:
            logger.warning("identify_unknown_face error: %s", e)
            return None

    # ------------------------------------------------------------------
    # Main start-up loop
    # ------------------------------------------------------------------

    async def start_up(self):
        """Main loop: waits for wake signals, opens a fresh Gemini session each time."""
        self._event_loop = asyncio.get_event_loop()
        self.running = True
        logger.info(
            "Cognitive Brain ready (Model: %s). Waiting for first face...", MODEL_ID
        )

        while self.running:
            self._wake_event.clear()
            await self._wake_event.wait()
            if not self.running:
                break

            logger.info("Cognitive Brain: opening new Gemini Live session...")
            await self._run_session()
            logger.info("Cognitive Brain: session ended. Waiting for next wake...")

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def _run_session(self):
        """Open one Gemini Live session and run until sleep() is called."""
        MAX_RETRIES = 5
        retry_count = 0

        _TYPE_MAP = {
            "STRING": "STRING",
            "NUMBER": "NUMBER",
            "INTEGER": "INTEGER",
            "BOOLEAN": "BOOLEAN",
        }

        def _build_schema(params: Dict[str, Any], required: List[str]) -> types.Schema:
            if not params:
                return types.Schema(type="OBJECT", properties={})
            props = {
                name: types.Schema(
                    type=_TYPE_MAP.get(spec.get("type", "STRING"), "STRING"),
                    description=spec.get("description", ""),
                )
                for name, spec in params.items()
            }
            return types.Schema(type="OBJECT", properties=props, required=required)

        # Built-in tool declarations
        builtin_declarations = [
            types.FunctionDeclaration(
                name="analyze_scene",
                description="Analyze the current visual scene to identify objects, people, or layout.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "question": types.Schema(
                            type="STRING",
                            description="Specific question about what to look for.",
                        )
                    },
                    required=["question"],
                ),
            ),
            types.FunctionDeclaration(
                name="remember",
                description="Store a piece of information in long-term memory.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "content": types.Schema(
                            type="STRING",
                            description="The information to remember.",
                        )
                    },
                    required=["content"],
                ),
            ),
            types.FunctionDeclaration(
                name="recall",
                description="Search long-term memory for relevant information.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "query": types.Schema(
                            type="STRING", description="Search query."
                        )
                    },
                    required=["query"],
                ),
            ),
            types.FunctionDeclaration(
                name="register_me",
                description=(
                    "Register the current speaker's face so Reachy can recognise them next time. "
                    "Call this when the user tells you their name for the first time."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "name": types.Schema(
                            type="STRING", description="The person's name."
                        )
                    },
                    required=["name"],
                ),
            ),
            types.FunctionDeclaration(
                name="get_memories_for_me",
                description="Retrieve memories specific to the current person.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "query": types.Schema(
                            type="STRING", description="What to search for."
                        )
                    },
                    required=["query"],
                ),
            ),
        ]

        # Robot movement tool declarations (from RobotMCPServer)
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

        all_tool = types.Tool(
            function_declarations=builtin_declarations + robot_declarations
        )
        logger.info(
            "Registered %d built-in + %d robot tools with Gemini.",
            len(builtin_declarations),
            len(robot_declarations),
        )

        # Build system prompt — inject person context if available
        with self._person_lock:
            person_name = self._active_person_name
            person_context = self._active_person_context

        if person_name and person_context:
            person_section = (
                f"\n\nYou are currently speaking with {person_name}. "
                f"Here is what you know about them:\n{person_context}"
            )
        else:
            person_section = ""

        system_instr = (
            "You are Reachy, a friendly robot companion.\n"
            "You are cheerful, curious, and warm.\n"
            "You can see the world by calling the analyze_scene tool.\n"
            "You have a long-term memory. Use 'remember' to save important details.\n"
            "Use 'recall' for general memory search.\n"
            "Use 'get_memories_for_me' to retrieve memories specific to the current person.\n"
            "If the user tells you their name for the first time, call 'register_me' "
            "to register their face so you can recognise them next time.\n"
            "Be concise in your spoken responses."
            + person_section
        )

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

        while self._session_active or retry_count == 0:
            self._connection_lost.clear()

            try:
                async with self.client.aio.live.connect(
                    model=MODEL_ID, config=config
                ) as session:
                    self._session = session
                    self._session_active = True
                    retry_count = 0
                    logger.info("Gemini Live session established.")

                    self._receive_task = asyncio.create_task(self._receive_loop())
                    self._send_task = asyncio.create_task(self._send_loop())

                    while self._session_active and self.running:
                        try:
                            await asyncio.wait_for(
                                self._connection_lost.wait(), timeout=0.1
                            )
                            logger.warning(
                                "Connection lost detected — will attempt reconnect."
                            )
                            break
                        except asyncio.TimeoutError:
                            pass

                    for task in [self._receive_task, self._send_task]:
                        if task and not task.done():
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass

                    if self._connection_lost.is_set():
                        try:
                            ws = getattr(session, "_ws", None)
                            if ws is not None:
                                await asyncio.wait_for(ws.close(), timeout=2.0)
                        except Exception:
                            pass

            except Exception as e:
                logger.error("Gemini Live session error: %s", e)
                traceback.print_exc()
            finally:
                self._session = None
                logger.info("Gemini Live session closed.")

            if not self._session_active or not self.running:
                break

            retry_count += 1
            if retry_count > MAX_RETRIES:
                logger.error(
                    "Max reconnect retries (%d) exceeded. Giving up.", MAX_RETRIES
                )
                self._session_active = False
                break

            delay = min(2**retry_count, 30)
            logger.info(
                "Reconnecting in %ds (attempt %d/%d)...",
                delay, retry_count, MAX_RETRIES,
            )
            await asyncio.sleep(delay)

        self._session_active = False

    # ------------------------------------------------------------------
    # Send / receive loops
    # ------------------------------------------------------------------

    async def _send_loop(self):
        """Consume audio from local stream and send to Gemini."""
        logger.info("[SEND] Send loop started.")
        buffer = bytearray()
        BUFFER_LIMIT = 2048
        MIC_GAIN = 1.0
        _frames_received = 0
        _chunks_sent = 0
        _last_stats_time = time.time()
        _logged_sample_rate = False

        while self._session_active and self._session:
            try:
                frame = await self._send_queue.get()
                sr, data = frame
                _frames_received += 1

                if len(data.shape) == 2 and data.shape[1] == 2:
                    data = np.mean(data, axis=1)

                if not _logged_sample_rate:
                    logger.info(
                        "[SEND] First audio frame: sr=%d samples=%d dtype=%s",
                        sr, len(data), data.dtype,
                    )
                    if sr != AUDIO_SAMPLE_RATE:
                        logger.warning(
                            "[SEND] Sample rate mismatch! Robot=%dHz Gemini=%dHz — resampling.",
                            sr, AUDIO_SAMPLE_RATE,
                        )
                    _logged_sample_rate = True

                if sr != AUDIO_SAMPLE_RATE:
                    num_samples = int(len(data) * AUDIO_SAMPLE_RATE / sr)
                    data = resample(data, num_samples).astype(np.float32)

                data = np.clip(data * MIC_GAIN, -1.0, 1.0)
                audio_int16 = (data * 32767.0).astype(np.int16)
                buffer.extend(audio_int16.tobytes())

                if len(buffer) >= BUFFER_LIMIT:
                    await self._session.send(
                        input={"data": bytes(buffer), "mime_type": "audio/pcm"}
                    )
                    buffer.clear()
                    _chunks_sent += 1

                now = time.time()
                if now - _last_stats_time >= 5.0:
                    logger.info(
                        "[SEND STATS] frames_in=%d chunks_sent=%d queue=%d",
                        _frames_received, _chunks_sent, self._send_queue.qsize(),
                    )
                    _frames_received = 0
                    _chunks_sent = 0
                    _last_stats_time = now

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error sending to Gemini: %s", e)
                traceback.print_exc()
                self._connection_lost.set()
                break

        logger.info("[SEND] Send loop exited.")

    async def _receive_loop(self):
        """Receive responses (audio / tool calls) from Gemini."""
        logger.info("[RECV] Receive loop started.")
        while self._session_active:
            try:
                if not self._session:
                    break

                async for response in self._session.receive():
                    if response.voice_activity_detection_signal:
                        logger.info("[RECV] VAD: %s", response.voice_activity_detection_signal)
                    if response.voice_activity:
                        logger.info("[RECV] VOICE ACTIVITY: %s", response.voice_activity)
                    if response.go_away:
                        logger.warning("[RECV] GO_AWAY: %s", response.go_away)
                    if response.session_resumption_update:
                        logger.info(
                            "[RECV] SESSION RESUMPTION: %s",
                            response.session_resumption_update,
                        )
                    if response.setup_complete:
                        logger.info("[RECV] SETUP COMPLETE")
                    if response.usage_metadata:
                        logger.info(
                            "[RECV] USAGE: prompt=%d response=%d",
                            response.usage_metadata.prompt_token_count,
                            response.usage_metadata.response_token_count,
                        )

                    if response.server_content:
                        model_turn = response.server_content.model_turn
                        if model_turn:
                            for part in model_turn.parts:
                                if part.inline_data:
                                    mime_type = part.inline_data.mime_type
                                    if "audio" in mime_type:
                                        audio_int16 = np.frombuffer(
                                            part.inline_data.data, dtype=np.int16
                                        )
                                        await self.output_queue.put(
                                            (AUDIO_OUT_SAMPLE_RATE, audio_int16)
                                        )
                                    else:
                                        logger.info(
                                            "[RECV] Non-audio inline_data: %s", mime_type
                                        )
                                elif part.text:
                                    logger.info("[RECV] Text: %s", part.text)

                        if response.server_content.generation_complete:
                            logger.info("[RECV] GENERATION COMPLETE — flushing audio")
                            await self.output_queue.put(TURN_END)
                        if response.server_content.turn_complete:
                            logger.info("[RECV] TURN COMPLETE")
                        if response.server_content.interrupted:
                            logger.info("[RECV] INTERRUPTED")
                            await self.output_queue.put(TURN_END)

                    if response.tool_call:
                        for fc in response.tool_call.function_calls:
                            logger.info("Executing tool: %s", fc.name)
                            args = dict(fc.args or {})
                            result_text = await self._dispatch_tool(fc.name, args)

                            await self._session.send(
                                input=types.ToolResponse(
                                    function_responses=[
                                        types.FunctionResponse(
                                            name=fc.name,
                                            response={"result": result_text},
                                        )
                                    ]
                                )
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error receiving from Gemini: %s", e)
                self._connection_lost.set()
                break

        logger.info("[RECV] Receive loop exited.")

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    async def _dispatch_tool(self, name: str, args: Dict[str, Any]) -> str:
        """Route a Gemini tool call to the appropriate handler."""
        try:
            if name == "analyze_scene":
                question = args.get("question", "What do you see?")
                if self.robotics_brain:
                    return await self.robotics_brain.capture_and_analyze(question)
                return "Vision unavailable."

            elif name == "remember":
                content = args.get("content", "")
                if self.memory_server:
                    with self._person_lock:
                        pid = self._active_person_id
                        sid = self._active_session_id
                    return self.memory_server.remember_for(pid, sid, content)
                return "Memory unavailable."

            elif name == "recall":
                query = args.get("query", "")
                if self.memory_server:
                    with self._person_lock:
                        pid = self._active_person_id
                    if pid:
                        return self.memory_server.recall_for(pid, query)
                    return "I don't know who I'm speaking with yet — please tell me your name so I can look up your memories."
                return "Memory unavailable."

            elif name == "register_me":
                person_name = args.get("name", "").strip()
                if self.robot_mcp:
                    return await self.robot_mcp.dispatch(
                        "register_face", {"name": person_name}
                    )
                return "Registration unavailable."

            elif name == "get_memories_for_me":
                query = args.get("query", "")
                if self.memory_server:
                    with self._person_lock:
                        pid = self._active_person_id
                    if pid:
                        return self.memory_server.recall_for(pid, query)
                    return "I don't know who you are yet. Please tell me your name."
                return "Memory unavailable."

            elif self.robot_mcp is not None:
                return await self.robot_mcp.dispatch(name, args)

            else:
                return f"No handler for tool: {name}"

        except Exception as e:
            logger.error("Tool dispatch error for '%s': %s", name, e)
            return f"Tool {name} failed: {e}"

    # ------------------------------------------------------------------
    # Mid-session context injection
    # ------------------------------------------------------------------

    async def _inject_person_context_message(
        self, context_str: str, name: str
    ) -> None:
        """Send a text turn into the live session to update person context."""
        if not self._session or not self._session_active:
            return
        try:
            text = (
                f"[Context update: You are now speaking with {name}. "
                f"{context_str}]"
            )
            await self._session.send(input=text, end_of_turn=True)
            logger.info("Injected person context for '%s' into live session.", name)
        except Exception as e:
            logger.warning("Failed to inject person context: %s", e)

    # ------------------------------------------------------------------
    # External audio API (called by LocalStream)
    # ------------------------------------------------------------------

    async def receive(self, frame: Tuple[int, np.ndarray]):
        """Push an audio frame from LocalStream into the send queue."""
        if self._session_active:
            await self._send_queue.put(frame)

    async def emit(self) -> Optional[Tuple[int, np.ndarray]]:
        """Pull an audio frame (or TURN_END sentinel) for LocalStream to play."""
        try:
            return self.output_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def shutdown(self):
        """Close connection."""
        self.running = False
        self._session_active = False
        self._wake_event.set()
        if self._receive_task:
            self._receive_task.cancel()
        if self._send_task:
            self._send_task.cancel()
        logger.info("Cognitive Brain stopped.")
