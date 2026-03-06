"""FaceWatcher: background thread that polls the robot camera for faces.

Maintains a two-state machine:

  SLEEPING  — antennas down, breathing suppressed, mic gated off.
              Still scanning for faces.
  AWAKE     — antennas up, breathing active, mic forwarding to Gemini.

Transitions:
  SLEEPING → AWAKE:  face detected for WAKE_MIN_FRAMES within WAKE_WINDOW_SIZE.
  AWAKE   → SLEEPING: no face for SLEEP_TIMEOUT_S seconds.

While AWAKE, periodically runs LBPH identification on the largest detected face.
If LBPH returns Unknown, a rate-limited Gemini vision fallback is fired async.
When a person is identified, their memory context is injected into the brain and
a session is started in the memory server.
"""

import asyncio
import base64
import logging
import threading
import time
from collections import deque
from typing import Any, Optional

import cv2
import numpy as np

try:
    from src.drivers.moves.primitives import AntennaMove
except ImportError:
    from drivers.moves.primitives import AntennaMove

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timing & detection tunables
# ---------------------------------------------------------------------------
POLL_INTERVAL = 0.1           # seconds between camera polls (10 Hz)
WAKE_WINDOW_SIZE = 10         # sliding window size for wake detection
WAKE_MIN_FRAMES = 2           # min detections within window to wake
SLEEP_TIMEOUT_S = 15.0        # seconds without a face before going back to sleep

# Haar cascade parameters (used for the wake/sleep gate — FaceIdentifier has its own)
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
MIN_FACE_SIZE = (40, 40)

# Antenna positions (radians)
ANTENNA_UP = np.deg2rad(90.0)
ANTENNA_DOWN = np.deg2rad(-90.0)
ANTENNA_MOVE_DURATION = 1.1

# ---------------------------------------------------------------------------
# Person tracking tunables
# ---------------------------------------------------------------------------
TRACK_YAW_GAIN = 0.8
TRACK_DEAD_ZONE = 0.05
TRACK_SMOOTHING = 0.7

# ---------------------------------------------------------------------------
# Identification tunables
# ---------------------------------------------------------------------------
IDENTIFY_INTERVAL_S = 3.0    # run LBPH at most every N seconds when AWAKE
FALLBACK_COOLDOWN_S = 30.0   # min seconds between Gemini fallback calls


class FaceWatcher:
    """Polls the robot camera and drives SLEEPING ↔ AWAKE transitions.

    Public state consumed by LocalStream:
        face_watcher.awake  → bool, True when robot should be interactive.
    """

    def __init__(
        self,
        robot: Any,
        movement_manager: Any,
        brain: Any = None,
        face_identifier: Any = None,
        memory_server: Any = None,
    ):
        self._robot = robot
        self._moves = movement_manager
        self._brain = brain
        self._identifier = face_identifier
        self._memory = memory_server

        # Shared wake/sleep state
        self._awake = False
        self._lock = threading.Lock()

        # Internal state machine
        self._detection_window: deque = deque(maxlen=WAKE_WINDOW_SIZE)
        self._last_face_time: float = 0.0

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Person tracking — smoothed yaw offset fed to MovementManager
        self._track_yaw: float = 0.0
        self._track_lock = threading.Lock()

        # Person identification state (all guarded by _identify_lock)
        self._identify_lock = threading.Lock()
        self._current_person_id: Optional[int] = None
        self._current_person_name: str = "Unknown"
        self._current_session_id: Optional[int] = None
        self._last_identified_name: str = ""
        self._last_identify_time: float = 0.0
        self._last_fallback_time: float = 0.0

        # Event loop reference — set by main.py after init so we can fire
        # async callbacks from this background thread.
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # Latest frame slot — written by _detect_face(), read by MonitorServer
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_lock = threading.Lock()

        # Debug stats
        self._stats_time: float = 0.0
        self._stats_frames: int = 0
        self._stats_detections: int = 0
        self._stats_errors: int = 0
        self._stats_none_frames: int = 0

        # Haar cascade for the wake/sleep gate (fast, lightweight)
        self._detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self._detector.empty():
            logger.error("FaceWatcher: failed to load Haar cascade.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="face-watcher"
        )
        self._thread.start()
        logger.info("FaceWatcher started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("FaceWatcher stopped.")

    @property
    def face_present(self) -> bool:
        """True when the robot is AWAKE. Used by LocalStream to gate the mic."""
        with self._lock:
            return self._awake

    @property
    def awake(self) -> bool:
        with self._lock:
            return self._awake

    @property
    def current_person_id(self) -> Optional[int]:
        with self._identify_lock:
            return self._current_person_id

    @property
    def current_person_name(self) -> str:
        with self._identify_lock:
            return self._current_person_name

    @property
    def latest_frame(self) -> Optional[np.ndarray]:
        """Most recent camera frame captured by the face watcher loop.

        Thread-safe. Returns None only before the first frame arrives.
        Used by MonitorServer to stream video without making independent
        get_frame() calls (which compete with the GStreamer pipeline).
        """
        with self._latest_frame_lock:
            return self._latest_frame

    def force_sleep(self) -> None:
        """Immediately transition to SLEEPING, bypassing the face-timeout.

        Safe to call from any thread (e.g. from a Gemini tool callback).
        """
        with self._lock:
            if not self._awake:
                logger.info("FaceWatcher.force_sleep: already sleeping, no-op.")
                return
        logger.info("FaceWatcher.force_sleep: forcing AWAKE → SLEEPING transition.")
        self._enter_sleep()

    def get_face_tracking_offsets(self) -> tuple:
        """Return a 6-DOF secondary offset tuple for MovementManager.

        Only yaw (index 5) is populated.
        """
        with self._track_lock:
            yaw = self._track_yaw
        return (0.0, 0.0, 0.0, 0.0, 0.0, yaw)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        logger.info("FaceWatcher loop running — starting in SLEEPING state.")
        with self._lock:
            self._awake = False
        self._detection_window.clear()
        self._stats_time = time.monotonic()

        while not self._stop_event.is_set():
            loop_start = time.monotonic()

            detected, frame = self._detect_face()

            with self._lock:
                was_awake = self._awake

            self._detection_window.append(detected)
            is_face_present = sum(self._detection_window) >= WAKE_MIN_FRAMES

            if is_face_present:
                self._last_face_time = time.monotonic()

            if not was_awake:
                if is_face_present:
                    self._enter_awake()
            else:
                since_last_face = time.monotonic() - self._last_face_time
                if since_last_face >= SLEEP_TIMEOUT_S:
                    self._enter_sleep()
                elif detected and frame is not None and self._identifier is not None:
                    # Run person identification (debounced)
                    now = time.monotonic()
                    if now - self._last_identify_time >= IDENTIFY_INTERVAL_S:
                        self._last_identify_time = now
                        self._run_identification(frame)

            # Periodic debug summary (every 2 seconds)
            now = time.monotonic()
            if now - self._stats_time >= 2.0:
                state_str = "AWAKE" if was_awake else "SLEEPING"
                logger.debug(
                    "[FaceWatcher] state=%s  frames=%d  detections=%d  "
                    "none_frames=%d  errors=%d  window=%d  person='%s'",
                    state_str,
                    self._stats_frames,
                    self._stats_detections,
                    self._stats_none_frames,
                    self._stats_errors,
                    sum(self._detection_window),
                    self._current_person_name,
                )
                self._stats_frames = 0
                self._stats_detections = 0
                self._stats_none_frames = 0
                self._stats_errors = 0
                self._stats_time = now

            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, POLL_INTERVAL - elapsed))

        logger.debug("FaceWatcher loop exited.")

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _enter_awake(self) -> None:
        logger.info("FaceWatcher: SLEEPING → AWAKE (face confirmed)")
        with self._lock:
            self._awake = True

        # Wake movement system first so AntennaMove can play
        self._moves.set_sleeping(False)

        # Raise antennas
        try:
            move = AntennaMove(
                target_antennas=(ANTENNA_UP, ANTENNA_UP),
                duration=ANTENNA_MOVE_DURATION,
                start_antennas=(ANTENNA_DOWN, ANTENNA_DOWN),
            )
            self._moves.queue_move(move)
        except Exception as e:
            logger.warning("FaceWatcher: antenna up failed: %s", e)

        # Open a fresh Gemini session
        if self._brain is not None:
            self._brain.wake()

        # Start a session record (person_id unknown until LBPH runs)
        if self._memory is not None:
            try:
                session_id = self._memory.start_session(person_id=None)
                with self._identify_lock:
                    self._current_session_id = session_id
                    self._current_person_id = None
                    self._current_person_name = "Unknown"
                    self._last_identified_name = ""
                self._last_identify_time = 0.0  # force immediate LBPH attempt
                logger.info("FaceWatcher: started session %d", session_id)
            except Exception as e:
                logger.warning("FaceWatcher: start_session failed: %s", e)

    def _enter_sleep(self) -> None:
        logger.info("FaceWatcher: → SLEEPING (antennas down, breathing off)")

        # End the current session before tearing down the brain
        if self._memory is not None:
            with self._identify_lock:
                session_id = self._current_session_id
                self._current_session_id = None
                self._current_person_id = None
                self._current_person_name = "Unknown"
                self._last_identified_name = ""
            if session_id is not None:
                try:
                    self._memory.end_session(session_id)
                    logger.info("FaceWatcher: ended session %d", session_id)
                except Exception as e:
                    logger.warning("FaceWatcher: end_session failed: %s", e)

        # Tear down Gemini session
        if self._brain is not None:
            self._brain.sleep()

        # Tell MovementManager to sleep
        self._moves.set_sleeping(True)

        # Lower antennas
        try:
            move = AntennaMove(
                target_antennas=(ANTENNA_DOWN, ANTENNA_DOWN),
                duration=ANTENNA_MOVE_DURATION,
                start_antennas=(ANTENNA_UP, ANTENNA_UP),
            )
            self._moves.queue_move(move)
        except Exception as e:
            logger.warning("FaceWatcher: antenna down failed: %s", e)

        with self._lock:
            self._awake = False
        self._detection_window.clear()

        # Zero out tracking
        with self._track_lock:
            self._track_yaw = 0.0

    # ------------------------------------------------------------------
    # Detection (Haar cascade — lightweight, 10 Hz)
    # ------------------------------------------------------------------

    def _detect_face(self):
        """Grab a frame and run Haar detection. Returns (found: bool, frame)."""
        if self._detector.empty():
            return False, None

        try:
            frame = self._robot.media.get_frame()
        except Exception as e:
            self._stats_errors += 1
            logger.debug("FaceWatcher: camera read error: %s", e)
            return False, None

        if frame is None:
            self._stats_none_frames += 1
            return False, None

        # Cache the latest frame so MonitorServer can stream it without
        # competing with this loop for the GStreamer pipeline.
        with self._latest_frame_lock:
            self._latest_frame = frame

        self._stats_frames += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._detector.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_FACE_SIZE,
        )

        found = len(faces) > 0
        if found:
            self._stats_detections += 1

        is_awake = self.awake

        # --- Person tracking (body-yaw correction, only while AWAKE) ---
        if is_awake:
            if found:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                frame_w = frame.shape[1]
                norm_offset = (x + w / 2.0) / frame_w - 0.5
                if abs(norm_offset) < TRACK_DEAD_ZONE:
                    norm_offset = 0.0
                raw_yaw = norm_offset * TRACK_YAW_GAIN
                with self._track_lock:
                    self._track_yaw = (
                        TRACK_SMOOTHING * self._track_yaw
                        + (1.0 - TRACK_SMOOTHING) * raw_yaw
                    )
            else:
                with self._track_lock:
                    self._track_yaw *= TRACK_SMOOTHING
        else:
            with self._track_lock:
                self._track_yaw = 0.0

        if not is_awake and found:
            sizes = [f"{w}x{h}" for (_, _, w, h) in faces]
            logger.debug(
                "[FaceWatcher] ASLEEP detected %d face(s): %s frame=%dx%d",
                len(faces), ", ".join(sizes), frame.shape[1], frame.shape[0],
            )
        elif is_awake and not found:
            logger.debug("[FaceWatcher] AWAKE but no faces in view.")

        return found, frame

    # ------------------------------------------------------------------
    # Person identification (LBPH, every IDENTIFY_INTERVAL_S)
    # ------------------------------------------------------------------

    def _run_identification(self, frame: np.ndarray) -> None:
        """Run LBPH identification on the current frame. Never raises."""
        try:
            result = self._identifier.identify_face(frame)
            if not result["found"]:
                return

            if result["is_known"]:
                self._handle_known_person(result["name"])
            else:
                # Gemini fallback — rate-limited, fire-and-forget
                now = time.monotonic()
                if (
                    self._brain is not None
                    and hasattr(self._brain, "identify_unknown_face")
                    and now - self._last_fallback_time >= FALLBACK_COOLDOWN_S
                    and self._event_loop is not None
                ):
                    self._last_fallback_time = now
                    crop = result.get("face_crop")
                    if crop is not None:
                        try:
                            _, buf = cv2.imencode(".jpg", crop)
                            face_b64 = base64.b64encode(buf.tobytes()).decode()
                            asyncio.run_coroutine_threadsafe(
                                self._brain.identify_unknown_face(face_b64),
                                self._event_loop,
                            )
                            logger.debug("FaceWatcher: fired Gemini fallback identification.")
                        except Exception as e:
                            logger.debug("FaceWatcher: fallback encode error: %s", e)
        except Exception as e:
            logger.warning("_run_identification error: %s", e)

    def _handle_known_person(self, name: str) -> None:
        """Update person state when LBPH recognises someone."""
        with self._identify_lock:
            name_changed = name != self._last_identified_name
            self._current_person_name = name
            self._last_identified_name = name

        if name_changed:
            logger.info("FaceWatcher: identified person '%s'", name)
            if self._memory is not None:
                try:
                    person_id = self._memory.get_or_create_person(face_label=name)
                    with self._identify_lock:
                        self._current_person_id = person_id
                        session_id = self._current_session_id

                    # Link session to this person
                    if session_id is not None:
                        self._memory.update_session_person(session_id, person_id)

                    # Inject context into brain
                    if self._brain is not None and hasattr(self._brain, "set_active_person"):
                        context_str = self._memory.get_person_context(person_id)
                        self._brain.set_active_person(person_id, name, context_str)

                    logger.info(
                        "FaceWatcher: person '%s' linked (id=%d, session=%s)",
                        name, person_id, session_id,
                    )
                except Exception as e:
                    logger.warning("_handle_known_person error: %s", e)
        else:
            # Same person — just bump last_seen (already debounced by IDENTIFY_INTERVAL_S)
            with self._identify_lock:
                pid = self._current_person_id
            if pid is not None and self._memory is not None:
                try:
                    self._memory.update_person_last_seen(pid)
                except Exception as e:
                    logger.debug("update_person_last_seen error: %s", e)
