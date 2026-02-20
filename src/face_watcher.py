"""FaceWatcher: background thread that polls the robot camera for faces.

Maintains a two-state machine:

  SLEEPING  — antennas down, breathing suppressed, mic gated off.
              Still scanning for faces.
  AWAKE     — antennas up, breathing active, mic forwarding to Gemini.

Transitions:
  SLEEPING → AWAKE:  face detected for WAKE_CONSECUTIVE consecutive frames.
  AWAKE   → SLEEPING: no face for SLEEP_TIMEOUT_S seconds.
"""

import cv2
import logging
import threading
import time
import numpy as np
from typing import Any, Optional
from collections import deque

try:
    from src.drivers.moves.primitives import AntennaMove
except ImportError:
    from drivers.moves.primitives import AntennaMove

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timing & detection tunables
# ---------------------------------------------------------------------------
POLL_INTERVAL = 0.1          # seconds between camera polls (10 Hz)
WAKE_WINDOW_SIZE = 8         # sliding window size for wake detection
WAKE_MIN_FRAMES = 3          # min detections within window to wake
SLEEP_TIMEOUT_S = 15.0       # seconds without a face before going back to sleep

# Haar cascade parameters
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
# Proportional gain: how many radians of body yaw per unit of normalised
# horizontal face offset (offset ranges -0.5 … +0.5 across the frame).
# Larger = more aggressive tracking.
TRACK_YAW_GAIN = 0.8
# Dead-zone: ignore offsets smaller than this (normalised units) to avoid
# jitter when the face is roughly centred.
TRACK_DEAD_ZONE = 0.05
# Low-pass smoothing for the yaw offset (0 = no smoothing, 1 = frozen).
TRACK_SMOOTHING = 0.7


class FaceWatcher:
    """Polls the robot camera and drives SLEEPING ↔ AWAKE transitions.

    Public state consumed by LocalStream:
        face_watcher.awake  → bool, True when robot should be interactive.
    """

    def __init__(self, robot: Any, movement_manager: Any, brain: Any = None):
        self._robot = robot
        self._moves = movement_manager
        self._brain = brain  # CognitiveBrain — wake()/sleep() control Gemini session

        # Shared state (read by LocalStream via .awake property)
        self._awake = False
        self._lock = threading.Lock()

        # Internal state machine
        self._detection_window: deque = deque(maxlen=WAKE_WINDOW_SIZE)
        self._last_face_time: float = 0.0  # monotonic timestamp of last detection

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Person tracking — smoothed yaw offset fed to MovementManager
        self._track_yaw: float = 0.0       # smoothed body-yaw correction (radians)
        self._track_lock = threading.Lock()

        # Debug stats — logged once per second
        self._stats_time: float = 0.0
        self._stats_frames: int = 0
        self._stats_detections: int = 0
        self._stats_errors: int = 0
        self._stats_none_frames: int = 0

        # OpenCV Haar face detector (CPU, no extra deps)
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
        self._thread = threading.Thread(target=self._loop, daemon=True, name="face-watcher")
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
        """True when the robot is AWAKE (face present or within timeout).

        Used by LocalStream to gate the mic → Gemini audio path.
        """
        with self._lock:
            return self._awake

    @property
    def awake(self) -> bool:
        with self._lock:
            return self._awake

    def get_face_tracking_offsets(self) -> tuple:
        """Return a 6-DOF secondary offset tuple for MovementManager.

        Only yaw (index 5) is populated — a proportional correction that
        rotates the body toward the detected face.  All other DOF are zero
        so this doesn't interfere with head pitch/roll or speech sway.
        """
        with self._track_lock:
            yaw = self._track_yaw
        return (0.0, 0.0, 0.0, 0.0, 0.0, yaw)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        logger.info("FaceWatcher loop running — starting in SLEEPING state.")
        # We start sleeping — MovementManager._sleeping is already True from init,
        # so just make sure our internal state matches. No antenna move needed
        # because the robot hasn't moved its antennas yet.
        with self._lock:
            self._awake = False
        self._detection_window.clear()
        self._stats_time = time.monotonic()

        while not self._stop_event.is_set():
            loop_start = time.monotonic()

            detected = self._detect_face()

            with self._lock:
                was_awake = self._awake

            self._detection_window.append(detected)
            
            # Check if we meet the wake criteria (3 out of 5 frames)
            if sum(self._detection_window) >= WAKE_MIN_FRAMES:
                is_face_present = True
            else:
                is_face_present = False

            if is_face_present:
                self._last_face_time = time.monotonic()

            if not was_awake:
                # SLEEPING — check if we should wake
                if is_face_present:
                    self._enter_awake()
            else:
                # AWAKE — check if we should sleep
                since_last_face = time.monotonic() - self._last_face_time
                if since_last_face >= SLEEP_TIMEOUT_S:
                    self._enter_sleep()

            # Periodic debug summary (every 2 seconds)
            now = time.monotonic()
            if now - self._stats_time >= 2.0:
                state_str = "AWAKE" if was_awake else "SLEEPING"
                logger.debug(
                    "[FaceWatcher] state=%s  frames=%d  detections=%d  "
                    "none_frames=%d  errors=%d  consec=%d",
                    state_str,
                    self._stats_frames,
                    self._stats_detections,
                    self._stats_none_frames,
                    self._stats_errors,
                    sum(self._detection_window),
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
        # Wake the movement system first so the AntennaMove can play
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

    def _enter_sleep(self) -> None:
        logger.info("FaceWatcher: → SLEEPING (antennas down, breathing off)")
        # Tear down Gemini session first
        if self._brain is not None:
            self._brain.sleep()
        # Tell MovementManager to sleep (clears queue, suppresses breathing),
        # then queue the antenna-down move (processed after the clear).
        self._moves.set_sleeping(True)
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

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect_face(self) -> bool:
        """Grab a frame from the robot camera and run Haar face detection.

        While awake, also computes a smoothed body-yaw correction from the
        largest detected face's horizontal position and stores it so
        get_face_tracking_offsets() can supply it to MovementManager.
        """
        if self._detector.empty():
            return False

        try:
            frame = self._robot.media.get_frame()
        except Exception as e:
            self._stats_errors += 1
            logger.debug("FaceWatcher: camera read error: %s", e)
            return False

        if frame is None:
            self._stats_none_frames += 1
            return False

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

        # --- Person tracking (only while awake) ---
        if is_awake:
            if found:
                # Pick the largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                frame_w = frame.shape[1]
                # Normalised horizontal offset: -0.5 (left) … +0.5 (right)
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
                # Decay toward zero when face is lost
                with self._track_lock:
                    self._track_yaw *= TRACK_SMOOTHING
        else:
            # Sleeping — zero out tracking so it doesn't linger on wake
            with self._track_lock:
                self._track_yaw = 0.0

        # Conditional logging
        if not is_awake and found:
            sizes = [f"{w}x{h}" for (_, _, w, h) in faces]
            logger.debug(
                "[FaceWatcher] ASLEEP detected %d face(s): %s frame=%dx%d",
                len(faces), ", ".join(sizes), frame.shape[1], frame.shape[0],
            )
        elif is_awake and not found:
            logger.debug("[FaceWatcher] AWAKE but no faces in view.")

        return found
