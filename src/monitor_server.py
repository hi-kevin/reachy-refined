"""MonitorServer: HTTP status and live video server for Reachy Mini.

Runs as a background daemon thread alongside main.py.
Exposes five endpoints on port 8765:

  GET /video             — MJPEG live stream (15 fps)
  GET /status            — JSON snapshot of robot state
  GET /memories/recent   — JSON: last 10 ST memories for active person
  GET /memories/people   — JSON: all known people + their memories
  GET /log               — JSON: last 50 log lines

Usage in main.py:
    log_capture = LogCapture()
    logging.getLogger().addHandler(log_capture)
    ...
    monitor = MonitorServer(robot, face_watcher, moves, memory, log_capture)
    monitor.start()
"""

from __future__ import annotations

import collections
import http.server
import json
import logging
import socketserver
import sqlite3
import threading
import time
from datetime import datetime
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _make_placeholder_jpeg(width: int = 640, height: int = 480) -> bytes:
    """Generate a black JPEG frame with 'No Camera Signal' text."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        img, "No Camera Signal",
        (width // 2 - 130, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2, cv2.LINE_AA,
    )
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return buf.tobytes()


_PLACEHOLDER_JPEG = _make_placeholder_jpeg()

# ---------------------------------------------------------------------------
# Log capture handler
# ---------------------------------------------------------------------------

class LogCapture(logging.Handler):
    """Ring-buffer logging handler — keeps the last N formatted log lines."""

    def __init__(self, maxlen: int = 200) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._buffer: collections.deque[str] = collections.deque(maxlen=maxlen)
        self.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s %(message)s"
        ))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            with self._lock:
                self._buffer.append(line)
        except Exception:
            self.handleError(record)

    def get_lines(self, n: int = 50) -> list[str]:
        """Return the last n lines, oldest first."""
        with self._lock:
            lines = list(self._buffer)
        return lines[-n:]


# ---------------------------------------------------------------------------
# Threading HTTP server
# ---------------------------------------------------------------------------

class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """HTTPServer that spawns a thread per connection (needed for /video)."""
    daemon_threads = True
    allow_reuse_address = True


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

_BOUNDARY = b"frame"
_CRLF = b"\r\n"


class _MonitorRequestHandler(http.server.BaseHTTPRequestHandler):

    def __init__(self, *args: Any, server_state: "MonitorServer", **kwargs: Any) -> None:
        self._state = server_state
        super().__init__(*args, **kwargs)  # calls handle() immediately

    def log_message(self, format: str, *args: Any) -> None:
        # Route HTTP access logs through the Python logger (captured by LogCapture)
        # Skip /video per-frame noise since that connection stays open
        path = getattr(self, "path", "") or ""
        if "/video" not in path:
            logger.debug("HTTP %s %s", self.address_string(), format % args)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        routes = {
            "/video":           self._handle_video,
            "/status":          self._handle_status,
            "/memories/recent": self._handle_memories_recent,
            "/memories/people": self._handle_memories_people,
            "/log":             self._handle_log,
        }
        handler = routes.get(path)
        if handler:
            try:
                handler()
            except Exception as exc:
                logger.warning("MonitorServer handler error (%s): %s", path, exc)
                try:
                    self._send_json({"error": str(exc)}, 500)
                except Exception:
                    pass
        else:
            self.send_error(404, "Not found")

    # ------------------------------------------------------------------
    # /video — MJPEG stream
    # ------------------------------------------------------------------

    def _handle_video(self) -> None:
        client = self.address_string()
        logger.info("MonitorServer: /video client connected (%s)", client)

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        target_interval = 1.0 / 15  # 15 fps
        frames_sent = 0
        try:
            while True:
                t0 = time.monotonic()
                jpeg = self._state._get_latest_frame_jpeg()
                if jpeg is None:
                    jpeg = _PLACEHOLDER_JPEG  # always send something

                frame_header = (
                    b"--" + _BOUNDARY + _CRLF
                    + b"Content-Type: image/jpeg" + _CRLF
                    + b"Content-Length: " + str(len(jpeg)).encode() + _CRLF
                    + _CRLF
                )
                self.wfile.write(frame_header)
                self.wfile.write(jpeg)
                self.wfile.write(_CRLF)
                self.wfile.flush()
                frames_sent += 1

                if frames_sent % 150 == 0:  # log every ~10s
                    logger.debug(
                        "MonitorServer: /video streaming to %s (%d frames sent)",
                        client, frames_sent,
                    )

                elapsed = time.monotonic() - t0
                time.sleep(max(0.0, target_interval - elapsed))

        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # client disconnected — normal
        finally:
            logger.info(
                "MonitorServer: /video client disconnected (%s, %d frames sent)",
                client, frames_sent,
            )

    # ------------------------------------------------------------------
    # /status — robot state snapshot
    # ------------------------------------------------------------------

    def _handle_status(self) -> None:
        fw = self._state._face_watcher
        mv = self._state._moves
        mv_status = mv.get_status() if mv else {}
        freq = mv_status.get("loop_frequency", {})

        payload: dict[str, Any] = {
            "awake":            getattr(fw, "awake", False),
            "face_present":     getattr(fw, "face_present", False),
            "person_id":        getattr(fw, "current_person_id", None),
            "person_name":      getattr(fw, "current_person_name", None),
            "session_id":       getattr(fw, "_current_session_id", None),
            "queue_size":       mv_status.get("queue_size", 0),
            "breathing_active": mv_status.get("breathing_active", False),
            "is_listening":     mv_status.get("is_listening", False),
            "loop_hz": {
                "last":  round(freq.get("last", 0.0), 1),
                "mean":  round(freq.get("mean", 0.0), 1),
                "min":   round(freq.get("min", 0.0), 1),
            },
            "timestamp": datetime.now().isoformat(),
        }
        self._send_json(payload)

    # ------------------------------------------------------------------
    # /memories/recent — last 10 ST memories for active person
    # ------------------------------------------------------------------

    def _handle_memories_recent(self) -> None:
        fw = self._state._face_watcher
        memory = self._state._memory
        pid = getattr(fw, "current_person_id", None)
        pname = getattr(fw, "current_person_name", None)

        rows = []
        if pid is not None and memory is not None:
            try:
                with sqlite3.connect(memory.db_path, check_same_thread=False) as conn:
                    rows = conn.execute(
                        """SELECT id, content, timestamp, session_id, consolidated
                           FROM short_term_memories
                           WHERE person_id = ?
                           ORDER BY timestamp DESC LIMIT 10""",
                        (pid,),
                    ).fetchall()
            except Exception as exc:
                logger.warning("memories/recent DB error: %s", exc)

        self._send_json({
            "person_id":   pid,
            "person_name": pname,
            "memories": [
                {
                    "id":           r[0],
                    "content":      r[1],
                    "timestamp":    r[2],
                    "session_id":   r[3],
                    "consolidated": r[4],
                }
                for r in reversed(rows)  # oldest first
            ],
        })

    # ------------------------------------------------------------------
    # /memories/people — all known people with memories
    # ------------------------------------------------------------------

    def _handle_memories_people(self) -> None:
        memory = self._state._memory
        result: list[dict] = []

        if memory is not None:
            try:
                with sqlite3.connect(memory.db_path, check_same_thread=False) as conn:
                    people = conn.execute(
                        """SELECT id, display_name, face_label,
                                  first_seen, last_seen, visit_count
                           FROM people ORDER BY last_seen DESC"""
                    ).fetchall()

                    for pid, dname, flabel, fseen, lseen, vcount in people:
                        lt = conn.execute(
                            """SELECT summary FROM long_term_memories
                               WHERE person_id = ? ORDER BY created_at DESC LIMIT 1""",
                            (pid,),
                        ).fetchone()
                        st = conn.execute(
                            """SELECT content, timestamp FROM short_term_memories
                               WHERE person_id = ? ORDER BY timestamp DESC LIMIT 5""",
                            (pid,),
                        ).fetchall()
                        result.append({
                            "id":           pid,
                            "display_name": dname,
                            "face_label":   flabel,
                            "first_seen":   fseen,
                            "last_seen":    lseen,
                            "visit_count":  vcount,
                            "lt_summary":   lt[0] if lt else None,
                            "recent_memories": [
                                {"content": r[0], "timestamp": r[1]}
                                for r in reversed(st)
                            ],
                        })
            except Exception as exc:
                logger.warning("memories/people DB error: %s", exc)

        self._send_json({"people": result})

    # ------------------------------------------------------------------
    # /log — last N log lines
    # ------------------------------------------------------------------

    def _handle_log(self) -> None:
        lines = self._state._log_capture.get_lines(50)
        self._send_json({"lines": lines, "count": len(lines)})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Main server class
# ---------------------------------------------------------------------------

class MonitorServer:
    """HTTP monitor server — starts as a daemon thread, non-blocking."""

    def __init__(
        self,
        robot: Any,
        face_watcher: Any,
        moves: Any,
        memory: Any,
        log_capture: LogCapture,
        port: int = 8765,
    ) -> None:
        self._robot = robot
        self._face_watcher = face_watcher
        self._moves = moves
        self._memory = memory
        self._log_capture = log_capture
        self._port = port

        # Latest JPEG frame slot — written by video loop, read by /video handler
        self._frame_lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None

        # Build server with handler factory that passes self as server_state
        def _handler_factory(*args: Any, **kwargs: Any) -> _MonitorRequestHandler:
            return _MonitorRequestHandler(*args, server_state=self, **kwargs)

        self._httpd = _ThreadingHTTPServer(("", port), _handler_factory)

    def start(self) -> None:
        """Start video capture loop and HTTP server in daemon threads."""
        # Video capture loop
        video_thread = threading.Thread(
            target=self._video_capture_loop,
            daemon=True,
            name="monitor-video",
        )
        video_thread.start()

        # HTTP server
        server_thread = threading.Thread(
            target=self._httpd.serve_forever,
            daemon=True,
            name="monitor-server",
        )
        server_thread.start()

        logger.info("MonitorServer: listening on http://0.0.0.0:%d", self._port)

    def _video_capture_loop(self) -> None:
        """Encode frames from FaceWatcher at ~15 Hz, store JPEG in slot.

        We read from face_watcher.latest_frame rather than calling
        robot.media.get_frame() directly.  FaceWatcher already polls the
        camera at 10 Hz (even while sleeping) and caches the last good
        frame, so we never compete with the GStreamer pipeline and we
        always have something to show.
        """
        TARGET_FPS = 15
        INTERVAL = 1.0 / TARGET_FPS

        while True:
            t0 = time.monotonic()
            try:
                frame = getattr(self._face_watcher, "latest_frame", None)
                if frame is not None:
                    ok, buf = cv2.imencode(
                        ".jpg", frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 70],
                    )
                    if ok:
                        with self._frame_lock:
                            self._latest_jpeg = buf.tobytes()
            except Exception as exc:
                logger.debug("MonitorServer video capture error: %s", exc)

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, INTERVAL - elapsed))

    def _get_latest_frame_jpeg(self) -> Optional[bytes]:
        """Thread-safe read of the latest JPEG frame bytes."""
        with self._frame_lock:
            return self._latest_jpeg
