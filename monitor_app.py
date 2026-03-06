"""Reachy Mini Desktop Monitor
================================
Live video + status dashboard for the Reachy Mini robot.

Prerequisites (Windows):
    pip install pillow

Run:
    python monitor_app.py

The app connects to http://reachy-mini.local:8765 (set REACHY_HOST env var to
override).  The robot must be running with MonitorServer enabled (src/main.py).
"""

from __future__ import annotations

import io
import json
import os
import queue
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
from tkinter import ttk
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Pillow check — give a clear error before anything else imports fail
# ---------------------------------------------------------------------------
try:
    from PIL import Image, ImageTk
except ImportError:
    import sys
    sys.exit(
        "\nPillow is required for image display.\n"
        "Install it with:  pip install pillow\n"
    )

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REACHY_HOST = os.getenv("REACHY_HOST", "reachy-mini.local")
BASE_URL = f"http://{REACHY_HOST}:8765"

VIDEO_W, VIDEO_H = 640, 480
POLL_STATUS_MS   = 1000
POLL_MEMORIES_MS = 2000
POLL_PEOPLE_MS   = 10_000
POLL_LOG_MS      = 2000
FRAME_POLL_MS    = 33       # ~30 fps UI check
FETCH_TIMEOUT    = 2.5      # seconds for JSON requests

# Dark theme colours
BG          = "#0d0d1a"
BG_PANEL    = "#12122a"
BG_WIDGET   = "#1a1a35"
FG          = "#c8d0e0"
FG_DIM      = "#606880"
ACCENT      = "#5577ff"
GREEN       = "#00dd88"
RED         = "#ff4455"
FONT_MONO   = ("Courier New", 8)
FONT_LABEL  = ("Segoe UI", 9)
FONT_TITLE  = ("Segoe UI", 10, "bold")
FONT_BADGE  = ("Segoe UI", 11, "bold")


# ---------------------------------------------------------------------------
# Background fetch helpers  (all network I/O stays off the main thread)
# ---------------------------------------------------------------------------

def _fetch_json_bg(path: str, result_queue: queue.Queue, tag: str) -> None:
    """Fetch BASE_URL+path in a daemon thread, put (tag, data_or_None) in queue."""
    try:
        with urllib.request.urlopen(BASE_URL + path, timeout=FETCH_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            result_queue.put((tag, data))
    except Exception:
        result_queue.put((tag, None))


def _ts_short(ts: Optional[str]) -> str:
    """Trim an ISO timestamp to 'YYYY-MM-DD HH:MM'."""
    if not ts:
        return "—"
    return ts[:16].replace("T", " ")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class MonitorApp:

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Reachy Mini Monitor")
        root.configure(bg=BG)
        root.resizable(True, True)
        root.minsize(900, 640)

        # MJPEG frame queue: reader thread → main thread
        self._frame_queue: queue.Queue[bytes] = queue.Queue(maxsize=4)
        self._current_photo: Optional[ImageTk.PhotoImage] = None  # GC guard

        # JSON result queue: fetch threads → main thread
        self._json_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

        # Connection state for each endpoint
        self._conn_video   = False
        self._conn_status  = False
        self._conn_mem     = False
        self._conn_people  = False
        self._conn_log     = False

        # Track whether a fetch is in-flight (avoid pile-up)
        self._fetching: set[str] = set()

        self._build_ui()
        self._start_mjpeg_thread()

        # Drain result queue every 100ms on main thread
        self.root.after(100, self._drain_json_queue)

        # Kick off first fetches (staggered slightly)
        self.root.after(100,  lambda: self._kick_fetch("status",  "/status"))
        self.root.after(300,  lambda: self._kick_fetch("mem",     "/memories/recent"))
        self.root.after(500,  lambda: self._kick_fetch("people",  "/memories/people"))
        self.root.after(700,  lambda: self._kick_fetch("log",     "/log"))

        # Frame consumer
        self.root.after(FRAME_POLL_MS, self._process_frame_queue)

    # ------------------------------------------------------------------
    # Background fetch orchestration
    # ------------------------------------------------------------------

    def _kick_fetch(self, tag: str, path: str) -> None:
        """Launch a daemon thread to fetch path unless one is already in flight."""
        if tag in self._fetching:
            return
        self._fetching.add(tag)
        t = threading.Thread(
            target=_fetch_json_bg,
            args=(path, self._json_queue, tag),
            daemon=True,
            name=f"fetch-{tag}",
        )
        t.start()

    def _drain_json_queue(self) -> None:
        """Process all pending fetch results and re-schedule next fetch."""
        reschedule = {
            "status": (POLL_STATUS_MS,   "/status"),
            "mem":    (POLL_MEMORIES_MS,  "/memories/recent"),
            "people": (POLL_PEOPLE_MS,    "/memories/people"),
            "log":    (POLL_LOG_MS,       "/log"),
        }
        while True:
            try:
                tag, data = self._json_queue.get_nowait()
            except queue.Empty:
                break

            self._fetching.discard(tag)

            # Dispatch to appropriate handler
            if tag == "status":
                self._apply_status(data)
            elif tag == "mem":
                self._apply_memories_recent(data)
            elif tag == "people":
                self._apply_people(data)
            elif tag == "log":
                self._apply_log(data)

            # Schedule next fetch
            if tag in reschedule:
                delay_ms, path = reschedule[tag]
                self.root.after(delay_ms, lambda t=tag, p=path: self._kick_fetch(t, p))

        self.root.after(100, self._drain_json_queue)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = self.root

        # ── Outer grid: 2 columns, 3 rows ─────────────────────────────
        root.columnconfigure(0, weight=0)   # left: fixed video width
        root.columnconfigure(1, weight=1)   # right: status/log expand
        root.rowconfigure(0, weight=1)      # top row: video + panels
        root.rowconfigure(1, weight=0)      # bottom row: people browser
        root.rowconfigure(2, weight=0)      # status bar

        # ── Left column: video ─────────────────────────────────────────
        left = tk.Frame(root, bg=BG, width=VIDEO_W)
        left.grid(row=0, column=0, rowspan=1, sticky="ns", padx=(8, 4), pady=8)
        left.grid_propagate(False)

        self._video_label = tk.Label(
            left,
            bg="#000000",
            width=VIDEO_W,
            height=VIDEO_H,
            text="Connecting to robot…",
            fg=FG_DIM,
            font=FONT_LABEL,
        )
        self._video_label.pack(fill="both", expand=True)

        self._video_url_label = tk.Label(
            left, text=f"  {BASE_URL}/video",
            bg=BG, fg=FG_DIM, font=FONT_MONO, anchor="w",
        )
        self._video_url_label.pack(fill="x")

        # ── Right column: status / memories / log ─────────────────────
        right = tk.Frame(root, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=0)
        right.rowconfigure(1, weight=1)
        right.rowconfigure(2, weight=2)

        self._build_status_panel(right)
        self._build_memories_panel(right)
        self._build_log_panel(right)

        # ── Bottom row: people browser ─────────────────────────────────
        bottom = tk.Frame(root, bg=BG)
        bottom.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 4))
        bottom.columnconfigure(0, weight=1)
        self._build_people_panel(bottom)

        # ── Status bar ─────────────────────────────────────────────────
        self._statusbar = tk.Label(
            root,
            text=f"  Connecting to {BASE_URL} …",
            bg="#0a0a18", fg=FG_DIM,
            font=FONT_MONO, anchor="w", pady=2,
        )
        self._statusbar.grid(row=2, column=0, columnspan=2, sticky="ew")

    # ── Status panel ──────────────────────────────────────────────────

    def _build_status_panel(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(
            parent, text="  Robot Status  ",
            bg=BG_PANEL, fg=ACCENT, font=FONT_TITLE,
            bd=1, relief="flat", labelanchor="nw",
        )
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        frame.columnconfigure(1, weight=1)

        self._awake_badge = tk.Label(
            frame, text="OFFLINE", bg=BG_WIDGET, fg=FG_DIM,
            font=FONT_BADGE, width=10, pady=4,
        )
        self._awake_badge.grid(row=0, column=0, rowspan=2, padx=10, pady=8)

        self._person_label  = self._status_row(frame, "Person",    "—", 1)
        self._session_label = self._status_row(frame, "Session",   "—", 2)
        self._hz_label      = self._status_row(frame, "Loop Hz",   "—", 3)
        self._queue_label   = self._status_row(frame, "Queue",     "—", 4)
        self._breath_label  = self._status_row(frame, "Breathing", "—", 5)
        self._listen_label  = self._status_row(frame, "Listening", "—", 6)

    def _status_row(self, parent: tk.Frame, key: str, val: str, row: int) -> tk.Label:
        tk.Label(parent, text=f"{key}:", bg=BG_PANEL, fg=FG_DIM,
                 font=FONT_LABEL, anchor="e", width=10
                 ).grid(row=row, column=1, sticky="e", padx=(0, 4), pady=1)
        lbl = tk.Label(parent, text=val, bg=BG_PANEL, fg=FG,
                       font=FONT_LABEL, anchor="w")
        lbl.grid(row=row, column=2, sticky="w", padx=(0, 10), pady=1)
        return lbl

    # ── Recent memories panel ─────────────────────────────────────────

    def _build_memories_panel(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(
            parent, text="  Recent Memories  ",
            bg=BG_PANEL, fg=ACCENT, font=FONT_TITLE,
            bd=1, relief="flat", labelanchor="nw",
        )
        frame.grid(row=1, column=0, sticky="nsew", pady=(0, 6))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        scroll = ttk.Scrollbar(frame, orient="vertical")
        self._mem_list = tk.Listbox(
            frame,
            bg=BG_WIDGET, fg=FG, selectbackground=ACCENT,
            font=FONT_MONO, bd=0, highlightthickness=0,
            yscrollcommand=scroll.set, activestyle="none",
        )
        scroll.config(command=self._mem_list.yview)
        self._mem_list.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=6)
        scroll.grid(row=0, column=1, sticky="ns", pady=6)

    # ── Log panel ─────────────────────────────────────────────────────

    def _build_log_panel(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(
            parent, text="  Log Tail  ",
            bg=BG_PANEL, fg=ACCENT, font=FONT_TITLE,
            bd=1, relief="flat", labelanchor="nw",
        )
        frame.grid(row=2, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        scroll = ttk.Scrollbar(frame, orient="vertical")
        self._log_text = tk.Text(
            frame,
            state="disabled", wrap="none",
            bg=BG_WIDGET, fg="#88cc88",
            font=FONT_MONO, bd=0, highlightthickness=0,
            yscrollcommand=scroll.set,
        )
        scroll.config(command=self._log_text.yview)
        self._log_text.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=6)
        scroll.grid(row=0, column=1, sticky="ns", pady=6)

    # ── People browser panel ──────────────────────────────────────────

    def _build_people_panel(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(
            parent, text="  Known People  ",
            bg=BG_PANEL, fg=ACCENT, font=FONT_TITLE,
            bd=1, relief="flat", labelanchor="nw",
        )
        frame.grid(row=0, column=0, sticky="ew")
        frame.columnconfigure(0, weight=1)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Monitor.Treeview",
                        background=BG_WIDGET, foreground=FG,
                        fieldbackground=BG_WIDGET,
                        rowheight=20, font=FONT_MONO)
        style.configure("Monitor.Treeview.Heading",
                        background=BG_PANEL, foreground=ACCENT,
                        font=FONT_LABEL)
        style.map("Monitor.Treeview",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", "#ffffff")])

        cols = ("last_seen", "visits", "summary")
        self._people_tree = ttk.Treeview(
            frame,
            columns=cols,
            displaycolumns=cols,
            show="tree headings",
            height=7,
            style="Monitor.Treeview",
        )
        self._people_tree.heading("#0",        text="Name",       anchor="w")
        self._people_tree.heading("last_seen", text="Last Seen",  anchor="w")
        self._people_tree.heading("visits",    text="Visits",     anchor="center")
        self._people_tree.heading("summary",   text="LT Summary", anchor="w")

        self._people_tree.column("#0",        width=140, stretch=False)
        self._people_tree.column("last_seen", width=130, stretch=False)
        self._people_tree.column("visits",    width=55,  stretch=False, anchor="center")
        self._people_tree.column("summary",   width=400, stretch=True)

        scroll = ttk.Scrollbar(frame, orient="vertical",
                               command=self._people_tree.yview)
        self._people_tree.configure(yscrollcommand=scroll.set)
        self._people_tree.grid(row=0, column=0, sticky="ew", padx=(6, 0), pady=6)
        scroll.grid(row=0, column=1, sticky="ns", pady=6)

    # ------------------------------------------------------------------
    # MJPEG reader thread
    # ------------------------------------------------------------------

    def _start_mjpeg_thread(self) -> None:
        t = threading.Thread(
            target=self._mjpeg_reader,
            daemon=True,
            name="mjpeg-reader",
        )
        t.start()

    def _mjpeg_reader(self) -> None:
        """Read MJPEG stream; put JPEG bytes into frame queue."""
        RECONNECT = 3.0
        CHUNK = 8192
        SOI = b"\xff\xd8"
        EOI = b"\xff\xd9"

        while True:
            try:
                req = urllib.request.Request(
                    BASE_URL + "/video",
                    headers={"Connection": "keep-alive"},
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    buf = b""
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        buf += chunk

                        while True:
                            start = buf.find(SOI)
                            if start == -1:
                                buf = buf[-1:]
                                break
                            end = buf.find(EOI, start + 2)
                            if end == -1:
                                buf = buf[start:]
                                break
                            jpeg = buf[start: end + 2]
                            buf = buf[end + 2:]
                            if self._frame_queue.full():
                                try:
                                    self._frame_queue.get_nowait()
                                except queue.Empty:
                                    pass
                            try:
                                self._frame_queue.put_nowait(jpeg)
                            except queue.Full:
                                pass

            except Exception:
                pass

            time.sleep(RECONNECT)

    # ------------------------------------------------------------------
    # Frame consumer (main thread via after())
    # ------------------------------------------------------------------

    def _process_frame_queue(self) -> None:
        try:
            jpeg = self._frame_queue.get_nowait()
            img = Image.open(io.BytesIO(jpeg))
            img = img.resize((VIDEO_W, VIDEO_H), Image.LANCZOS)
            self._current_photo = ImageTk.PhotoImage(img)
            self._video_label.configure(
                image=self._current_photo, text="", bg="#000000"
            )
            if not self._conn_video:
                self._conn_video = True
                self._update_statusbar()
        except queue.Empty:
            pass
        except Exception:
            if self._conn_video:
                self._conn_video = False
                self._video_label.configure(
                    image="", text="Video stream lost — reconnecting…",
                    bg="#000000", fg=FG_DIM,
                )
                self._update_statusbar()
        finally:
            self.root.after(FRAME_POLL_MS, self._process_frame_queue)

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _update_statusbar(self) -> None:
        def dot(ok: bool, label: str) -> str:
            return ("✔" if ok else "✘") + " " + label

        parts = [
            dot(self._conn_video,  "video"),
            dot(self._conn_status, "status"),
            dot(self._conn_mem,    "memories"),
            dot(self._conn_people, "people"),
            dot(self._conn_log,    "log"),
        ]
        self._statusbar.configure(
            text=f"  {BASE_URL}   " + "    ".join(parts)
        )

    # ------------------------------------------------------------------
    # JSON result appliers (called on main thread from _drain_json_queue)
    # ------------------------------------------------------------------

    def _apply_status(self, data: Optional[dict]) -> None:
        changed = (data is not None) != self._conn_status
        self._conn_status = data is not None
        if changed:
            self._update_statusbar()

        if data:
            awake = data.get("awake", False)
            fp    = data.get("face_present", False)
            if awake:
                badge_text, badge_bg, badge_fg = "AWAKE",    GREEN,    "#000000"
            elif fp:
                badge_text, badge_bg, badge_fg = "FACE",     "#ffaa00","#000000"
            else:
                badge_text, badge_bg, badge_fg = "SLEEPING", BG_WIDGET, FG_DIM

            self._awake_badge.configure(text=badge_text, bg=badge_bg, fg=badge_fg)

            person = data.get("person_name") or "Unknown"
            pid    = data.get("person_id")
            self._person_label.configure(
                text=person + (f"  (id={pid})" if pid else "")
            )
            self._session_label.configure(text=str(data.get("session_id") or "—"))

            hz = data.get("loop_hz", {})
            self._hz_label.configure(
                text=f"{hz.get('mean',0):.0f} avg  /  {hz.get('last',0):.0f} last  /  {hz.get('min',0):.0f} min"
            )
            self._queue_label.configure(text=str(data.get("queue_size", 0)))
            self._breath_label.configure(text="yes" if data.get("breathing_active") else "no")
            self._listen_label.configure(text="yes" if data.get("is_listening") else "no")
        else:
            self._awake_badge.configure(text="OFFLINE", bg=BG_WIDGET, fg=FG_DIM)

    def _apply_memories_recent(self, data: Optional[dict]) -> None:
        changed = (data is not None) != self._conn_mem
        self._conn_mem = data is not None
        if changed:
            self._update_statusbar()

        if data:
            mems = data.get("memories", [])
            self._mem_list.delete(0, "end")
            if not mems:
                self._mem_list.insert("end", "  (no memories for current person)")
            for m in mems:
                ts     = _ts_short(m.get("timestamp"))
                text   = m.get("content", "")
                marker = "✓" if m.get("consolidated") else "·"
                self._mem_list.insert("end", f"  {marker} [{ts}]  {text}")

    def _apply_people(self, data: Optional[dict]) -> None:
        changed = (data is not None) != self._conn_people
        self._conn_people = data is not None
        if changed:
            self._update_statusbar()

        if data:
            self._people_tree.delete(*self._people_tree.get_children())
            for person in data.get("people", []):
                name    = person.get("display_name") or person.get("face_label") or "?"
                lseen   = _ts_short(person.get("last_seen"))
                visits  = str(person.get("visit_count", 0))
                summary = (person.get("lt_summary") or "No long-term summary yet")[:120]

                iid = self._people_tree.insert(
                    "", "end",
                    text=f"  {name}",
                    values=(lseen, visits, summary),
                )
                for mem in person.get("recent_memories", []):
                    ts      = _ts_short(mem.get("timestamp"))
                    content = mem.get("content", "")[:120]
                    self._people_tree.insert(
                        iid, "end",
                        text=f"    · {content}",
                        values=(ts, "", ""),
                    )

    def _apply_log(self, data: Optional[dict]) -> None:
        changed = (data is not None) != self._conn_log
        self._conn_log = data is not None
        if changed:
            self._update_statusbar()

        if data:
            lines = data.get("lines", [])
            self._log_text.configure(state="normal")
            self._log_text.delete("1.0", "end")
            self._log_text.insert("end", "\n".join(lines))
            self._log_text.see("end")
            self._log_text.configure(state="disabled")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = MonitorApp(root)
    root.mainloop()
