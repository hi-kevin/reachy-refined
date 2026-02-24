"""MemoryConsolidator: daily LLM job that converts short-term → long-term memories.

Runs a background thread that wakes once per day (default 2:00 AM) and:
  1. Fetches all sessions that have ended and have unconsolidated short-term memories.
  2. Groups them by person.
  3. Calls Gemini generate_content (NOT Live) to write a narrative paragraph per person.
  4. Writes long-term memory entries and marks the source sessions as consolidated.

Uses a separate asyncio event loop (created in the scheduler thread) so it doesn't
interfere with the main asyncio loop running CognitiveBrain.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONSOLIDATION_MODEL = os.getenv("CONSOLIDATION_MODEL_ID", "gemini-2.5-flash")
CONSOLIDATION_HOUR = 2    # 2 AM
CONSOLIDATION_MINUTE = 0


class MemoryConsolidator:
    """Runs a daily Gemini summarisation job over unconsolidated session memories."""

    def __init__(self, memory_server, api_key: str = None):
        self.memory = memory_server
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(
            api_key=self.api_key,
            http_options={"api_version": "v1beta"},
        )
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule(self) -> None:
        """Start the background scheduler thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.debug("MemoryConsolidator: already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._scheduler_loop, daemon=True, name="memory-consolidator"
        )
        self._thread.start()
        logger.info(
            "MemoryConsolidator: scheduled daily at %02d:%02d.",
            CONSOLIDATION_HOUR, CONSOLIDATION_MINUTE,
        )

    def stop(self) -> None:
        """Signal the scheduler thread to stop."""
        self._stop_event.set()
        logger.info("MemoryConsolidator: stop requested.")

    def run_now(self) -> None:
        """Trigger a consolidation pass immediately (blocking, for testing)."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.run_daily())
        finally:
            loop.close()

    # ------------------------------------------------------------------
    # Scheduler thread
    # ------------------------------------------------------------------

    def _scheduler_loop(self) -> None:
        """Wake at CONSOLIDATION_HOUR:CONSOLIDATION_MINUTE each day and consolidate."""
        while not self._stop_event.is_set():
            now = datetime.now()
            target = now.replace(
                hour=CONSOLIDATION_HOUR,
                minute=CONSOLIDATION_MINUTE,
                second=0,
                microsecond=0,
            )
            if now >= target:
                target += timedelta(days=1)

            wait_secs = (target - now).total_seconds()
            logger.info(
                "MemoryConsolidator: next run in %.1f hours (%s).",
                wait_secs / 3600,
                target.strftime("%Y-%m-%d %H:%M"),
            )

            # Sleep in 60-second chunks so we respond quickly to stop()
            elapsed = 0.0
            while elapsed < wait_secs and not self._stop_event.is_set():
                chunk = min(60.0, wait_secs - elapsed)
                time.sleep(chunk)
                elapsed += chunk

            if self._stop_event.is_set():
                break

            # Run consolidation in a fresh event loop
            logger.info("MemoryConsolidator: starting daily consolidation pass.")
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.run_daily())
            except Exception as e:
                logger.error("MemoryConsolidator: unhandled error: %s", e)
            finally:
                loop.close()

        logger.info("MemoryConsolidator: scheduler thread exited.")

    # ------------------------------------------------------------------
    # Consolidation logic
    # ------------------------------------------------------------------

    async def run_daily(self) -> None:
        """Fetch all unconsolidated sessions, group by person, summarise each."""
        try:
            sessions = self.memory.get_unconsolidated_sessions()
        except Exception as e:
            logger.error("MemoryConsolidator: get_unconsolidated_sessions failed: %s", e)
            return

        if not sessions:
            logger.info("MemoryConsolidator: nothing to consolidate.")
            return

        logger.info(
            "MemoryConsolidator: consolidating %d session(s).", len(sessions)
        )

        # Group by person_id (None → 0 for Reachy's own memories)
        by_person: Dict[int, List[dict]] = {}
        for session in sessions:
            pid = session.get("person_id") or 0
            by_person.setdefault(pid, []).append(session)

        for person_id, person_sessions in by_person.items():
            person_name = person_sessions[0].get("person_name") or "Unknown"
            real_person_id = person_sessions[0].get("person_id")  # may be None
            try:
                await self.consolidate_person(real_person_id, person_name, person_sessions)
            except Exception as e:
                logger.error(
                    "MemoryConsolidator: consolidate_person failed for %s: %s",
                    person_name, e,
                )

    async def consolidate_person(
        self,
        person_id: Optional[int],
        person_name: str,
        sessions: List[dict],
    ) -> None:
        """Summarise all session memories for one person into a long-term entry."""
        all_memories: List[str] = []
        session_ids: List[int] = []
        covers_from: Optional[str] = None
        covers_to: Optional[str] = None

        for session in sessions:
            session_ids.append(session["session_id"])
            for mem in session.get("memories", []):
                content = mem.get("content", "").strip()
                ts = mem.get("timestamp", "")
                if content:
                    all_memories.append(f"[{ts[:16]}] {content}")
                if ts:
                    if covers_from is None or ts < covers_from:
                        covers_from = ts
                    if covers_to is None or ts > covers_to:
                        covers_to = ts

        if not all_memories:
            logger.info(
                "MemoryConsolidator: no memories to consolidate for %s.", person_name
            )
            self.memory.mark_consolidated(session_ids)
            return

        memories_text = "\n".join(all_memories)

        if person_id is not None:
            subject = f"a person named {person_name}"
        else:
            subject = "Reachy the robot's own observations and experiences"

        prompt = (
            f"You are helping a friendly robot called Reachy consolidate its memories.\n"
            f"The following are short-term memory notes about {subject}.\n"
            f"Write a concise narrative paragraph (3-5 sentences) capturing the key facts, "
            f"preferences, and notable events that would be useful to remember in future "
            f"conversations. Do not include raw timestamps. Write in third person.\n\n"
            f"Memory notes:\n{memories_text}"
        )

        try:
            response = await self.client.aio.models.generate_content(
                model=CONSOLIDATION_MODEL,
                contents=[types.Part.from_text(text=prompt)],
            )
            summary = (response.text or "").strip()
        except Exception as e:
            logger.error(
                "MemoryConsolidator: Gemini call failed for %s: %s", person_name, e
            )
            return

        if not summary:
            logger.warning(
                "MemoryConsolidator: empty summary for %s — skipping.", person_name
            )
            return

        try:
            self.memory.write_long_term(
                person_id=person_id,
                summary=summary,
                covers_from=covers_from or "",
                covers_to=covers_to or "",
            )
            self.memory.mark_consolidated(session_ids)
            logger.info(
                "MemoryConsolidator: wrote long-term memory for '%s' "
                "covering %d session(s), %d memory notes.",
                person_name, len(sessions), len(all_memories),
            )
        except Exception as e:
            logger.error(
                "MemoryConsolidator: failed to write long-term memory for %s: %s",
                person_name, e,
            )
