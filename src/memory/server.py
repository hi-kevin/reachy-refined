import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

DB_PATH = "memories.db"


class MemoryServer:
    """
    Person-aware memory store with short-term and long-term memory.

    Schema:
      people               — face registry with display names and descriptions
      sessions             — one row per AWAKE episode
      short_term_memories  — raw per-session memories (consolidated=0/1)
      long_term_memories   — LLM-written narrative summaries per person
      memories_search      — FTS5 index over short + long term content
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    # DB initialisation
    # ------------------------------------------------------------------

    def _init_db(self):
        """Create all tables if they don't exist (safe to run against existing DB)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript("""
                    -- People registry
                    CREATE TABLE IF NOT EXISTS people (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        face_label TEXT UNIQUE,
                        display_name TEXT,
                        gemini_description TEXT,
                        first_seen TEXT,
                        last_seen TEXT,
                        visit_count INTEGER DEFAULT 0
                    );

                    -- One row per AWAKE episode
                    CREATE TABLE IF NOT EXISTS sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        person_id INTEGER REFERENCES people(id),
                        started_at TEXT,
                        ended_at TEXT
                    );

                    -- Raw per-session memories
                    CREATE TABLE IF NOT EXISTS short_term_memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        person_id INTEGER REFERENCES people(id),
                        session_id INTEGER REFERENCES sessions(id),
                        content TEXT,
                        timestamp TEXT,
                        consolidated INTEGER DEFAULT 0
                    );

                    -- LLM-written narrative summaries
                    CREATE TABLE IF NOT EXISTS long_term_memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        person_id INTEGER REFERENCES people(id),
                        summary TEXT,
                        covers_from TEXT,
                        covers_to TEXT,
                        created_at TEXT
                    );

                    -- Unified FTS index (populated explicitly by write methods)
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_search USING fts5(
                        content,
                        source UNINDEXED,
                        record_id UNINDEXED,
                        person_id UNINDEXED
                    );
                """)
            logger.info("Memory DB initialised at %s", self.db_path)
        except Exception as e:
            logger.error("Failed to init Memory DB: %s", e)

    # ------------------------------------------------------------------
    # People registry
    # ------------------------------------------------------------------

    def get_or_create_person(self, face_label: str, display_name: str = None) -> int:
        """Return person_id for face_label, creating a new record if needed.

        Also bumps visit_count and last_seen on every call for existing people.
        """
        now = datetime.now().isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT id FROM people WHERE face_label = ?", (face_label,)
                ).fetchone()
                if row:
                    person_id = row[0]
                    conn.execute(
                        "UPDATE people SET last_seen=?, visit_count=visit_count+1"
                        " WHERE id=?",
                        (now, person_id),
                    )
                    return person_id
                # New person
                dn = display_name or face_label
                cur = conn.execute(
                    "INSERT INTO people (face_label, display_name, first_seen, last_seen, visit_count)"
                    " VALUES (?, ?, ?, ?, 1)",
                    (face_label, dn, now, now),
                )
                logger.info("Created new person record: %s (id=%d)", dn, cur.lastrowid)
                return cur.lastrowid
        except Exception as e:
            logger.error("get_or_create_person error: %s", e)
            raise

    def update_person_description(self, person_id: int, description: str) -> None:
        """Store the Gemini-generated physical description for future fallback matching."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE people SET gemini_description=? WHERE id=?",
                    (description, person_id),
                )
        except Exception as e:
            logger.error("update_person_description error: %s", e)

    def update_person_last_seen(self, person_id: int) -> None:
        """Bump last_seen and visit_count without creating a full person record."""
        now = datetime.now().isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE people SET last_seen=?, visit_count=visit_count+1 WHERE id=?",
                    (now, person_id),
                )
        except Exception as e:
            logger.error("update_person_last_seen error: %s", e)

    def find_person_by_description(self, description: str) -> List[Dict]:
        """Return all people that have a stored Gemini description.

        Returns raw rows for the caller (e.g. CognitiveBrain) to compare
        descriptions using LLM-side reasoning.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT id, face_label, display_name, gemini_description"
                    " FROM people WHERE gemini_description IS NOT NULL",
                ).fetchall()
            return [
                {
                    "id": r[0],
                    "face_label": r[1],
                    "display_name": r[2],
                    "gemini_description": r[3],
                }
                for r in rows
            ]
        except Exception as e:
            logger.error("find_person_by_description error: %s", e)
            return []

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(self, person_id: Optional[int]) -> int:
        """Insert a new session row and return its id."""
        now = datetime.now().isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "INSERT INTO sessions (person_id, started_at) VALUES (?, ?)",
                    (person_id, now),
                )
                return cur.lastrowid
        except Exception as e:
            logger.error("start_session error: %s", e)
            raise

    def update_session_person(self, session_id: int, person_id: int) -> None:
        """Link a session to a person once their identity is confirmed."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE sessions SET person_id=? WHERE id=?",
                    (person_id, session_id),
                )
        except Exception as e:
            logger.error("update_session_person error: %s", e)

    def end_session(self, session_id: int) -> None:
        """Mark a session as ended."""
        now = datetime.now().isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE sessions SET ended_at=? WHERE id=?",
                    (now, session_id),
                )
        except Exception as e:
            logger.error("end_session error: %s", e)

    # ------------------------------------------------------------------
    # Person-aware memory write/read
    # ------------------------------------------------------------------

    def remember_for(
        self,
        person_id: Optional[int],
        session_id: Optional[int],
        content: str,
    ) -> str:
        """Store a memory in short_term_memories and the FTS index."""
        now = datetime.now().isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Primary store
                cur = conn.execute(
                    "INSERT INTO short_term_memories"
                    " (person_id, session_id, content, timestamp, consolidated)"
                    " VALUES (?, ?, ?, ?, 0)",
                    (person_id, session_id, content, now),
                )
                record_id = cur.lastrowid
                # FTS index
                conn.execute(
                    "INSERT INTO memories_search (content, source, record_id, person_id)"
                    " VALUES (?, 'short', ?, ?)",
                    (content, str(record_id), str(person_id) if person_id else ""),
                )
            logger.info("Stored ST memory (person=%s, session=%s): %s...",
                        person_id, session_id, content[:50])
            return "Memory stored."
        except Exception as e:
            logger.error("remember_for error: %s", e)
            return f"Error: {e}"

    def recall_for(self, person_id: int, query: str, limit: int = 5) -> str:
        """FTS search over memories for a specific person (short + long term)."""
        pid_str = str(person_id)
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Short-term via FTS + join
                st_rows = conn.execute(
                    """
                    SELECT st.content, st.timestamp, 'recent' as kind
                    FROM short_term_memories st
                    JOIN memories_search ms
                      ON ms.source = 'short'
                     AND CAST(ms.record_id AS INTEGER) = st.id
                    WHERE ms.content MATCH ?
                      AND st.person_id = ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (query, person_id, limit),
                ).fetchall()

                # Long-term via FTS + join
                lt_rows = conn.execute(
                    """
                    SELECT lt.summary, lt.created_at, 'long-term' as kind
                    FROM long_term_memories lt
                    JOIN memories_search ms
                      ON ms.source = 'long'
                     AND CAST(ms.record_id AS INTEGER) = lt.id
                    WHERE ms.content MATCH ?
                      AND lt.person_id = ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (query, person_id, limit),
                ).fetchall()

            rows = lt_rows + st_rows
            if not rows:
                return "No relevant memories found for this person."
            return "\n".join(f"[{r[2]}|{r[1][:10]}] {r[0]}" for r in rows)
        except Exception as e:
            logger.error("recall_for error: %s", e)
            return f"Error: {e}"

    def get_person_context(self, person_id: int) -> str:
        """Build a context string for injection into the Gemini system prompt."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                person = conn.execute(
                    "SELECT display_name, face_label, first_seen, visit_count"
                    " FROM people WHERE id=?",
                    (person_id,),
                ).fetchone()

                if not person:
                    return ""

                display_name, face_label, first_seen, visit_count = person
                name = display_name or face_label or "this person"

                # Latest long-term summary
                lt = conn.execute(
                    "SELECT summary FROM long_term_memories WHERE person_id=?"
                    " ORDER BY created_at DESC LIMIT 1",
                    (person_id,),
                ).fetchone()

                # Recent short-term memories (last 5)
                st_rows = conn.execute(
                    "SELECT content, timestamp FROM short_term_memories"
                    " WHERE person_id=? ORDER BY timestamp DESC LIMIT 5",
                    (person_id,),
                ).fetchall()

            lines = [
                f"You are talking to {name}.",
                f"First met: {first_seen[:10] if first_seen else 'unknown'}.",
                f"Total visits: {visit_count}.",
            ]
            if lt:
                lines.append(f"\nWhat you know about them:\n{lt[0]}")
            if st_rows:
                lines.append("\nRecent memories:")
                for content, ts in reversed(st_rows):
                    lines.append(f"  [{ts[:16]}] {content}")

            return "\n".join(lines)
        except Exception as e:
            logger.error("get_person_context error: %s", e)
            return ""

    # ------------------------------------------------------------------
    # Consolidation support
    # ------------------------------------------------------------------

    def get_unconsolidated_sessions(self) -> List[Dict]:
        """Return sessions that have ended and have un-consolidated ST memories.

        Each entry includes the person's name and a 'memories' list.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                sessions = conn.execute(
                    """
                    SELECT s.id, s.person_id, s.started_at, s.ended_at,
                           p.display_name, p.face_label
                    FROM sessions s
                    LEFT JOIN people p ON s.person_id = p.id
                    WHERE s.ended_at IS NOT NULL
                      AND EXISTS (
                          SELECT 1 FROM short_term_memories st
                          WHERE st.session_id = s.id AND st.consolidated = 0
                      )
                    ORDER BY s.started_at ASC
                    """,
                ).fetchall()

                result = []
                for sid, pid, started, ended, display_name, face_label in sessions:
                    memories = conn.execute(
                        "SELECT content, timestamp FROM short_term_memories"
                        " WHERE session_id=? AND consolidated=0 ORDER BY timestamp ASC",
                        (sid,),
                    ).fetchall()
                    result.append(
                        {
                            "session_id": sid,
                            "person_id": pid,
                            "person_name": display_name or face_label or "Unknown",
                            "started_at": started,
                            "ended_at": ended,
                            "memories": [
                                {"content": m[0], "timestamp": m[1]} for m in memories
                            ],
                        }
                    )
            return result
        except Exception as e:
            logger.error("get_unconsolidated_sessions error: %s", e)
            return []

    def write_long_term(
        self,
        person_id: Optional[int],
        summary: str,
        covers_from: str,
        covers_to: str,
    ) -> None:
        """Insert a long-term memory summary."""
        now = datetime.now().isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "INSERT INTO long_term_memories"
                    " (person_id, summary, covers_from, covers_to, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (person_id, summary, covers_from, covers_to, now),
                )
                record_id = cur.lastrowid
                conn.execute(
                    "INSERT INTO memories_search (content, source, record_id, person_id)"
                    " VALUES (?, 'long', ?, ?)",
                    (summary, str(record_id), str(person_id) if person_id else ""),
                )
            logger.info("Wrote long-term memory for person_id=%s (%d chars)",
                        person_id, len(summary))
        except Exception as e:
            logger.error("write_long_term error: %s", e)

    def mark_consolidated(self, session_ids: List[int]) -> None:
        """Mark all short_term_memories for the given sessions as consolidated."""
        if not session_ids:
            return
        try:
            placeholders = ",".join(["?"] * len(session_ids))
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    f"UPDATE short_term_memories SET consolidated=1"
                    f" WHERE session_id IN ({placeholders})",
                    session_ids,
                )
            logger.info("Marked %d sessions as consolidated.", len(session_ids))
        except Exception as e:
            logger.error("mark_consolidated error: %s", e)
