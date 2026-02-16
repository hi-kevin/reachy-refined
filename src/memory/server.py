import sqlite3
import logging
import json
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

DB_PATH = "memories.db"

class MemoryServer:
    """
    Handles long-term memory storage and retrieval using SQLite FTS.
    """
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database with FTS table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                        content,
                        timestamp UNINDEXED
                    );
                """)
                conn.commit()
            logger.info(f"Memory DB initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to init Memory DB: {e}")

    def remember(self, text: str) -> str:
        """Store a new memory."""
        try:
            timestamp = datetime.now().isoformat()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO memories_fts (content, timestamp) VALUES (?, ?)",
                    (text, timestamp)
                )
                conn.commit()
            logger.info(f"Stored memory: {text[:50]}...")
            return "Memory stored."
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return f"Error: {e}"

    def recall(self, query: str, limit: int = 5) -> str:
        """Search memories using FTS."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Use FTS MATCH query
                cursor = conn.execute(
                    "SELECT content, timestamp FROM memories_fts WHERE content MATCH ? ORDER BY rank LIMIT ?",
                    (query, limit)
                )
                rows = cursor.fetchall()
            
            if not rows:
                return "No relevant memories found."
            
            results = [f"[{row[1]}] {row[0]}" for row in rows]
            return "\n".join(results)
        except Exception as e:
            logger.error(f"Error recalling memory: {e}")
            return f"Error: {e}"

    def get_recent(self, limit: int = 5) -> str:
        """Get most recent memories."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT content, timestamp FROM memories_fts ORDER BY rowid DESC LIMIT ?",
                    (limit,)
                )
                rows = cursor.fetchall()

            if not rows:
                return "No memories yet."

            results = [f"[{row[1]}] {row[0]}" for row in rows][::-1] # Reverse to chrono
            return "\n".join(results)
        except Exception as e:
            logger.error(f"Error getting recent memories: {e}")
            return f"Error: {e}"
