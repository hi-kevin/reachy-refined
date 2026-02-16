from fastmcp import FastMCP
import sqlite3
import datetime
import os

# Initialize FastMCP
mcp = FastMCP("ReachyMemory")

# Database Setup
DB_PATH = "memories.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # People/Entities table
    c.execute('''CREATE TABLE IF NOT EXISTS entities
                 (name TEXT PRIMARY KEY, created_at TEXT)''')
    
    # Memories table
    # type: 'SHORT_TERM' or 'LONG_TERM'
    c.execute('''CREATE TABLE IF NOT EXISTS memories
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  entity_name TEXT,
                  text TEXT,
                  memory_type TEXT,
                  timestamp TEXT,
                  FOREIGN KEY(entity_name) REFERENCES entities(name))''')
    
    conn.commit()
    conn.close()

init_db()

@mcp.tool()
def add_memory(entity_name: str, text: str):
    """
    Adds a new memory for a person or the robot.
    By default, new memories are SHORT_TERM.
    entity_name: Name of the person (e.g. "Dave") or "ROBOT".
    text: The content of the memory.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Ensure entity exists
    c.execute("INSERT OR IGNORE INTO entities (name, created_at) VALUES (?, ?)", 
              (entity_name, datetime.datetime.now().isoformat()))
    
    # Insert memory
    c.execute("INSERT INTO memories (entity_name, text, memory_type, timestamp) VALUES (?, ?, ?, ?)",
              (entity_name, text, "SHORT_TERM", datetime.datetime.now().isoformat()))
    
    conn.commit()
    conn.close()
    return f"Memory added for {entity_name}."

@mcp.tool()
def get_memories(entity_name: str, memory_type: str = "ALL") -> str:
    """
    Retrieves memories for a specific entity.
    memory_type: 'SHORT_TERM', 'LONG_TERM', or 'ALL' (default).
    Returns a formatted string of memories.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    query = "SELECT timestamp, memory_type, text FROM memories WHERE entity_name = ?"
    params = [entity_name]
    
    if memory_type != "ALL":
        query += " AND memory_type = ?"
        params.append(memory_type)
        
    query += " ORDER BY timestamp DESC"
    
    c.execute(query, tuple(params))
    rows = c.fetchall()
    conn.close()
    
    if not rows:
        return f"No memories found for {entity_name}."
    
    result = f"Memories for {entity_name}:\n"
    for ts, m_type, text in rows:
        result += f"[{ts}] ({m_type}): {text}\n"
    
    return result

@mcp.tool()
def list_people() -> list[str]:
    """
    Lists all known people (excluding 'ROBOT' if desired, but here we list all).
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name FROM entities")
    rows = c.fetchall()
    conn.close()
    return [r[0] for r in rows]

@mcp.tool()
def consolidate_memories() -> str:
    """
    Moves SHORT_TERM memories to LONG_TERM.
    In a real system, this would use an LLM to summarize.
    Here, we simply change the type and concatenate generic info.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Find all SHORT_TERM memories
    c.execute("SELECT id, text FROM memories WHERE memory_type = 'SHORT_TERM'")
    rows = c.fetchall()
    
    count = 0
    for m_id, text in rows:
        # For now, just flip the switch. 
        # Ideally: Summarize multiple short-term into one long-term.
        c.execute("UPDATE memories SET memory_type = 'LONG_TERM' WHERE id = ?", (m_id,))
        count += 1
        
    conn.commit()
    conn.close()
    return f"Consolidated {count} memories from Short-term to Long-term."

if __name__ == "__main__":
    mcp.run()
