# PLAN.md — Person-Aware Memory & Face Recognition Upgrade

## Overview

```
PHASE 1 — src/memory/server.py       Schema overhaul + new MemoryServer methods
PHASE 2 — src/face_identifier.py     New file (replaces src/vision.py for active use)
PHASE 3 — src/face_watcher.py        Inject FaceIdentifier + MemoryServer, add identification loop
PHASE 4 — src/brain/cognitive.py     Person context injection, new tools, person-aware tool dispatch
PHASE 5 — src/memory/consolidator.py New file: daily LLM consolidation job
PHASE 6 — src/robot_mcp_server.py    Three new tool declarations + handlers
PHASE 7 — src/main.py                Wire all new components at startup
```

Dependency order: 1 → 2 → 3 & 4 (parallel) → 5 (any time after 1) → 6 → 7

---

## Phase 1: `src/memory/server.py` — Schema Overhaul

### Migration strategy
All new tables use `CREATE TABLE IF NOT EXISTS`. The existing `memories_fts` table is left intact — the old `remember`/`recall`/`get_recent` methods continue to write to it. New schema is purely additive. Safe to run against both a fresh DB and the existing robot DB.

### New tables to add in `_init_db`

```sql
CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_label TEXT UNIQUE,         -- LBPH label name, e.g. "Dave"
    display_name TEXT,
    gemini_description TEXT,        -- physical description from Gemini fallback
    first_seen TEXT,
    last_seen TEXT,
    visit_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER REFERENCES people(id),
    started_at TEXT,
    ended_at TEXT
);

CREATE TABLE IF NOT EXISTS short_term_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER REFERENCES people(id),  -- NULL = Reachy's own
    session_id INTEGER REFERENCES sessions(id),
    content TEXT,
    timestamp TEXT,
    consolidated INTEGER DEFAULT 0            -- 0=raw, 1=consolidated into LT
);

CREATE TABLE IF NOT EXISTS long_term_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER REFERENCES people(id),  -- NULL = Reachy's own
    summary TEXT,
    covers_from TEXT,
    covers_to TEXT,
    created_at TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_search USING fts5(
    content,
    source UNINDEXED,     -- "short" or "long"
    record_id UNINDEXED,
    person_id UNINDEXED
);
```

**FTS note:** Populate `memories_search` explicitly inside each write method (no triggers). Each insert into `short_term_memories` or `long_term_memories` is followed by an insert into `memories_search`.

**FTS search for `recall_for`:** FTS5 UNINDEXED columns can't filter efficiently in WHERE. Use a join:
```sql
SELECT st.content, st.timestamp
FROM short_term_memories st
JOIN memories_search ms ON ms.source='short' AND CAST(ms.record_id AS INTEGER)=st.id
WHERE ms.content MATCH ?
  AND st.person_id = ?
ORDER BY rank LIMIT ?
```

### New methods (keep all existing methods unchanged)

```python
def get_or_create_person(self, face_label: str, display_name: str = None) -> int:
    # SELECT id FROM people WHERE face_label=?
    # If found: UPDATE last_seen, visit_count++; return id
    # If not found: INSERT; return lastrowid

def update_person_description(self, person_id: int, description: str) -> None:
    # UPDATE people SET gemini_description=? WHERE id=?

def update_person_last_seen(self, person_id: int) -> None:
    # UPDATE people SET last_seen=now, visit_count=visit_count+1 WHERE id=?

def start_session(self, person_id: Optional[int]) -> int:
    # INSERT INTO sessions (person_id, started_at) VALUES (?, now)
    # Returns session_id

def update_session_person(self, session_id: int, person_id: int) -> None:
    # UPDATE sessions SET person_id=? WHERE id=?

def end_session(self, session_id: int) -> None:
    # UPDATE sessions SET ended_at=now WHERE id=?

def remember_for(self, person_id: Optional[int], session_id: Optional[int], content: str) -> str:
    # INSERT into short_term_memories + memories_search
    # Also calls self.remember(content) for legacy compat
    # Returns "Memory stored."

def recall_for(self, person_id: int, query: str, limit: int = 5) -> str:
    # FTS search over memories_search filtered by person_id (join pattern above)
    # Searches both short and long term memories

def get_person_context(self, person_id: int) -> str:
    # Returns formatted string:
    # "You are talking to {display_name} (first seen {first_seen}, {visit_count} visits).
    #  What you remember about them:
    #  {long_term summary if exists}
    #  Recent memories:
    #  {last 5 short_term_memories}"

def get_unconsolidated_sessions(self) -> List[Dict]:
    # SELECT sessions with ended_at IS NOT NULL and unconsolidated ST memories
    # JOIN people for person_name
    # For each session, fetch memories: SELECT content, timestamp FROM short_term_memories
    #   WHERE session_id=? AND consolidated=0 ORDER BY timestamp ASC
    # Returns: [{session_id, person_id, person_name, started_at, ended_at,
    #             memories: [{content, timestamp}]}]

def write_long_term(self, person_id: int, summary: str, covers_from: str, covers_to: str) -> None:
    # INSERT INTO long_term_memories; also INSERT into memories_search with source='long'

def mark_consolidated(self, session_ids: List[int]) -> None:
    # UPDATE short_term_memories SET consolidated=1
    # WHERE session_id IN ({placeholders})

def find_person_by_description(self, description: str) -> List[Dict]:
    # SELECT id, face_label, display_name, gemini_description FROM people
    # WHERE gemini_description IS NOT NULL
    # Returns all rows; caller does LLM-side matching
    # (simple implementation — future improvement: semantic similarity)
```

---

## Phase 2: `src/face_identifier.py` — New File

**Do NOT delete `src/vision.py`** — leave it untouched. The new `FaceIdentifier` is a separate, production-ready replacement.

### Constants
```python
KNOWN_FACES_DIR = "known_faces"           # subdirs per person: known_faces/{name}/*.jpg
RECOGNIZER_MODEL_PATH = "face_recognizer.yml"
LABEL_MAP_PATH = "label_map.pkl"
LBPH_THRESHOLD = 80                       # distance < this → known person
HAAR_SCALE = 1.1
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_SIZE = (40, 40)
```

### Class `FaceIdentifier`

**`__init__`:** Create `known_faces/`, init Haar cascade, init LBPH recognizer, call `load_model()`.

**`load_model()`:** Load `.yml` and `.pkl` if they exist. Log count.

**`save_model()`:** Save recognizer + pickle label_map.

**`train_model() -> str`:**
- Walk `known_faces/`:
  - **Subdirectory mode** (primary): for each subdir, `name = subdir.name`, iterate `.jpg/.png` inside.
  - **Flat file mode** (backward compat): for each `Name_0.jpg` file, `name = filename.split('_')[0]`.
- Read each file as grayscale. Build `faces` list and `labels` list and `label_map`.
- If images found: `self.recognizer.train(faces, np.array(labels))`, `save_model()`.
- Return summary string.

**`identify_face(frame: np.ndarray) -> dict`:**
```
Returns always-consistent dict:
{
  "found": bool,
  "name": str,           # "Unknown" if not found or distance >= threshold
  "confidence": float,   # LBPH distance (lower = better; 999.0 if no face)
  "face_crop": ndarray,  # BGR (for encoding/display); None if no face
  "face_gray": ndarray,  # grayscale (for training); None if no face
  "bbox": (x,y,w,h),    # None if no face
  "is_known": bool,
}
```
- Run Haar detection; if no faces → return not-found dict.
- Pick largest face by area.
- If `label_map` not empty and recognizer available: run `self.recognizer.predict(face_gray)`.
- `is_known = distance < LBPH_THRESHOLD`.

**`capture_training_images(robot, name: str, count: int = 5) -> str`:**
- Create `known_faces/{name}/` dir.
- Loop `count` times:
  - `frame = robot.media.get_frame()`
  - Run Haar detect on frame; skip if no face.
  - Save `face_gray` crop as `known_faces/{name}/{name}_{timestamp_ms}.jpg`.
  - `time.sleep(0.5)`.
- Call `train_model()`.
- Return summary string with count of images captured.

**`register_single_frame(frame: np.ndarray, name: str) -> str`:**
- Run `identify_face` on frame to get the gray crop.
- Save to `known_faces/{name}/`.
- Call `train_model()`.

---

## Phase 3: `src/face_watcher.py` — Identity + Session Tracking

### Constructor signature change
```python
def __init__(
    self,
    robot: Any,
    movement_manager: Any,
    brain: Any = None,
    face_identifier: Any = None,    # NEW
    memory_server: Any = None,      # NEW
):
```
Store as `self._identifier` and `self._memory`.

### New module-level constant
```python
IDENTIFY_INTERVAL_S = 3.0   # seconds between LBPH calls while AWAKE
FALLBACK_COOLDOWN_S = 30.0  # min seconds between Gemini fallback calls
```

### New instance variables (add to `__init__`)
```python
self._current_person_id: Optional[int] = None
self._current_person_name: str = "Unknown"
self._current_session_id: Optional[int] = None
self._last_identified_name: str = ""
self._identify_lock = threading.Lock()
self._last_identify_time: float = 0.0
self._last_fallback_time: float = 0.0   # rate-limit Gemini fallback
self._event_loop = None                 # set by main.py after init
```

### New thread-safe properties
```python
@property
def current_person_id(self) -> Optional[int]:
    with self._identify_lock:
        return self._current_person_id

@property
def current_person_name(self) -> str:
    with self._identify_lock:
        return self._current_person_name
```

### `_detect_face` — add identification branch
After the existing `found = len(faces) > 0` and tracking block, add at the bottom:
```python
# Person identification (AWAKE only, debounced)
if is_awake and found and self._identifier is not None:
    now = time.monotonic()
    if now - self._last_identify_time >= IDENTIFY_INTERVAL_S:
        self._last_identify_time = now
        self._run_identification(frame)
```

### New method `_run_identification(frame)`
```python
def _run_identification(self, frame: np.ndarray) -> None:
    try:
        result = self._identifier.identify_face(frame)
        if not result["found"]:
            return
        if result["is_known"]:
            self._handle_known_person(result["name"], result)
        else:
            # Gemini fallback — rate-limited, fire-and-forget
            now = time.monotonic()
            if (self._brain is not None
                    and hasattr(self._brain, "identify_unknown_face")
                    and now - self._last_fallback_time >= FALLBACK_COOLDOWN_S):
                self._last_fallback_time = now
                crop = result.get("face_crop")
                if crop is not None and self._event_loop is not None:
                    import base64, cv2 as _cv2
                    _, buf = _cv2.imencode(".jpg", crop)
                    face_b64 = base64.b64encode(buf.tobytes()).decode()
                    asyncio.run_coroutine_threadsafe(
                        self._brain.identify_unknown_face(face_b64),
                        self._event_loop
                    )
    except Exception as e:
        logger.warning("_run_identification error: %s", e)
```

### New method `_handle_known_person(name, result)`
```python
def _handle_known_person(self, name: str, result: dict) -> None:
    with self._identify_lock:
        name_changed = (name != self._last_identified_name)
        self._current_person_name = name
        self._last_identified_name = name

    if name_changed and self._memory is not None:
        person_id = self._memory.get_or_create_person(face_label=name)
        with self._identify_lock:
            self._current_person_id = person_id
        # Link person to current session
        with self._identify_lock:
            session_id = self._current_session_id
        if session_id is not None:
            self._memory.update_session_person(session_id, person_id)
        # Inject context into brain
        if self._brain is not None and hasattr(self._brain, "set_active_person"):
            context_str = self._memory.get_person_context(person_id) if self._memory else ""
            self._brain.set_active_person(person_id, name, context_str)
        logger.info("FaceWatcher: identified '%s' (id=%s)", name, person_id)
    elif self._memory is not None:
        with self._identify_lock:
            pid = self._current_person_id
        if pid is not None:
            self._memory.update_person_last_seen(pid)
```

### `_enter_awake` — add session start (append after `self._brain.wake()`)
```python
if self._memory is not None:
    session_id = self._memory.start_session(person_id=None)
    with self._identify_lock:
        self._current_session_id = session_id
        self._current_person_id = None
        self._current_person_name = "Unknown"
        self._last_identified_name = ""
    self._last_identify_time = 0.0   # force immediate LBPH on first AWAKE frame
    logger.info("FaceWatcher: started session %d", session_id)
```

### `_enter_sleep` — add session end (prepend before `self._brain.sleep()`)
```python
if self._memory is not None:
    with self._identify_lock:
        session_id = self._current_session_id
        self._current_session_id = None
        self._current_person_id = None
        self._current_person_name = "Unknown"
        self._last_identified_name = ""
    if session_id is not None:
        self._memory.end_session(session_id)
        logger.info("FaceWatcher: ended session %d", session_id)
```

---

## Phase 4: `src/brain/cognitive.py` — Person Context + New Tools

### New instance variables in `__init__`
```python
self._active_person_id: Optional[int] = None
self._active_person_name: Optional[str] = None
self._active_person_context: str = ""
self._active_session_id: Optional[int] = None
self._person_lock = threading.Lock()
```
Add `import threading` if not already present (it's not in current cognitive.py).

### New public methods

**`set_active_person(self, person_id, name, context_str)`** — called from FaceWatcher thread:
```python
def set_active_person(self, person_id: int, name: str, context_str: str) -> None:
    with self._person_lock:
        self._active_person_id = person_id
        self._active_person_name = name
        self._active_person_context = context_str
    logger.info("CognitiveBrain: active person → '%s' (id=%d)", name, person_id)
    # If session already open, inject context as a text turn now
    if self._session is not None and self._session_active and self._event_loop:
        asyncio.run_coroutine_threadsafe(
            self._inject_person_context_message(context_str, name),
            self._event_loop
        )
```

Store the loop ref: add `self._event_loop = None` to `__init__` and set it in `start_up()` before the first `await`:
```python
async def start_up(self):
    self._event_loop = asyncio.get_event_loop()
    ...
```

**`_inject_person_context_message(context_str, name)`** — async, injects into live session:
```python
async def _inject_person_context_message(self, context_str: str, name: str) -> None:
    if not self._session or not self._session_active:
        return
    try:
        text = f"[Context update: You are now speaking with {name}. {context_str}]"
        await self._session.send(input=text, end_of_turn=True)
        logger.info("Injected person context for '%s' into live session.", name)
    except Exception as e:
        logger.warning("Failed to inject person context: %s", e)
```

**`identify_unknown_face(face_crop_b64)`** — async, Gemini vision fallback:
```python
async def identify_unknown_face(self, face_crop_b64: str) -> Optional[str]:
    if not self.robotics_brain or not self.memory_server:
        return None
    try:
        import base64
        image_bytes = base64.b64decode(face_crop_b64)
        description = await self.robotics_brain.analyze_scene(
            image_bytes,
            "Describe this person's appearance briefly: hair color, approximate age, "
            "glasses, distinctive features. Be concise (1-2 sentences)."
        )
        if not description:
            return None
        candidates = self.memory_server.find_person_by_description(description)
        if candidates:
            name = candidates[0].get("display_name") or candidates[0].get("face_label")
            logger.info("Gemini fallback identified: %s", name)
            return name
        logger.info("Gemini fallback: unknown face described as: %s", description[:80])
        return None
    except Exception as e:
        logger.warning("identify_unknown_face error: %s", e)
        return None
```

**`active_session_id` property** (for FaceWatcher to write to):
```python
@property
def active_session_id(self) -> Optional[int]:
    with self._person_lock:
        return self._active_session_id

@active_session_id.setter
def active_session_id(self, value: Optional[int]) -> None:
    with self._person_lock:
        self._active_session_id = value
```

### `_run_session` — dynamic system prompt
Replace the hardcoded `system_instr` string:
```python
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

system_instr = f"""You are Reachy, a robot companion.
You are cheerful, curious, and warm.
You can see the world by calling the analyze_scene tool.
You have a long-term memory. Use 'remember' to save important details.
Use 'recall' for general search. Use 'get_memories_for_me' for person-specific memories.
If the user tells you their name for the first time, call 'register_me' to register their face.
Be concise in your spoken responses.{person_section}"""
```

### New tool declarations in `builtin_declarations`
Add after the existing three (analyze_scene, remember, recall):
```python
types.FunctionDeclaration(
    name="register_me",
    description=(
        "Register the current speaker's face so Reachy can recognise them next time. "
        "Call this when the user tells you their name for the first time."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={"name": types.Schema(type="STRING", description="The person's name.")},
        required=["name"]
    )
),
types.FunctionDeclaration(
    name="get_memories_for_me",
    description="Retrieve memories specific to the current person.",
    parameters=types.Schema(
        type="OBJECT",
        properties={"query": types.Schema(type="STRING", description="What to search for.")},
        required=["query"]
    )
),
```

### `_receive_loop` tool dispatch changes

Replace `remember` handler (person-aware):
```python
elif fc.name == "remember":
    content = args.get("content", "")
    if self.memory_server:
        with self._person_lock:
            pid = self._active_person_id
            sid = self._active_session_id
        result_text = self.memory_server.remember_for(pid, sid, content)
    else:
        result_text = "Memory unavailable."
```

Replace `recall` handler (person-aware):
```python
elif fc.name == "recall":
    query = args.get("query", "")
    if self.memory_server:
        with self._person_lock:
            pid = self._active_person_id
        result_text = (self.memory_server.recall_for(pid, query)
                       if pid else self.memory_server.recall(query))
    else:
        result_text = "Memory unavailable."
```

Add before the `elif self.robot_mcp is not None:` fallback:
```python
elif fc.name == "register_me":
    name = args.get("name", "")
    if self.robot_mcp:
        result_text = await self.robot_mcp.dispatch("register_face", {"name": name})
    else:
        result_text = "Registration unavailable."

elif fc.name == "get_memories_for_me":
    query = args.get("query", "")
    if self.memory_server:
        with self._person_lock:
            pid = self._active_person_id
            pname = self._active_person_name
        if pid:
            result_text = self.memory_server.recall_for(pid, query)
        else:
            result_text = "I don't know who you are yet. Please tell me your name."
    else:
        result_text = "Memory unavailable."
```

---

## Phase 5: `src/memory/consolidator.py` — New File

```python
CONSOLIDATION_MODEL = os.getenv("CONSOLIDATION_MODEL_ID", "gemini-2.5-flash")
CONSOLIDATION_HOUR = 2   # 2 AM daily
```

### Class `MemoryConsolidator`

**`__init__(self, memory_server, api_key=None)`**
- Store `self.memory`, init `genai.Client` with `v1beta`.
- `self._stop_event = threading.Event()`
- `self._thread = None`

**`async consolidate_person(self, person_id, person_name, sessions) -> None`**
- Collect all memories across sessions; find `covers_from` / `covers_to` timestamps.
- Build prompt asking Gemini to write a 3-5 sentence narrative about this person.
- Call `await self.client.aio.models.generate_content(model=CONSOLIDATION_MODEL, contents=[...])`.
- If response.text: call `self.memory.write_long_term(...)` then `self.memory.mark_consolidated(session_ids)`.
- Log and continue on any error (never crash the scheduler).

**`async run_daily(self) -> None`**
- `sessions = self.memory.get_unconsolidated_sessions()`
- Group by `person_id` (None → key 0 for Reachy's own).
- For each person: `await self.consolidate_person(person_id, person_name, sessions)`.

**`schedule(self) -> None`**
- Start daemon thread `_scheduler_loop`.

**`_scheduler_loop(self) -> None`**
- Calculate seconds until next 2:00 AM. Sleep in 60-second increments (checking `_stop_event`).
- When it's time: create a fresh `asyncio.new_event_loop()`, run `run_daily()`, close the loop.
- Reschedule for next day.

**`stop(self) -> None`** — set `_stop_event`.

---

## Phase 6: `src/robot_mcp_server.py` — Three New Tools

### Constructor change
```python
def __init__(
    self,
    movement_manager,
    face_watcher=None,
    face_identifier=None,    # NEW
    memory_server=None,      # NEW
):
    ...
    self.face_identifier = face_identifier
    self.memory_server = memory_server
```

### Three new entries in `TOOL_DECLARATIONS`
```python
{
    "name": "register_face",
    "description": "Capture photos and register the current person's face. Takes 'name' param.",
    "parameters": {"name": {"type": "STRING", "description": "Person's name."}},
    "required": ["name"],
},
{
    "name": "who_am_i_talking_to",
    "description": "Return the name and recognition status of the person currently in view.",
    "parameters": {},
    "required": [],
},
{
    "name": "my_memories",
    "description": "Return Reachy's memory context for the current person.",
    "parameters": {},
    "required": [],
},
```

### Three new handler methods

**`_register_face`**: Use `await asyncio.to_thread(self.face_identifier.capture_training_images, robot, name, 5)` to avoid blocking. Also call `self.memory_server.get_or_create_person(face_label=name, display_name=name)`.

**`_who_am_i_talking_to`**: Read `self.face_watcher.current_person_name` and `current_person_id`. Return formatted string.

**`_my_memories`**: Get `pid = self.face_watcher.current_person_id`, call `self.memory_server.get_person_context(pid)`.

---

## Phase 7: `src/main.py` — Wire Everything

### Import additions (both try/except branches)
```python
from src.face_identifier import FaceIdentifier
from src.memory.consolidator import MemoryConsolidator
```

### New globals
```python
consolidator = None
identifier = None
```

### `main()` additions (in order)

After step 2 (MovementManager):
```python
# 2b. FaceIdentifier (LBPH)
identifier = FaceIdentifier()
```

After step 3 (MemoryServer):
```python
# 3b. Memory consolidator (daily LLM job)
consolidator = MemoryConsolidator(memory_server=memory)
consolidator.schedule()
```

Change step 6 (FaceWatcher):
```python
watcher = FaceWatcher(
    robot=robot,
    movement_manager=moves,
    brain=brain,
    face_identifier=identifier,
    memory_server=memory,
)
watcher.start()
# Give FaceWatcher the event loop so it can fire async callbacks from its thread
loop = asyncio.get_event_loop()
watcher._event_loop = loop
```

Change step 6b (RobotMCPServer):
```python
robot_mcp = RobotMCPServer(
    movement_manager=moves,
    face_watcher=watcher,
    face_identifier=identifier,
    memory_server=memory,
)
brain.robot_mcp = robot_mcp
```

### `shutdown()` addition
```python
if consolidator:
    consolidator.stop()
```

---

## Files Created/Modified

| File | Action |
|------|--------|
| `src/memory/server.py` | Edit: additive schema + ~12 new methods |
| `src/face_identifier.py` | **NEW** |
| `src/face_watcher.py` | Edit: constructor + 3 new methods + 2 modified methods |
| `src/brain/cognitive.py` | Edit: constructor + 4 new methods + tool dispatch |
| `src/memory/consolidator.py` | **NEW** |
| `src/robot_mcp_server.py` | Edit: constructor + 3 new tools + 3 handlers |
| `src/main.py` | Edit: imports + init + wiring |
| `src/vision.py` | **Leave untouched** |

---

## Key Design Decisions

- **LBPH every 3s when AWAKE** — fast on robot CPU, debounced to avoid spam
- **Gemini fallback rate-limited to 30s** — avoids API hammering on every unrecognised frame
- **`asyncio.run_coroutine_threadsafe` with stored loop ref** — FaceWatcher runs in a thread; the main asyncio loop is passed in via `watcher._event_loop = loop` in `main.py`
- **`capture_training_images` wrapped in `asyncio.to_thread`** — it calls `time.sleep` + camera I/O, must not block the async event loop
- **Old `memories_fts` table preserved** — zero migration risk; new tables are additive
- **Consolidation uses `asyncio.new_event_loop()`** — the scheduler thread is separate from the main loop
- **`set_active_person` is synchronous** — called from FaceWatcher's thread; async injection happens via `run_coroutine_threadsafe`
- **System prompt rebuilt per session** — context injected at `_run_session` start; mid-session updates via `_inject_person_context_message` text turn
