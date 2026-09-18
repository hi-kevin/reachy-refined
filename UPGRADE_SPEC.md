# UPGRADE_SPEC.md — Gemini Robotics Modernisation

> **How to use this document.** It is the sole input to a fresh session. Read
> `CLAUDE.md` first for environment and deployment rules, then work the
> workstreams below in the order given in [§8 Sequencing](#8-sequencing).
> [§3 Decisions needed](#3-decisions-needed-before-any-code-changes) must be
> answered by Kevin before W2 starts; everything in W1 can begin immediately.
>
> Written 2026-09-18 against commit `c8862f1` on branch `cognitive-functions`.

---

## 1. Goal

Kevin's stated target for this robot:

> "See and recognize people, face track, talk to them differently based on their
> profile, including learning about them and storing it in a retrievable way."

The architecture for all of that already exists in this repo and is well
factored. The gap is not design — it is that **the long-term half of the memory
system is disconnected and has never actually run**, and the perception and
model layers are one to two generations behind.

---

## 2. Current state

What is built and working:

| Component | State |
|---|---|
| `src/brain/cognitive.py` | Gemini Live session, tool dispatch, person context injection, manual VAD |
| `src/face_watcher.py` | AWAKE/SLEEPING state machine, face tracking, LBPH identification every 3s |
| `src/face_identifier.py` | Haar detect + LBPH recognise, subdirectory-per-person training |
| `src/memory/server.py` | SQLite, FTS5, per-person short-term / long-term schema |
| `src/memory/consolidator.py` | Daily 2 AM job, LLM consolidation ST → LT |
| `src/robot_mcp_server.py` | Movement + identity tools exposed to Gemini |
| `src/drivers/moves/` | 100Hz threaded control loop, primary/secondary blending |

This spec does **not** propose rewriting any of it. `drivers/moves/`,
`local_stream.py`, `face_watcher.py`'s state machine and the memory schema are
sound and should be left structurally alone.

---

## 3. Decisions needed before any code changes

### D1 — The Live model — ✅ RESOLVED 2026-09-18: upgrade

Kevin has replaced the old "DO NOT CHANGE THE MODEL" instruction with a standing
model policy in `CLAUDE.md`: **always use the newest available models, never
downgrade unless explicitly asked.** W2 is unblocked; no further approval needed
to move the Live model forward.

- Current in code: `gemini-3.1-flash-live-preview` (`cognitive.py:22`). As of
  Sept 2026 this is documented as a **legacy preview** model.
- Target: `gemini-3.8-live`, released 2026-09-15. Native speech-to-speech,
  97 languages with mid-conversation switching, and — the reason it matters
  here — **asynchronous (`NON_BLOCKING`) function calling**.

Check for anything newer than `gemini-3.8-live` before pinning it; the policy is
"newest", not "this specific id".

Why async tool calls matter for this specific robot: every tool call today
blocks the conversation. When Gemini calls `analyze_scene`, the person gets
silence for the full round-trip of a separate vision API call. Same for
`recall`, and same for `register_face`, which takes **five photos over ~2.5
seconds** (`face_identifier.py:249`). With `NON_BLOCKING` the robot can say
"hold still a sec" and keep talking while the camera works.

Async tool calls are therefore in scope for W4 — see
[§7](#async-tool-calls-w4).

### D2 — Face recognition replacement (scopes W3)

LBPH is the weakest link and is actively unsafe for a memory robot (see
[B2](#b2--lbph-will-confidently-misidentify-strangers-privacy)). Options:

- **(a) Replace with ArcFace embeddings** (`buffalo_l` ONNX via `onnxruntime`,
  512-dim, cosine similarity). Correct fix. Needs a new dependency on the robot
  and a re-enrolment of existing faces.
- **(b) Tighten LBPH thresholds and add a confirmation step.** Cheap, keeps the
  current model, reduces but does not eliminate misidentification.

Recommend (a). Confirm before starting W3 — it is the only item here that
changes what is installed on the robot.

### D3 — Re-enrolment consent

Whatever W3 decides, faces are biometric data for household members and
visitors. Confirm whether Greachy should ask before storing a new face, and
whether a `forget_me` tool should exist. Cheap now, awkward to retrofit.

---

## 4. Workstream 1 — Bugs that defeat the stated goal (P0)

These are verified by reading the code, not inferred. Fix these before anything
else; W2–W5 are improvements, W1 is the difference between the feature working
and not.

### B1 — Long-term memory has never been written. Not once.

**This is the headline bug.** `session_id` is never propagated from FaceWatcher
to CognitiveBrain, which silently disables the entire long-term memory pipeline.

The chain:

1. `cognitive.py:75` declares `self._active_session_id`, and `:149` defines a
   setter for it.
2. **Nothing ever assigns it.** Grep `active_session_id` across `src/` — it
   appears only inside `cognitive.py`. `FaceWatcher` creates the session
   (`_memory.start_session()`) and keeps the id in its own
   `_current_session_id`, but never pushes it to the brain.
3. So at `cognitive.py:739`, the `remember` handler reads `sid = None` and every
   short-term memory is stored with `session_id = NULL`.
4. `get_unconsolidated_sessions()` (`memory/server.py:343`) selects sessions
   `WHERE EXISTS (SELECT 1 FROM short_term_memories st WHERE st.session_id = s.id ...)`.
   With every `session_id` NULL, **this matches nothing, forever.**
5. The 2 AM consolidator therefore finds zero sessions and writes zero
   long-term memories.
6. `get_person_context()` reads `long_term_memories` — always empty. The robot's
   entire "what I know about you" is the last 5 short-term rows.

**Fix:** assign `brain.active_session_id = session_id` wherever FaceWatcher
starts a session (`_enter_awake`), and clear it in `_enter_sleep`. The setter
already exists and is already lock-guarded.

**Do not stop there.** Confirm the fix end-to-end on the robot: check that
`short_term_memories.session_id` is non-NULL after a conversation, then invoke
the consolidator manually (do not wait for 2 AM) and confirm a row lands in
`long_term_memories`. Consider a backfill for existing NULL-session rows, or
accept the loss and say so — Kevin's call.

### B2 — LBPH will confidently misidentify strangers (privacy)

`face_identifier.py:27` sets `LBPH_THRESHOLD = 80`, and `recognizer.predict()`
**always returns the nearest known label** regardless of whether the face is
actually in the training set. A stranger whose LBPH distance lands under 80 is
identified as whoever they happen to be closest to.

For a robot whose whole purpose is per-person memory, the failure mode is not
"wrong name" — it is **Greachy greeting a stranger by a family member's name and
reciting that family member's private memories to them.**

Short-term mitigation regardless of D2: lower the threshold, and require N
consecutive agreeing identifications before calling `set_active_person`. The
proper fix is W3.

### B3 — Gemini face-recognition fallback is broken two ways

`find_person_by_description()` (`memory/server.py:146`) **ignores its
`description` argument entirely.** It runs
`SELECT ... FROM people WHERE gemini_description IS NOT NULL` and returns every
row. `cognitive.py` then takes `candidates[0]` — i.e. the lowest rowid — and
logs it as an identification. It is not matching anything; it returns the
oldest registered person with a description, every time.

Second bug, same path: the result is **discarded anyway**.
`face_watcher.py:495` fires `identify_unknown_face` via
`asyncio.run_coroutine_threadsafe(...)` and never reads the future. The returned
name goes nowhere.

**Fix:** either wire it up properly — pass the candidate descriptions to the
model and let it choose, then actually consume the result — or delete the path.
A fallback that silently returns the wrong person is worse than no fallback.
Recommend wiring it up using ER-2 (see W4), since that is what it was reaching
for.

### B4 — Reconnects throw away the conversation

`cognitive.py` has a retry loop with exponential backoff (`MAX_RETRIES = 5`),
and it *logs* `session_resumption_update` at `:627`, including the new handle.
But `LiveConnectConfig` (`:359`) sets **neither `session_resumption` nor
`context_window_compression`**, and the handle is never stored or passed back.

Consequences:

- Audio-only Live sessions cap at ~15 minutes and the connection at ~10. This
  app does not stream video to Live (it uses the `analyze_scene` tool with a
  separate vision call), so the 2-minute audio+video cap does not apply — but
  the 10/15-minute ones do.
- On every reconnect the conversation restarts cold: history gone, and the
  startup greeting at `:406` fires again, so **Greachy re-greets the person
  mid-conversation.**

**Fix:** add `context_window_compression` (sliding window) and
`session_resumption`, store `new_handle` from the update messages, pass it on
reconnect, and suppress the startup greeting when resuming rather than starting
fresh.

Verified type names against `google-genai` 2.24.0:

```python
types.ContextWindowCompressionConfig(
    trigger_tokens=None,              # None = server default
    sliding_window=types.SlidingWindow(),
)
types.SessionResumptionConfig(handle=previous_handle)   # fields: handle, transparent
```

---

## 5. Workstream 2 — Model currency

### M1 — The vision model is two generations stale

`src/brain/robotics.py:13` pins `gemini-robotics-er-1.5-preview`.

ER **1.6** was documented for discontinuation at the end of August 2026 — 1.5 is
a generation older than that. Move to **`gemini-robotics-er-2-preview`**.

- Input: text, image, video, audio. Context 131,072 in / 65,536 out.
- Supports function calling, structured outputs, code execution, thinking,
  caching, search grounding.
- Does **not** support the Live API — the non-streaming variant is correct here,
  since `analyze_scene` is a discrete request/response call.

Also review `http_options={"api_version": "v1alpha"}` at `robotics.py:25` — check
whether ER-2 still requires the alpha endpoint or should move to `v1beta`.

### M2 — Live model

`gemini-3.1-flash-live-preview` → `gemini-3.8-live` (`cognitive.py:22`).
Approved under the `CLAUDE.md` model policy — see
[D1](#d1--the-live-model--resolved-2026-09-18-upgrade).

Two things to check as part of this swap, not after it:

- **Voice name.** `cognitive.py:373` pins `"Kore"`. Confirm it is still a valid
  prebuilt voice for 3.8-live before deploying; a bad voice name fails at
  connect.
- **Manual VAD.** `cognitive.py:365` sets
  `automatic_activity_detection(disabled=True)` and `_send_loop` hand-rolls VAD
  from RMS energy (`SPEECH_THRESHOLD = 0.02`, `MIC_GAIN = 3.0`,
  `SILENCE_DURATION = 0.8`). That workaround predates the new model — re-test
  whether server-side VAD now performs better, and delete the hand-rolled path
  if it does.

### M3 — Consolidator model

`memory/consolidator.py:31` uses `gemini-2.5-flash`. Once B1 makes this code
path actually execute, it is worth moving to `gemini-3.8-flash`. Low risk — it
is an offline batch job with no latency constraint. Note the env override
`CONSOLIDATION_MODEL_ID` already exists, so this is a default change only.

---

## 6. Workstream 3 — Identity you can trust

Scope depends on [D2](#d2--face-recognition-replacement-scopes-w3). Assuming (a):

Replace LBPH recognition with **ArcFace embeddings**, keeping everything else:

- Detection stays local and fast. Haar is adequate for the 10Hz tracking loop
  and does not need to change; the tracking loop must stay on-robot regardless,
  because the Live API caps video at 1 FPS and cannot drive head tracking.
- Recognition becomes: crop → 112×112 → ONNX ArcFace (`buffalo_l`) → 512-dim
  vector → cosine similarity against stored vectors.
- **Cosine similarity has a meaningful, tunable "no match" threshold**, which is
  what LBPH lacks and what makes B2 fixable rather than merely mitigable.
- Store vectors in the existing `people` table (new BLOB column). Keep
  `known_faces/` images so re-enrolment is possible after a model change.

Migration: existing `known_faces/{name}/*.jpg` can be re-embedded offline on the
robot — no need to re-photograph anyone. Verify `onnxruntime` installs on the
robot's aarch64 Python before committing to this path.

Keep `face_recognizer.yml` / `label_map.pkl` on disk until the new path is
confirmed working, then clean up.

---

## 7. Workstream 4 — Gemini Robotics ER 2 capability

Only after W1 and W2. The headline framing from the ER-2 launch:

> **ER-2 does not replace the Live conversation loop.** Both
> `gemini-robotics-er-2-preview` and `gemini-robotics-er-2-streaming-preview`
> return **text only**; the streaming docs say to route output to a separate
> TTS. ER-2 is a perception layer beside the voice model, not a replacement for
> it. The two-model split this repo already has is the correct architecture.

Worthwhile ER-2 capabilities for this robot, in value order:

1. **Multi-person disambiguation.** Today `identify_face()` picks the largest
   face by area and nothing else. ER-2's pointing API returns
   `[{"point": [y, x], "label": "..."}]` normalised to a 1000×1000 grid, plus
   `box_2d` bounding boxes — enough to answer "there are three people, which one
   is talking to me" instead of "whoever is closest."
2. **Fix B3 properly.** Description-based re-identification of an unknown face,
   with the candidate list actually passed to the model and the answer actually
   consumed.
3. **Richer greetings.** Scene and appearance context at wake time — what
   someone is wearing, carrying, whether they are waving.

Do **not** adopt ER-2's video progress-tracking / success-failure detection
features. They are aimed at manipulation tasks and have no use here.

### Async tool calls (W4)

Depends on M2 having landed. Re-declare the slow tools as `NON_BLOCKING` and
give each a `FunctionResponseScheduling`:

- `SILENT` — movement and expression tools. The body language is the
  acknowledgement.
- `WHEN_IDLE` — `remember`, `register_face`. Mention at the next natural pause.
- `INTERRUPT` — use sparingly; it is jarring.

**Verified gotcha:** the enum member is **`INTERRUPT`**, not `INTERRUPTED` as
the Live API docs write it. Confirmed against `google-genai` 2.24.0.

---

## 8. Sequencing

```
W1  Bugs          B1 → B4 → B3        (B2 mitigation now, real fix in W3)
W2  Models        M1, M2, M3          (all approved — model policy in CLAUDE.md)
W3  Identity      ArcFace             (blocked on D2)
W4  ER-2          perception + async  (after W1, W2)
W5  Hygiene       dead code           (any time)
```

**B1 first, alone, and verified on the robot before anything else is touched.**
It is a handful of lines, and until it is fixed every other memory improvement
is invisible — there is no long-term memory to improve.

---

## 9. Workstream 5 — Hygiene

Verified unreferenced by any import in `src/`, `dashboard/`, or `monitor_app.py`:

- `src/drivers/gemini_handler.py` — **856 lines**, a second complete Gemini Live
  implementation. Appears to be the reference implementation from
  `gamepop/reachy-mini-gemini` cited in `CLAUDE.md`. It carries its own
  `MODEL = "gemini-3.1-flash-live-preview"` at `:41`, which will confuse any
  future model-migration grep.
- `src/drivers/head_wobbler.py` — 181 lines, unreferenced. Its only consumer
  would be `speech_tapper.py`, whose sole inbound reference is from
  `head_wobbler.py` itself — so the pair is dead together.

Confirm with Kevin before deleting (they may be parked deliberately), and check
`git log` for recent activity on them first.

`PLAN.md` describes the person-aware memory upgrade that has since shipped. Once
this spec's W1 lands, `PLAN.md` is historical — consider archiving it so the two
documents are not mistaken for each other.

---

## 10. Verification rules (from CLAUDE.md — non-negotiable)

- **Never run Python locally.** Windows dev box, no packages installed. All
  checks run on the robot via SSH.
- **The user handles all deployment.** Do not run `deploy.bat` or
  `deploy_and_run.bat`.
- **Never `pkill -f python`** on the robot — it kills the camera pipeline and
  motor controller and needs a physical reboot. Use `scripts/kill_remote.bat`.
- Inspection is fine:
  ```
  ssh pollen@reachy-mini.local "cd ~/reachy-refined && source .venv/bin/activate && <command>"
  ```

For B1 specifically, the verification is a SQL check, not a log line:

```sql
SELECT COUNT(*) FROM short_term_memories WHERE session_id IS NULL;  -- should stop growing
SELECT COUNT(*) FROM long_term_memories;                            -- should become > 0
```

---

## 11. Facts verified for this spec

So the next session does not re-derive them:

- ER-2 model IDs: `gemini-robotics-er-2-preview`,
  `gemini-robotics-er-2-streaming-preview`. Streaming is **text output only**.
  Non-streaming does **not** support the Live API.
- ER 1.6 was scheduled for discontinuation end of August 2026; `er-1.5` in
  `robotics.py` is older still.
- `gemini-3.8-live` released 2026-09-15; async `NON_BLOCKING` tool calls.
- Live API caps video at **1 FPS** — head tracking cannot be delegated to it.
- Live API session caps: ~15 min audio-only, ~2 min audio+video, ~10 min
  connection. This app is audio-only to Live.
- `FunctionResponseScheduling` members: `SILENT`, `WHEN_IDLE`, **`INTERRUPT`**
  (the docs' "INTERRUPTED" is wrong). Verified against `google-genai` 2.24.0.
- `ContextWindowCompressionConfig` fields: `trigger_tokens`, `sliding_window`.
  `SessionResumptionConfig` fields: `handle`, `transparent`. Same SDK version.

Everything in §4 was verified by reading this repo at `c8862f1`. Nothing was
run against the robot — no hardware access from this session — so all runtime
claims are static-analysis conclusions and should be confirmed on-device.
