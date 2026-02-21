# CLAUDE.md — reachy-refined

## Project Overview

Reachy Mini conversational robot companion. Identifies people via face recognition, maintains persistent memories per individual, and holds natural conversations using Google Gemini Live API.

## References
The Reachy SDK: https://github.com/pollen-robotics/reachy_mini

Have a look at this repo for an example of it working: https://github.com/gamepop/reachy-mini-gemini/blob/main/reachy_mini_gemini_app/gemini_handler.py

And the Google API for the model we are using. DO NOT CHANGE THE MODEL. https://docs.cloud.google.com/vertex-ai/generative-ai/docs/reference/rest


## Robot Environment

- **Robot hostname:** `reachy-mini.local`
- **SSH user:** `pollen`
- **Robot project path:** `~/reachy-refined/`
- **Robot venv:** `~/reachy-refined/.venv`
- **Run command on robot:** `cd ~/reachy-refined && source .venv/bin/activate && python -m src.main`

**Critical rules:**
- **Never run Python locally.** All Python must run on the robot via SSH.
- Source code lives locally; deploy to robot before testing.
- For quick ad-hoc tests, SSH directly and run on robot. For permanent code changes, edit locally and deploy via `scripts/deploy.bat`.
- **NEVER run `pkill -f python`, `pkill python`, or any broad process kill on the robot.** The robot OS runs critical system Python processes (camera pipeline, motor controller, GStreamer daemons, etc.). Killing them crashes the robot and requires a physical reboot. To stop only the app, always use `scripts/kill_remote.bat` (which runs `pkill -f 'src.main'`) or the deploy script.

Remember, the local system is windows and none of the packages are installed. Run checks for libraries and other source remotely.

## Deployment Workflow

```
scripts/deploy.bat          # Deploy src/ to robot (kills running process, syncs files, checks encoding)
scripts/deploy_and_run.bat  # Deploy and immediately start the app
scripts/kill_remote.bat     # Kill running src.main process on robot
scripts/setup_remote.bat    # One-time: create venv and install dependencies on robot
```

`deploy.bat` does:
1. `pkill -f 'src.main'` on robot
2. Clears local `__pycache__`
3. `scp -r src/`, `.env`, `requirements.txt`, `scripts/check_encoding.py` to robot
4. Runs encoding check on remote

To run a command on the robot:
```
ssh pollen@reachy-mini.local "cd ~/reachy-refined && source .venv/bin/activate && <command>"
```

## Architecture

```
src/
├── main.py                  # Entry point — wires all components together
├── brain/
│   ├── cognitive.py         # Gemini Live API session, audio I/O, tool dispatch
│   └── robotics.py          # Vision: captures frames, sends to Gemini vision model
├── drivers/
│   ├── local_stream.py      # Bidirectional audio bridge (robot SDK ↔ Gemini)
│   ├── head_wobbler.py      # Audio-driven head movement offsets
│   ├── speech_tapper.py     # Speech-driven sway offsets (VAD, loudness)
│   └── moves/
│       ├── core.py          # 100Hz MovementManager control loop (threaded)
│       ├── primitives.py    # Movement primitives
│       └── types.py         # Type definitions
├── memory/
│   └── server.py            # SQLite FTS5 memory store (remember/recall/get_recent)
├── memory_server.py         # FastMCP tool wrappers for memory
├── audio.py                 # AudioSystem: pyttsx3 TTS + PyAudio wrapper
├── robot.py                 # ReachyMini SDK abstraction layer
└── vision.py                # OpenCV LBPH face detection and recognition
```

**Startup sequence in `main.py`:**
1. `ReachyMini()` — robot SDK
2. `MovementManager(robot)` — start 100Hz control loop
3. `MemoryServer(db_path="memories.db")` — SQLite FTS
4. `RoboticsBrain(robot=robot)` — vision
5. `CognitiveBrain(robotics_brain, memory_server)` — Gemini Live
6. `LocalStream(handler=brain, robot=robot)` — audio bridge
7. `brain.start_up()` — blocks, runs Gemini Live session

## Key Components

### CognitiveBrain (`src/brain/cognitive.py`)
- Model: `gemini-2.5-flash-native-audio-preview-12-2025`
- API version: `v1beta`
- Audio: 16kHz in (mic), 24kHz out (Gemini) → resampled to 16kHz for robot speaker
- Tools exposed to Gemini: `analyze_scene`, `remember`, `recall`
- Async send/receive loops; output queue drops oldest frames if >10 queued

### MovementManager (`src/drivers/moves/core.py`)
- Runs in background thread at 100Hz
- Queues sequential moves; blends primary moves with secondary offsets (speech/face tracking)
- Listening mode freezes antenna position
- Idle breathing behavior when inactive

### MemoryServer (`src/memory/server.py`)
- SQLite with FTS5 virtual table
- `remember(text)` — stores with timestamp
- `recall(query)` — FTS MATCH search
- `get_recent(limit)` — chronological retrieval

### LocalStream (`src/drivers/local_stream.py`)
- Record loop: captures mic → forwards to `brain.receive()`
- Play loop: calls `brain.emit()` → plays on robot speaker
- Handles sample rate resampling

## Environment / Config

- `.env` file (not committed): must contain `GOOGLE_API_KEY`
- `dotenv-template` shows the required format
- `deploy.bat` copies `.env` to the robot on each deploy

## Dependencies (`requirements.txt`)

```
fastmcp
numpy
pyaudio
requests
python-dotenv
google-genai
reachy-mini[gstreamer]
pyttsx3
opencv-contrib-python
```

## Data Files (on robot, not version-controlled)

- `memories.db` — SQLite memory database
- `face_recognizer.yml` — trained LBPH face model
- `label_map.pkl` — face ID → name mapping
- `known_faces/` — face training images

## Branches

- `main` — stable branch
- `cognitive-functions` — current working branch

## Common Operations

**Deploy and test:**
```
scripts\deploy.bat
ssh pollen@reachy-mini.local "cd ~/reachy-refined && source .venv/bin/activate && python -m src.main"
```

**Check logs / run one-off script on robot:**
```
ssh pollen@reachy-mini.local "cd ~/reachy-refined && source .venv/bin/activate && python -c '<code>'"
```

**Kill the running app:**
```
scripts\kill_remote.bat
```

**First-time setup on robot:**
```
scripts\setup_remote.bat
```
