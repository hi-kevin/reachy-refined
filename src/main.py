
import asyncio
import logging
import signal
import sys
import os
import threading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ReachyGemini")

# Suppress noisy loggers
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("websockets.client").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from reachy_mini import ReachyMini
    from src.drivers.moves import MovementManager
    from src.drivers.moves.controller import MovementController
    from src.drivers.gemini_handler import GeminiLiveHandler
except ImportError as e:
    logger.error(f"Import Error: {e}")
    sys.exit(1)

robot = None
moves = None
controller = None
handler = None
stop_event = threading.Event()

async def main():
    global robot, moves, controller, handler, stop_event

    logger.info("--- Reachy Gemini Refactor Starting ---")

    # 1. Initialize Robot
    logger.info("Initializing ReachyMini...")
    robot = ReachyMini()
    
    # 2. Initialize Movement Manager (Controls head/antennas)
    logger.info("Initializing MovementManager...")
    moves = MovementManager(robot)
    moves.start()

    # 3. Initialize Movement Controller (High-level tools)
    logger.info("Initializing MovementController...")
    controller = MovementController(moves)

    # 4. Initialize Gemini Handler
    logger.info("Initializing GeminiLiveHandler...")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables.")
        return

    handler = GeminiLiveHandler(
        api_key=api_key,
        robot=robot,
        movement_controller=controller,
        use_camera=True,
        use_robot_audio=False, # Set to True to use robot mic/speaker if desired, defaults to local for now based on user request "removing code from cognitive and local_stream and using this library instead" - library defaults to local if use_robot_audio=False but it checks robot availability.
        # Wait, the library defaults use_robot_audio=False in init.
        # I'll stick to False (Local) for now to be safe, or True?
        # The user's rule says "Do not run the python locally, run it on the robot."
        # If running ON the robot, we should use robot audio probably.
        # But `local_stream.py` was using `robot.media`.
        # `GeminiLiveHandler` supports `use_robot_audio=True` to use `robot.media`.
        # So I should set it to `True` if running on robot.
        # The user rule: "The reachy robot is running and on the network as reachy-mini.local... Do not run the python locally, run it on the robot."
        # This means the code runs ON the robot.
        # So I should SET use_robot_audio=True.
        # However, `local_stream.py` used `robot.media`.
        # `GeminiLiveHandler` has `use_robot_audio` flag.
        # If `use_robot_audio=True`, it uses `robot.media`.
        # If `False`, it uses `pyaudio` (local mic).
        # Since code runs ON the robot, `pyaudio` would use robot's local mic (if any? maybe USB mic?).
        # But `robot.media` handles the specific hardware abstraction.
        # I'll set `use_robot_audio=True`.
    )
    # Actually, looking at `gemini_handler.py`:
    # if not use_robot_audio and PYAUDIO_AVAILABLE: use pyaudio
    # elif not use_robot_audio and not PYAUDIO_AVAILABLE: fallback to robot audio
    # So if I set True, it uses robot.media.
    # I should set True.

    handler.use_robot_audio = True # Force robot audio
    
    # 5. Run Handler
    logger.info("Starting Gemini Live Handler...")
    try:
        # handler.run is async but it uses a threading.Event for stopping?
        # No, handler.run is `async def run(self, stop_event: threading.Event)`.
        await handler.run(stop_event)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Handler Loop Error: {e}")
    finally:
        logger.info("Shutting down...")

async def shutdown(sig, loop):
    logger.info(f"Received exit signal {sig.name}...")
    stop_event.set()
    
    if moves:
        logger.info("Stopping movement manager...")
        moves.stop()
        
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    if os.name == 'nt':
        # Windows handling
        pass
    else:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s, loop)))
            except NotImplementedError:
                pass

    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Fatal Error: {e}")
    finally:
        loop.close()
