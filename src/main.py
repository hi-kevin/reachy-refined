import asyncio
import logging
import signal
import sys
import os
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

# Imports
# Add project root to sys.path to allow imports from src
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from reachy_mini import ReachyMini
    from src.drivers.moves import MovementManager
    from src.drivers.local_stream import LocalStream
    from src.brain.cognitive import CognitiveBrain
    from src.brain.robotics import RoboticsBrain
    from src.memory.server import MemoryServer
    from src.face_watcher import FaceWatcher
    from src.robot_mcp_server import RobotMCPServer
except ImportError:
    # Fallback for running directly from src/
    try:
        from drivers.moves import MovementManager
        from drivers.local_stream import LocalStream
        from brain.cognitive import CognitiveBrain
        from brain.robotics import RoboticsBrain
        from memory.server import MemoryServer
        from face_watcher import FaceWatcher
        from robot_mcp_server import RobotMCPServer
        from reachy_mini import ReachyMini
    except ImportError as e:
        logger.error(f"Import Error: {e}")
        sys.exit(1)

robot = None
stream = None
brain = None
moves = None
watcher = None

async def main():
    global robot, stream, brain, moves, watcher

    logger.info("--- Reachy Gemini Refactor Starting ---")

    # 1. Initialize Robot
    logger.info("Initializing ReachyMini...")
    robot = ReachyMini()

    # 2. Initialize Movement Manager (Controls head/antennas)
    #    camera_worker is set after FaceWatcher is created (see below).
    logger.info("Initializing MovementManager...")
    moves = MovementManager(robot)
    moves.start()

    # 3. Initialize Memory Server
    logger.info("Initializing MemoryServer...")
    memory = MemoryServer(db_path="memories.db")

    # 4. Initialize Robotics Brain (Vision) - inject robot for camera access
    logger.info("Initializing RoboticsBrain (Vision)...")
    vision = RoboticsBrain(robot=robot)

    # 5. Initialize Cognitive Brain (Audio/Reasoning) - inject dependencies
    #    RobotMCPServer is created after FaceWatcher (needs it for force_sleep),
    #    so we pass it in after watcher is ready.
    logger.info("Initializing CognitiveBrain (Audio/Reasoning)...")
    brain = CognitiveBrain(robotics_brain=vision, memory_server=memory)

    # 6. Start Face Watcher - gates mic, drives antenna state, and supplies
    #    face-tracking yaw offsets to MovementManager while awake.
    logger.info("Initializing FaceWatcher...")
    watcher = FaceWatcher(robot=robot, movement_manager=moves, brain=brain)
    watcher.start()

    # 6b. Now that FaceWatcher exists, create RobotMCPServer and inject into brain.
    logger.info("Initializing RobotMCPServer...")
    robot_mcp = RobotMCPServer(movement_manager=moves, face_watcher=watcher)
    brain.robot_mcp = robot_mcp

    # Wire FaceWatcher as the camera_worker so MovementManager polls
    # get_face_tracking_offsets() every 100 Hz tick.
    moves.camera_worker = watcher

    # 7. Initialize Audio Stream - connect to brain, gated by face watcher
    logger.info("Initializing LocalStream...")
    stream = LocalStream(handler=brain, robot=robot, face_watcher=watcher)

    # 8. Launch Audio Stream (mic hardware always running; gating is in record_loop)
    logger.info("Launching Audio Stream...")
    stream.launch()

    # 9. Start Brain Loop (Blocks until shutdown)
    logger.info("Starting Brain Loop (Gemini Live)...")
    try:
        await brain.start_up()
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Brain Loop Error: {e}")
    finally:
        logger.info("Shutting down...")

async def shutdown(sig, loop):
    logger.info(f"Received exit signal {sig.name}...")

    if watcher:
        logger.info("Stopping face watcher...")
        watcher.stop()

    if stream:
        logger.info("Closing audio stream...")
        stream.close()

    if brain:
        logger.info("Stopping brain...")
        await brain.shutdown()

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
    
    # Signal handling (Skip on Windows Proactor)
    if os.name == 'nt':
        # Windows Proactor doesn't support add_signal_handler
        # We'll rely on KeyboardInterrupt for basic stopping
        pass
    else:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s, loop)))
            except NotImplementedError:
                # Fallback if loop doesn't support it
                pass

    try:
        loop.run_until_complete(main())
    except Exception as e:
        logger.error(f"Fatal Error: {e}")
    finally:
        loop.close()
