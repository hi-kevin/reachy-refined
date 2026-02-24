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

# Add project root to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from reachy_mini import ReachyMini
    from src.drivers.moves import MovementManager
    from src.drivers.local_stream import LocalStream
    from src.brain.cognitive import CognitiveBrain
    from src.brain.robotics import RoboticsBrain
    from src.memory.server import MemoryServer
    from src.memory.consolidator import MemoryConsolidator
    from src.face_watcher import FaceWatcher
    from src.face_identifier import FaceIdentifier
    from src.robot_mcp_server import RobotMCPServer
except ImportError:
    # Fallback for running directly from src/
    try:
        from drivers.moves import MovementManager
        from drivers.local_stream import LocalStream
        from brain.cognitive import CognitiveBrain
        from brain.robotics import RoboticsBrain
        from memory.server import MemoryServer
        from memory.consolidator import MemoryConsolidator
        from face_watcher import FaceWatcher
        from face_identifier import FaceIdentifier
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
consolidator = None


async def main():
    global robot, stream, brain, moves, watcher, consolidator

    logger.info("--- Reachy Gemini Refactor Starting ---")

    # 1. Initialize Robot
    logger.info("Initializing ReachyMini...")
    robot = ReachyMini()

    # 2. Initialize Movement Manager
    logger.info("Initializing MovementManager...")
    moves = MovementManager(robot)
    moves.start()

    # 2b. Initialize FaceIdentifier (LBPH)
    logger.info("Initializing FaceIdentifier...")
    identifier = FaceIdentifier()
    logger.info(
        "FaceIdentifier ready — known people: %s",
        identifier.known_people or "(none yet)",
    )

    # 3. Initialize Memory Server
    logger.info("Initializing MemoryServer...")
    memory = MemoryServer(db_path="memories.db")

    # 3b. Initialize Memory Consolidator (daily LLM job, 2 AM)
    logger.info("Initializing MemoryConsolidator...")
    consolidator = MemoryConsolidator(memory_server=memory)
    consolidator.schedule()

    # 4. Initialize Robotics Brain (Vision)
    logger.info("Initializing RoboticsBrain (Vision)...")
    vision = RoboticsBrain(robot=robot)

    # 5. Initialize Cognitive Brain (Audio/Reasoning)
    #    robot_mcp injected after FaceWatcher is ready (step 6b)
    logger.info("Initializing CognitiveBrain (Audio/Reasoning)...")
    brain = CognitiveBrain(robotics_brain=vision, memory_server=memory)

    # 6. Initialize FaceWatcher with identity and memory support
    logger.info("Initializing FaceWatcher...")
    watcher = FaceWatcher(
        robot=robot,
        movement_manager=moves,
        brain=brain,
        face_identifier=identifier,
        memory_server=memory,
    )
    watcher.start()

    # Pass the running event loop to FaceWatcher so it can fire async callbacks
    # (identify_unknown_face, set_active_person injection) from its thread.
    loop = asyncio.get_event_loop()
    watcher._event_loop = loop

    # Wire FaceWatcher as the camera_worker for 100Hz face-tracking offsets
    moves.camera_worker = watcher

    # 6b. Create RobotMCPServer and inject into brain
    logger.info("Initializing RobotMCPServer...")
    robot_mcp = RobotMCPServer(
        movement_manager=moves,
        face_watcher=watcher,
        face_identifier=identifier,
        memory_server=memory,
    )
    brain.robot_mcp = robot_mcp

    # 7. Initialize Audio Stream
    logger.info("Initializing LocalStream...")
    stream = LocalStream(handler=brain, robot=robot, face_watcher=watcher)

    # 8. Launch Audio Stream
    logger.info("Launching Audio Stream...")
    stream.launch()

    # 9. Start Brain Loop (blocks until shutdown)
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

    if consolidator:
        logger.info("Stopping memory consolidator...")
        consolidator.stop()

    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Signal handling (Skip on Windows Proactor)
    if os.name == 'nt':
        pass
    else:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(
                    sig, lambda s=sig: asyncio.create_task(shutdown(s, loop))
                )
            except NotImplementedError:
                pass

    try:
        loop.run_until_complete(main())
    except Exception as e:
        logger.error(f"Fatal Error: {e}")
    finally:
        loop.close()
