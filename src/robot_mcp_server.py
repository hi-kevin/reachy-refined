"""RobotMCPServer: movement and interaction tools for Reachy Mini.

This module is intentionally NOT a standalone FastMCP process — it is a plain
Python class whose async methods are called directly by CognitiveBrain when
Gemini requests a tool.  New tools added here are automatically picked up by
the brain's tool registration without any other changes.

Registered tool names are declared in TOOL_DECLARATIONS so CognitiveBrain can
build the FunctionDeclaration list dynamically.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from reachy_mini.utils import create_head_pose

from .drivers.moves.primitives import AntennaMove, TargetMove

if TYPE_CHECKING:
    from .drivers.moves.core import MovementManager
    from .face_watcher import FaceWatcher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Antenna angle constants (degrees)
# ---------------------------------------------------------------------------
_ANTENNA_UP_DEG = 90.0
_ANTENNA_DOWN_DEG = -90.0

# ---------------------------------------------------------------------------
# Tool schema declarations — consumed by CognitiveBrain to build
# FunctionDeclaration objects for the Gemini session config.
# ---------------------------------------------------------------------------
TOOL_DECLARATIONS: List[Dict[str, Any]] = [
    {
        "name": "move_head",
        "description": (
            "Move Reachy's head to a named direction. "
            "Directions: center, up, down, left, right."
        ),
        "parameters": {
            "direction": {
                "type": "STRING",
                "description": "One of: center, up, down, left, right",
            }
        },
        "required": ["direction"],
    },
    {
        "name": "move_head_precise",
        "description": (
            "Move Reachy's head to exact angles (degrees). "
            "pitch > 0 looks down, yaw > 0 turns right, roll > 0 tilts right."
        ),
        "parameters": {
            "roll": {
                "type": "NUMBER",
                "description": "Roll angle in degrees (-45 to 45).",
            },
            "pitch": {
                "type": "NUMBER",
                "description": "Pitch angle in degrees (-30 up to 30 down).",
            },
            "yaw": {
                "type": "NUMBER",
                "description": "Yaw angle in degrees (-60 left to 60 right).",
            },
            "duration": {
                "type": "NUMBER",
                "description": "Move duration in seconds (default 1.0).",
            },
        },
        "required": ["roll", "pitch", "yaw"],
    },
    {
        "name": "move_antennas",
        "description": (
            "Move Reachy's antennas to specific angles. "
            "90° is fully up, -90° is fully down."
        ),
        "parameters": {
            "right_deg": {
                "type": "NUMBER",
                "description": "Right antenna angle in degrees (-90 to 90).",
            },
            "left_deg": {
                "type": "NUMBER",
                "description": "Left antenna angle in degrees (-90 to 90).",
            },
            "duration": {
                "type": "NUMBER",
                "description": "Move duration in seconds (default 0.5).",
            },
        },
        "required": ["right_deg", "left_deg"],
    },
    {
        "name": "express_emotion",
        "description": (
            "Express an emotion using head pose and antenna positions. "
            "Available emotions: happy, sad, surprised, curious, excited, "
            "sleepy, confused, angry, love."
        ),
        "parameters": {
            "emotion": {
                "type": "STRING",
                "description": (
                    "Emotion name. One of: happy, sad, surprised, curious, "
                    "excited, sleepy, confused, angry, love."
                ),
            }
        },
        "required": ["emotion"],
    },
    {
        "name": "nod",
        "description": "Nod head up and down (yes gesture).",
        "parameters": {
            "times": {
                "type": "INTEGER",
                "description": "Number of nods (default 2).",
            }
        },
        "required": [],
    },
    {
        "name": "shake_head",
        "description": "Shake head left and right (no gesture).",
        "parameters": {
            "times": {
                "type": "INTEGER",
                "description": "Number of shakes (default 2).",
            }
        },
        "required": [],
    },
    {
        "name": "tilt_head",
        "description": "Tilt head to one side (curious/thinking gesture).",
        "parameters": {
            "direction": {
                "type": "STRING",
                "description": "Tilt direction: 'left' or 'right'.",
            },
            "angle": {
                "type": "NUMBER",
                "description": "Tilt angle in degrees (default 20).",
            },
        },
        "required": ["direction"],
    },
    {
        "name": "go_to_sleep",
        "description": (
            "Immediately put Reachy to sleep: lowers antennas, stops breathing, "
            "and closes the Gemini session — bypassing the normal 15-second "
            "no-face timeout. Use this when the user says something like "
            "'go to sleep', 'goodnight', or 'take a rest'."
        ),
        "parameters": {},
        "required": [],
    },
    {
        "name": "get_robot_status",
        "description": "Return the current movement manager status (queue, frequency, pose).",
        "parameters": {},
        "required": [],
    },
]


class RobotMCPServer:
    """Exposes robot movement and interaction as callable async tools."""

    def __init__(
        self,
        movement_manager: "MovementManager",
        face_watcher: Optional["FaceWatcher"] = None,
    ) -> None:
        self.manager = movement_manager
        self.face_watcher = face_watcher

    # ------------------------------------------------------------------
    # Tool dispatcher — single entry point used by CognitiveBrain
    # ------------------------------------------------------------------

    async def dispatch(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Dispatch a tool call by name and return a result string."""
        handlers = {
            "move_head": self._move_head,
            "move_head_precise": self._move_head_precise,
            "move_antennas": self._move_antennas,
            "express_emotion": self._express_emotion,
            "nod": self._nod,
            "shake_head": self._shake_head,
            "tilt_head": self._tilt_head,
            "go_to_sleep": self._go_to_sleep,
            "get_robot_status": self._get_robot_status,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return f"Unknown robot tool: {tool_name}"
        try:
            return await handler(args)
        except Exception as e:
            logger.error("RobotMCPServer: tool %s raised: %s", tool_name, e)
            return f"Tool {tool_name} failed: {e}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_current_pose(self) -> Tuple[Any, Tuple[float, float]]:
        """Return (head_pose_matrix, antennas_tuple) from the robot."""
        head = self.manager.current_robot.get_current_head_pose()
        _, antennas = self.manager.current_robot.get_current_joint_positions()
        return head, tuple(float(a) for a in antennas[:2])

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    async def _move_head(self, args: Dict[str, Any]) -> str:
        direction = str(args.get("direction", "center")).lower()
        presets: Dict[str, Tuple[float, float, float]] = {
            "center":  (0.0,   0.0,  0.0),
            "up":      (0.0, -20.0,  0.0),
            "down":    (0.0,  20.0,  0.0),
            "left":    (0.0,   0.0, -30.0),
            "right":   (0.0,   0.0,  30.0),
            "forward": (0.0,   0.0,  0.0),
        }
        if direction not in presets:
            return f"Unknown direction '{direction}'. Choose: {', '.join(presets)}"
        roll, pitch, yaw = presets[direction]
        return await self._move_head_precise({"roll": roll, "pitch": pitch, "yaw": yaw, "duration": 1.0})

    async def _move_head_precise(self, args: Dict[str, Any]) -> str:
        roll     = float(args.get("roll", 0.0))
        pitch    = float(args.get("pitch", 0.0))
        yaw      = float(args.get("yaw", 0.0))
        duration = float(args.get("duration", 1.0))

        target_pose = create_head_pose(x=0, y=0, z=0, roll=roll, pitch=pitch, yaw=yaw, degrees=True)
        start_pose, start_antennas = self._get_current_pose()

        move = TargetMove(
            start_pose=start_pose,
            start_antennas=start_antennas,
            target_pose=target_pose,
            target_antennas=start_antennas,
            duration=duration,
        )
        self.manager.queue_move(move)
        await asyncio.sleep(duration)
        return f"Moved head to roll={roll}°, pitch={pitch}°, yaw={yaw}°"

    async def _move_antennas(self, args: Dict[str, Any]) -> str:
        right_deg = float(args.get("right_deg", 0.0))
        left_deg  = float(args.get("left_deg", 0.0))
        duration  = float(args.get("duration", 0.5))

        _, current_antennas = self._get_current_pose()
        target = (np.deg2rad(right_deg), np.deg2rad(left_deg))

        move = AntennaMove(
            target_antennas=target,
            duration=duration,
            start_antennas=(current_antennas[0], current_antennas[1]),
        )
        self.manager.queue_move(move)
        await asyncio.sleep(duration)
        return f"Moved antennas to right={right_deg}°, left={left_deg}°"

    async def _express_emotion(self, args: Dict[str, Any]) -> str:
        emotion = str(args.get("emotion", "")).lower()
        # Each entry: (right_antenna_deg, left_antenna_deg, pitch_deg, roll_deg)
        emotions: Dict[str, Tuple[float, float, float, float]] = {
            "happy":     ( 30.0, -30.0, -10.0,  0.0),
            "sad":       (-50.0, -50.0,  20.0,  0.0),
            "surprised": ( 60.0,  60.0,  -5.0,  0.0),
            "curious":   ( 40.0, -10.0,   0.0, 15.0),
            "excited":   ( 80.0, -80.0, -15.0,  0.0),
            "sleepy":    (-70.0, -70.0,  25.0,  0.0),
            "confused":  (  0.0,  40.0,   0.0,-15.0),
            "angry":     ( 20.0,  20.0,  10.0,  0.0),
            "love":      ( 10.0, -10.0,  -5.0,  0.0),
        }
        if emotion not in emotions:
            return f"Unknown emotion '{emotion}'. Choose: {', '.join(emotions)}"

        right_a, left_a, pitch, roll = emotions[emotion]
        duration = 1.5

        start_pose, start_antennas = self._get_current_pose()
        target_pose = create_head_pose(x=0, y=0, z=0, roll=roll, pitch=pitch, yaw=0, degrees=True)
        target_antennas = (np.deg2rad(right_a), np.deg2rad(left_a))

        move = TargetMove(
            start_pose=start_pose,
            start_antennas=start_antennas,
            target_pose=target_pose,
            target_antennas=target_antennas,
            duration=duration,
        )
        self.manager.queue_move(move)
        await asyncio.sleep(duration)
        return f"Expressed emotion: {emotion}"

    async def _nod(self, args: Dict[str, Any]) -> str:
        times = int(args.get("times", 2))
        times = max(1, min(times, 5))

        start_pose, start_antennas = self._get_current_pose()
        pose_down    = create_head_pose(0, 0, 0, 0,  15, 0, degrees=True)
        pose_up      = create_head_pose(0, 0, 0, 0, -10, 0, degrees=True)
        pose_neutral = create_head_pose(0, 0, 0, 0,   0, 0, degrees=True)
        duration = 0.4

        current = start_pose
        for _ in range(times):
            self.manager.queue_move(TargetMove(current, start_antennas, pose_down, start_antennas, duration))
            self.manager.queue_move(TargetMove(pose_down, start_antennas, pose_up, start_antennas, duration))
            current = pose_up
        self.manager.queue_move(TargetMove(current, start_antennas, pose_neutral, start_antennas, duration))

        total = times * 2 * duration + duration
        await asyncio.sleep(total)
        return f"Nodded {times} time(s)"

    async def _shake_head(self, args: Dict[str, Any]) -> str:
        times = int(args.get("times", 2))
        times = max(1, min(times, 5))

        start_pose, start_antennas = self._get_current_pose()
        pose_left    = create_head_pose(0, 0, 0, 0, 0, -20, degrees=True)
        pose_right   = create_head_pose(0, 0, 0, 0, 0,  20, degrees=True)
        pose_neutral = create_head_pose(0, 0, 0, 0, 0,   0, degrees=True)
        duration = 0.4

        current = start_pose
        for _ in range(times):
            self.manager.queue_move(TargetMove(current, start_antennas, pose_left, start_antennas, duration))
            self.manager.queue_move(TargetMove(pose_left, start_antennas, pose_right, start_antennas, duration))
            current = pose_right
        self.manager.queue_move(TargetMove(current, start_antennas, pose_neutral, start_antennas, duration))

        total = times * 2 * duration + duration
        await asyncio.sleep(total)
        return f"Shook head {times} time(s)"

    async def _tilt_head(self, args: Dict[str, Any]) -> str:
        direction = str(args.get("direction", "right")).lower()
        angle     = float(args.get("angle", 20.0))
        roll = angle if direction == "right" else -angle
        return await self._move_head_precise({"roll": roll, "pitch": 0, "yaw": 0, "duration": 1.0})

    async def _go_to_sleep(self, args: Dict[str, Any]) -> str:
        """Immediately trigger sleep — bypasses the 15 s face-timeout."""
        if self.face_watcher is not None:
            logger.info("RobotMCPServer: go_to_sleep called — forcing FaceWatcher to sleep.")
            self.face_watcher.force_sleep()
            return "Going to sleep now. Goodnight!"
        else:
            # Fallback: at least tell MovementManager to sleep
            self.manager.set_sleeping(True)
            return "Sleep mode activated (no FaceWatcher available)."

    async def _get_robot_status(self, args: Dict[str, Any]) -> str:
        status = self.manager.get_status()
        lines = [
            f"queue_size: {status['queue_size']}",
            f"breathing: {status['breathing_active']}",
            f"listening: {status['is_listening']}",
            f"loop_freq_mean: {status['loop_frequency']['mean']:.1f} Hz",
        ]
        return "\n".join(lines)
