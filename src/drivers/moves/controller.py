
import asyncio
import logging
import random
import numpy as np
from typing import Tuple, Dict

from reachy_mini.utils import create_head_pose
from .core import MovementManager
from .primitives import TargetMove

logger = logging.getLogger(__name__)

class MovementController:
    """High-level controller for Reachy Mini movements."""

    def __init__(self, movement_manager: MovementManager):
        self.manager = movement_manager

    async def move_head(self, direction: str) -> str:
        """Move head to a specific direction."""
        # Define preset directions (roll, pitch, yaw) in degrees
        # pitch: positive looks down
        # yaw: positive turns right
        directions = {
            "center": (0, 0, 0),
            "up": (0, -20, 0),
            "down": (0, 20, 0),
            "left": (0, 0, -30),
            "right": (0, 0, 30),
            "forward": (0, 0, 0),
        }
        
        target = directions.get(direction.lower(), (0, 0, 0))
        roll, pitch, yaw = target
        
        return await self.move_head_precise(roll, pitch, yaw)

    async def move_head_precise(self, roll: float, pitch: float, yaw: float) -> str:
        """Move head to precise angles."""
        try:
            # Create target pose
            target_pose = create_head_pose(
                x=0, y=0, z=0,
                roll=roll, pitch=pitch, yaw=yaw,
                degrees=True
            )
            
            # Get current pose from manager to interpolate from
            # (In a real implementation we might want to get actual current robot pose,
            # but manager's last commanded pose is decent for planning)
            # Actually, let's use the robot's current pose to be safe
            start_pose = self.manager.current_robot.get_current_head_pose()
            _, start_antennas = self.manager.current_robot.get_current_joint_positions()
            
            # Create a move
            move = TargetMove(
                start_pose=start_pose,
                start_antennas=start_antennas,
                target_pose=target_pose,
                target_antennas=start_antennas, # Keep antennas as is
                duration=1.0
            )
            
            self.manager.queue_move(move)
            await asyncio.sleep(1.0) # Wait for move to complete (approx)
            return f"Moved head to roll={roll}, pitch={pitch}, yaw={yaw}"
            
        except Exception as e:
            logger.error(f"Move head error: {e}")
            return f"Failed to move head: {e}"

    async def express_emotion(self, emotion: str) -> str:
        """Express an emotion using head and antennas via presets."""
        emotions = {
            "happy": {"antennas": (30, -30), "pitch": -10},
            "sad": {"antennas": (-50, -50), "pitch": 20},
            "surprised": {"antennas": (60, 60), "pitch": -5},
            "curious": {"antennas": (40, -10), "roll": 15},
            "excited": {"antennas": (80, -80), "pitch": -15},
            "sleepy": {"antennas": (-70, -70), "pitch": 25},
            "confused": {"antennas": (0, 40), "roll": -15},
            "angry": {"antennas": (20, 20), "pitch": 10},
            "love": {"antennas": (10, -10), "pitch": -5},
        }
        
        config = emotions.get(emotion.lower())
        if not config:
            return f"Unknown emotion: {emotion}"
            
        antennas = config.get("antennas", (0, 0))
        pitch = config.get("pitch", 0)
        roll = config.get("roll", 0)
        
        # We can combine these into one move
        try:
            start_pose = self.manager.current_robot.get_current_head_pose()
            _, start_antennas = self.manager.current_robot.get_current_joint_positions()
            
            target_pose = create_head_pose(x=0, y=0, z=0, roll=roll, pitch=pitch, yaw=0, degrees=True)
            
            move = TargetMove(
                start_pose=start_pose,
                start_antennas=start_antennas,
                target_pose=target_pose,
                target_antennas=antennas,
                duration=1.5
            )
            
            self.manager.queue_move(move)
            await asyncio.sleep(1.5)
            return f"Expressed {emotion}"
            
        except Exception as e:
            return f"Failed to express emotion: {e}"

    async def move_antennas(self, right_angle: float, left_angle: float) -> str:
        """Move antennas to specific angles."""
        try:
            start_pose = self.manager.current_robot.get_current_head_pose()
            _, start_antennas = self.manager.current_robot.get_current_joint_positions()
            
            move = TargetMove(
                start_pose=start_pose,
                start_antennas=start_antennas,
                target_pose=start_pose, # Keep head still
                target_antennas=(right_angle, left_angle),
                duration=0.5
            )
            
            self.manager.queue_move(move)
            await asyncio.sleep(0.5)
            return f"Moved antennas to {right_angle}, {left_angle}"
        except Exception as e:
            return f"Failed to move antennas: {e}"

    async def antenna_expression(self, expression: str) -> str:
        """Set antennas to a preset expression."""
        expressions = {
            "neutral": (0, 0),
            "alert": (45, -45),
            "droopy": (-60, -60),
            "asymmetric": (30, -50),
            "perky": (80, 80),
        }
        
        angles = expressions.get(expression.lower())
        if not angles:
            return f"Unknown expression: {expression}"
            
        return await self.move_antennas(*angles)

    async def nod_yes(self, times: int = 2) -> str:
        """Nod head up and down."""
        # This requires a sequence of moves.
        # Down, Up, Down, Up...
        # Since queue_move appends, we can just loop.
        try:
            start_pose = self.manager.current_robot.get_current_head_pose()
            _, start_antennas = self.manager.current_robot.get_current_joint_positions()
            
            pose_down = create_head_pose(0, 0, 0, 0, 15, 0, degrees=True)
            pose_up = create_head_pose(0, 0, 0, 0, -10, 0, degrees=True)
            pose_neutral = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            
            duration = 0.4
            
            current_start_pose = start_pose
            
            for _ in range(times):
                # Down
                self.manager.queue_move(TargetMove(current_start_pose, start_antennas, pose_down, start_antennas, duration))
                # Up
                self.manager.queue_move(TargetMove(pose_down, start_antennas, pose_up, start_antennas, duration))
                current_start_pose = pose_up
                
            # Return to center
            self.manager.queue_move(TargetMove(current_start_pose, start_antennas, pose_neutral, start_antennas, duration))
            
            # Wait roughly for total duration
            await asyncio.sleep(times * 2 * duration + duration)
            return f"Nodded {times} times"
        except Exception as e:
            return f"Failed to nod: {e}"

    async def shake_no(self, times: int = 2) -> str:
        """Shake head left and right."""
        try:
            start_pose = self.manager.current_robot.get_current_head_pose()
            _, start_antennas = self.manager.current_robot.get_current_joint_positions()
            
            pose_left = create_head_pose(0, 0, 0, 0, 0, -20, degrees=True)
            pose_right = create_head_pose(0, 0, 0, 0, 0, 20, degrees=True)
            pose_neutral = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            
            duration = 0.4
            
            current_start_pose = start_pose
            
            for _ in range(times):
                # Left
                self.manager.queue_move(TargetMove(current_start_pose, start_antennas, pose_left, start_antennas, duration))
                # Right
                self.manager.queue_move(TargetMove(pose_left, start_antennas, pose_right, start_antennas, duration))
                current_start_pose = pose_right
                
            # Return to center
            self.manager.queue_move(TargetMove(current_start_pose, start_antennas, pose_neutral, start_antennas, duration))
            
            await asyncio.sleep(times * 2 * duration + duration)
            return f"Shook head {times} times"
        except Exception as e:
            return f"Failed to shake: {e}"

    async def tilt_head(self, direction: str, angle: float = 20) -> str:
        """Tilt head to side."""
        roll = angle if direction == "right" else -angle
        return await self.move_head_precise(roll, 0, 0)

    async def look_at_camera(self) -> str:
        """Reset head to look forward."""
        return await self.move_head("center")

    async def do_dance(self, style: str = "default") -> str:
        """Perform a dance animation."""
        # Simple dance implementation
        return await self.express_emotion("happy")

    async def wake_up(self) -> str:
        """Wake up animation."""
        return await self.move_head("center")

    async def go_to_sleep(self) -> str:
        """Sleep animation."""
        return await self.express_emotion("sleepy")

    async def reset_position(self) -> str:
        """Reset to neutral."""
        return await self.move_head("center")
