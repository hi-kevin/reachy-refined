from __future__ import annotations
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.motion.move import Move
from reachy_mini.utils.interpolation import (
    compose_world_offset,
    linear_pose_interpolation,
)

from .types import FullBodyPose

class TargetMove(Move):
    """Move to a specific target pose with linear interpolation."""

    def __init__(
        self,
        start_pose: NDArray[np.float64],
        start_antennas: Tuple[float, float],
        target_pose: NDArray[np.float64],
        target_antennas: Tuple[float, float],
        duration: float,
    ):
        """Initialize target move.

        Args:
            start_pose: Starting head pose matrix
            start_antennas: Starting antenna positions
            target_pose: Target head pose matrix
            target_antennas: Target antenna positions
            duration: Duration of the move in seconds
        """
        self.start_pose = start_pose
        self.start_antennas = np.array(start_antennas)
        self.target_pose = target_pose
        self.target_antennas = np.array(target_antennas)
        self._duration = duration

    @property
    def duration(self) -> float:
        """Return move duration."""
        return self._duration

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate move at time t."""
        if t >= self._duration:
            return (self.target_pose, self.target_antennas, 0.0)

        alpha = t / self._duration

        # Interpolate head pose
        head_pose = linear_pose_interpolation(self.start_pose, self.target_pose, alpha)

        # Interpolate antennas
        antennas_interp = (1 - alpha) * self.start_antennas + alpha * self.target_antennas
        antennas = antennas_interp.astype(np.float64)

        return (head_pose, antennas, 0.0)

class BreathingMove(Move):
    """Breathing move with interpolation to neutral and then continuous breathing patterns."""

    def __init__(
        self,
        interpolation_start_pose: NDArray[np.float32],
        interpolation_start_antennas: Tuple[float, float],
        interpolation_duration: float = 1.0,
    ):
        """Initialize breathing move.

        Args:
            interpolation_start_pose: 4x4 matrix of current head pose to interpolate from
            interpolation_start_antennas: Current antenna positions to interpolate from
            interpolation_duration: Duration of interpolation to neutral (seconds)

        """
        self.interpolation_start_pose = interpolation_start_pose
        self.interpolation_start_antennas = np.array(interpolation_start_antennas)
        self.interpolation_duration = interpolation_duration

        # Neutral positions for breathing base
        self.neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self.neutral_antennas = np.array([0.0, 0.0])

        # Breathing parameters
        self.breathing_z_amplitude = 0.005  # 5mm gentle breathing
        self.breathing_frequency = 0.1  # Hz (6 breaths per minute)
        self.antenna_sway_amplitude = np.deg2rad(15)  # 15 degrees
        self.antenna_frequency = 0.5  # Hz (faster antenna sway)

    @property
    def duration(self) -> float:
        """Duration property required by official Move interface."""
        return float("inf")  # Continuous breathing (never ends naturally)

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate breathing move at time t."""
        if t < self.interpolation_duration:
            # Phase 1: Interpolate to neutral base position
            interpolation_t = t / self.interpolation_duration

            # Interpolate head pose
            head_pose = linear_pose_interpolation(
                self.interpolation_start_pose, self.neutral_head_pose, interpolation_t,
            )

            # Interpolate antennas
            antennas_interp = (
                1 - interpolation_t
            ) * self.interpolation_start_antennas + interpolation_t * self.neutral_antennas
            antennas = antennas_interp.astype(np.float64)

        else:
            # Phase 2: Breathing patterns from neutral base
            breathing_time = t - self.interpolation_duration

            # Gentle z-axis breathing
            z_offset = self.breathing_z_amplitude * np.sin(2 * np.pi * self.breathing_frequency * breathing_time)
            head_pose = create_head_pose(x=0, y=0, z=z_offset, roll=0, pitch=0, yaw=0, degrees=True, mm=False)

            # Antenna sway (opposite directions)
            antenna_sway = self.antenna_sway_amplitude * np.sin(2 * np.pi * self.antenna_frequency * breathing_time)
            antennas = np.array([antenna_sway, -antenna_sway], dtype=np.float64)

        # Return in official Move interface format: (head_pose, antennas_array, body_yaw)
        # body_yaw=None so MovementManager preserves the last cached yaw —
        # face tracking offsets handle yaw while breathing.
        return (head_pose, antennas, None)


class AntennaMove(Move):
    """Smoothly interpolate both antennas to a target position.

    Returns None for head and body_yaw so MovementManager preserves the last
    cached primary head pose — antennas move without disturbing any concurrent
    head positioning.
    """

    def __init__(
        self,
        target_antennas: Tuple[float, float],
        duration: float,
        start_antennas: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self._target = np.array(target_antennas, dtype=np.float64)
        self._start = np.array(start_antennas, dtype=np.float64)
        self._duration = max(duration, 1e-3)

    @property
    def duration(self) -> float:
        return self._duration

    def evaluate(
        self, t: float
    ) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        alpha = min(1.0, t / self._duration)
        antennas = (1.0 - alpha) * self._start + alpha * self._target
        return (None, antennas.astype(np.float64), None)


class ScanRotationMove(Move):
    """Robotic scan move: head snaps first, then the body catches up.

    Three phases:
      1. HEAD SNAP  (head_snap_duration):
         Head quickly rotates in world frame to face target_yaw direction.
         Body stays at start_yaw.  Looks like the robot is "looking" somewhere.

      2. BODY CATCH-UP  (rotate_duration):
         Body slowly rotates from start_yaw to target_yaw.
         Head pose is held constant in world frame (= facing target_yaw),
         so the IK naturally keeps the camera locked on that world direction
         while the body swings underneath.

      3. HOLD  (hold_duration):
         Both head and body hold at target_yaw.

    The head pose is always expressed in the world frame (as the SDK expects).
    """

    HEAD_SNAP_DURATION = 0.25   # seconds for the quick head snap

    def __init__(
        self,
        start_yaw: float,
        target_yaw: float,
        rotate_duration: float,
        hold_duration: float = 0.0,
    ) -> None:
        self._start_yaw = float(start_yaw)
        self._target_yaw = float(target_yaw)
        self._snap_duration = self.HEAD_SNAP_DURATION
        self._rotate_duration = max(float(rotate_duration), 1e-3)
        self._hold_duration = max(float(hold_duration), 0.0)

        # Pre-compute the two world-frame head poses
        self._head_at_start = self._head_pose_for_yaw(self._start_yaw)
        self._head_at_target = self._head_pose_for_yaw(self._target_yaw)

    @staticmethod
    def _head_pose_for_yaw(yaw: float) -> NDArray[np.float64]:
        """World-frame head pose facing in the given yaw direction (Rz(yaw))."""
        c, s = np.cos(yaw), np.sin(yaw)
        pose = np.eye(4, dtype=np.float64)
        pose[0, 0] = c;  pose[0, 1] = -s
        pose[1, 0] = s;  pose[1, 1] = c
        return pose

    @property
    def duration(self) -> float:
        return self._snap_duration + self._rotate_duration + self._hold_duration

    def evaluate(
        self, t: float
    ) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        t_rotate_start = self._snap_duration
        t_hold_start   = self._snap_duration + self._rotate_duration

        if t < t_rotate_start:
            # Phase 1: head snaps quickly to target direction; body stays put
            alpha = t / self._snap_duration
            head_pose = linear_pose_interpolation(
                self._head_at_start, self._head_at_target, alpha
            )
            body_yaw = self._start_yaw

        elif t < t_hold_start:
            # Phase 2: head locked on target world direction; body rotates to catch up
            alpha = (t - t_rotate_start) / self._rotate_duration
            body_yaw = self._start_yaw + alpha * (self._target_yaw - self._start_yaw)
            head_pose = self._head_at_target.copy()

        else:
            # Phase 3: hold — both at target
            body_yaw = self._target_yaw
            head_pose = self._head_at_target.copy()

        return (head_pose, None, float(body_yaw))


def combine_full_body(primary_pose: FullBodyPose, secondary_pose: FullBodyPose) -> FullBodyPose:
    """Combine primary and secondary full body poses.

    Args:
        primary_pose: (head_pose, antennas, body_yaw) - primary move
        secondary_pose: (head_pose, antennas, body_yaw) - secondary offsets

    Returns:
        Combined full body pose (head_pose, antennas, body_yaw)

    """
    primary_head, primary_antennas, primary_body_yaw = primary_pose
    secondary_head, secondary_antennas, secondary_body_yaw = secondary_pose

    # Combine head poses using compose_world_offset; the secondary pose must be an
    # offset expressed in the world frame (T_off_world) applied to the absolute
    # primary transform (T_abs).
    combined_head = compose_world_offset(primary_head, secondary_head, reorthonormalize=True)

    # Sum antennas and body_yaw
    combined_antennas = (
        primary_antennas[0] + secondary_antennas[0],
        primary_antennas[1] + secondary_antennas[1],
    )
    combined_body_yaw = primary_body_yaw + secondary_body_yaw

    return (combined_head, combined_antennas, combined_body_yaw)


def clone_full_body_pose(pose: FullBodyPose) -> FullBodyPose:
    """Create a deep copy of a full body pose tuple."""
    head, antennas, body_yaw = pose
    return (head.copy(), (float(antennas[0]), float(antennas[1])), float(body_yaw))
