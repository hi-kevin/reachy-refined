try:
    from reachy_mini import ReachyMini
except ImportError:
    print("ReachyMini package not found. Mocking...")
    ReachyMini = None

import time
import cv2
import numpy as np
import os

class Robot:
    def __init__(self, host='reachy-mini.local'):
        print(f"Connecting to Reachy Mini...")
        self.mini = None
        if ReachyMini:
            try:
                # CONNECTION_MODE: 'local' (on robot) or 'network' (remote)
                # Default to 'network' for dev machine, 'local' (empty args) for on-robot.
                mode = os.getenv("REACHY_CONNECTION_MODE", "network")
                
                print(f"Connection mode: {mode}")
                
                if mode == "local":
                    self.mini = ReachyMini() # On robot
                else:
                    self.mini = ReachyMini(connection_mode='network') # Remote
                
                print("Connected to Reachy Mini!")
            except Exception as e:
                print(f"Connection failed: {e}. Running in Mock Mode.")
        else:
            print("ReachyMini library missing. Running in Mock Mode.")

    def is_connected(self):
        return self.mini is not None

    def get_right_image(self):
        """Gets the latest frame from the right camera."""
        if self.mini:
             try:
                 # Check media_manager first (preferred for ReachyMini)
                 if hasattr(self.mini, 'media_manager'):
                     return self.mini.media_manager.get_frame()

                 # Fallback to old attributes
                 if hasattr(self.mini, 'right_camera'):
                     return self.mini.right_camera.last_frame
                 elif hasattr(self.mini, 'cameras') and hasattr(self.mini.cameras, 'right'):
                     return self.mini.cameras.right.last_frame
             except Exception as e:
                 print(f"DEBUG: get_right_image error: {e}")
                 pass
        else:
            print("DEBUG: Robot not connected")
        return None

    def look_at(self, x: float, y: float, z: float, duration: float = 1.0):
        """
        Directs the robot's head to look at a point in 3D space.
        
        Args:
            x: X coordinate (forward, meters).
            y: Y coordinate (left/right, meters).
            z: Z coordinate (up/down, meters).
            duration: Time in seconds to complete the movement.
        """
        if self.mini:
            try:
                self.mini.look_at(x=x, y=y, z=z, duration=duration)
            except:
                print("look_at execution failed")
        else:
            print(f"MOCK: Look at ({x}, {y}, {z})")

    def wiggle_antennas(self):
        if self.mini:
            try:
                self.mini.goto_target(antennas=[0.5, -0.5], duration=0.5)
                time.sleep(0.5)
                self.mini.goto_target(antennas=[-0.5, 0.5], duration=0.5)
                time.sleep(0.5)
                self.mini.goto_target(antennas=[0, 0], duration=0.5)
            except:
                print("wiggle execution failed")
        else:
            print("MOCK: Wiggle antennas")

    def turn_on(self, part="head"):
        """Attempts to turn on the robot motors."""
        if self.mini:
            try:
                # ReachyMini might not have explicit turn_on/off for everything, 
                # but let's try calling it if it exists or just pass.
                if hasattr(self.mini, 'turn_on'):
                    self.mini.turn_on(part)
                else:
                    print(f"DEBUG: ReachyMini has no turn_on method. Assuming auto-on.")
            except Exception as e:
                print(f"DEBUG: turn_on error: {e}")
        else:
            print("MOCK: Robot ON")

    def turn_off(self, part="head"):
        """Attempts to turn off the robot motors."""
        if self.mini:
            try:
                if hasattr(self.mini, 'turn_off'):
                    self.mini.turn_off(part)
                else:
                    print(f"DEBUG: ReachyMini has no turn_off method.")
            except Exception as e:
                print(f"DEBUG: turn_off error: {e}")
        else:
            print("MOCK: Robot OFF")

    def say(self, text):
        # Use Audio System (passed in or internal)
        # This wrapper mainly handles hardware motion.
        pass

if __name__ == "__main__":
    r = Robot()
    if r.is_connected():
        r.wiggle_antennas()
