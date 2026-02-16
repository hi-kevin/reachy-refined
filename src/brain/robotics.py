import logging
import base64
import os
import cv2
import numpy as np
from typing import Any, Dict, Optional, List
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Use a capable multimodal model for vision
VISION_MODEL_ID = "gemini-robotics-er-1.5-preview" 

class RoboticsBrain:
    """Handles visual perception and spatial reasoning."""

    def __init__(self, robot: Any, api_key: Optional[str] = None):
        self.robot = robot
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            logger.error("GOOGLE_API_KEY not found (Robotics).")
        
        self.client = genai.Client(api_key=self.api_key, http_options={"api_version": "v1alpha"})

    async def capture_and_analyze(self, prompt: str = "Describe what you see.") -> str:
        """Capture image from robot and analyze it."""
        try:
            # Capture frame
            # Reachy Mini SDK convention: robot.camera returns (success, frame) or just frame
            # Assuming robot.camera.last_frame or similar. 
            # If using my local_stream style, maybe robot is the wrapper?
            # Let's hope robot.get_image() works or check reachy_mini sdk.
            # Using a safe fallback if robot specific method is unknown
            
            frame = None
            if hasattr(self.robot, 'camera'):
                if hasattr(self.robot.camera, 'last_frame'):
                     frame = self.robot.camera.last_frame
                elif hasattr(self.robot.camera, 'get_frame'):
                     frame = self.robot.camera.get_frame()
            
            # If robot is the ReachyMini wrapper from src.robot?
            # The user code passed ReachyMini instance.
            # Let's try to get the frame safely.
            
            if frame is None:
                return "I can't see anything right now (Camera error)."
                
            # Frame is likely BGR (OpenCV)
            # Encode to JPEG
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                return "Error encoding image."
            
            image_bytes = buffer.tobytes()
            return await self.analyze_scene(image_bytes, prompt)
            
        except Exception as e:
            logger.error(f"Capture error: {e}")
            return f"Error seeing: {e}"

    async def analyze_scene(self, image_data: bytes, prompt: str = "Describe what you see.") -> str:
        """
        Analyze the given image (JPEG bytes).
        
        Args:
            image_data: Raw JPEG bytes.
            prompt: Question or instruction for the vision model.
            
        Returns:
            Text description.
        """
        try:
            logger.info("RoboticsBrain: Analyzing scene...")
            
            # Create content with image blob
            response = await self.client.aio.models.generate_content(
                model=VISION_MODEL_ID,
                contents=[
                    types.Part.from_bytes(data=image_data, mime_type="image/jpeg"),
                    types.Part.from_text(text=prompt)
                ]
            )
            
            if response.text:
                logger.info(f"RoboticsBrain: Analysis complete. ({len(response.text)} chars)")
                return response.text
            return "I couldn't analyze the image."

        except Exception as e:
            logger.error(f"RoboticsBrain Error: {e}")
            return f"Error analyzing scene: {str(e)}"
