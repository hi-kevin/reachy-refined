"""FaceIdentifier: LBPH-based face recognition for Reachy Mini.

  - Subdirectory-per-person storage: known_faces/{name}/*.jpg
  - Always returns a consistent dict from identify_face() — no None checks needed
  - capture_training_images() takes 5 photos over ~2.5 seconds
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and tunables
# ---------------------------------------------------------------------------
KNOWN_FACES_DIR = "known_faces"
RECOGNIZER_MODEL_PATH = "face_recognizer.yml"
LABEL_MAP_PATH = "label_map.pkl"
LBPH_THRESHOLD = 80          # LBPH distance: lower is better; < this → known
HAAR_SCALE = 1.1
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_SIZE = (40, 40)


class FaceIdentifier:
    """Detect and identify faces using Haar cascade + LBPH."""

    def __init__(self) -> None:
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

        # Haar cascade for detection
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            logger.error("FaceIdentifier: failed to load Haar cascade from %s", cascade_path)

        # LBPH for recognition
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError:
            logger.error(
                "FaceIdentifier: cv2.face not found. "
                "Install opencv-contrib-python."
            )
            self.recognizer = None

        self.label_map: Dict[int, str] = {}
        self.load_model()

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the trained LBPH model and label map from disk (if they exist)."""
        if self.recognizer is not None and os.path.exists(RECOGNIZER_MODEL_PATH):
            try:
                self.recognizer.read(RECOGNIZER_MODEL_PATH)
                logger.info("FaceIdentifier: loaded LBPH model from %s", RECOGNIZER_MODEL_PATH)
            except Exception as e:
                logger.warning("FaceIdentifier: could not load model: %s", e)

        if os.path.exists(LABEL_MAP_PATH):
            try:
                with open(LABEL_MAP_PATH, "rb") as f:
                    self.label_map = pickle.load(f)
                logger.info(
                    "FaceIdentifier: loaded label map — %d known people: %s",
                    len(self.label_map),
                    list(self.label_map.values()),
                )
            except Exception as e:
                logger.warning("FaceIdentifier: could not load label map: %s", e)
                self.label_map = {}
        else:
            self.label_map = {}

    def save_model(self) -> None:
        """Save the LBPH model and label map to disk."""
        if self.recognizer is not None:
            self.recognizer.save(RECOGNIZER_MODEL_PATH)
        with open(LABEL_MAP_PATH, "wb") as f:
            pickle.dump(self.label_map, f)

    def train_model(self) -> str:
        """Retrain LBPH on all images under known_faces/.

        Expects subdirectory structure: known_faces/{name}/*.jpg
        """
        if self.recognizer is None:
            return "Recognizer not initialized."

        faces = []
        labels = []
        label_map: Dict[int, str] = {}
        current_id = 0

        def _get_or_assign_id(name: str) -> int:
            nonlocal current_id
            for lid, lname in label_map.items():
                if lname == name:
                    return lid
            lid = current_id
            label_map[lid] = name
            current_id += 1
            return lid

        for entry in os.scandir(KNOWN_FACES_DIR):
            if entry.is_dir():
                # Subdirectory mode: known_faces/{name}/
                name = entry.name
                label_id = _get_or_assign_id(name)
                for img_entry in os.scandir(entry.path):
                    if img_entry.name.lower().endswith((".jpg", ".png")):
                        img = cv2.imread(img_entry.path, cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            continue
                        faces.append(img)
                        labels.append(label_id)

        if not faces:
            return "No training images found in known_faces/."

        try:
            self.recognizer.train(faces, np.array(labels))
            self.label_map = label_map
            self.save_model()
            summary = (
                f"Trained LBPH on {len(faces)} images of "
                f"{len(label_map)} people: {list(label_map.values())}"
            )
            logger.info(summary)
            return summary
        except Exception as e:
            logger.error("train_model error: %s", e)
            return f"Training failed: {e}"

    # ------------------------------------------------------------------
    # Core identification
    # ------------------------------------------------------------------

    def identify_face(self, frame: np.ndarray) -> Dict:
        """Detect and identify the largest face in a BGR frame.

        Always returns a dict — callers never need to do None checks:
          {
            "found":      bool,
            "name":       str,      # "Unknown" if no match or no face
            "confidence": float,    # LBPH distance (lower = better; 999 if no face)
            "face_crop":  ndarray,  # BGR crop; None if no face
            "face_gray":  ndarray,  # Grayscale crop; None if no face
            "bbox":       (x,y,w,h) or None,
            "is_known":   bool,
          }
        """
        empty = {
            "found": False,
            "name": "Unknown",
            "confidence": 999.0,
            "face_crop": None,
            "face_gray": None,
            "bbox": None,
            "is_known": False,
        }

        if frame is None:
            return empty

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception:
            return empty

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=HAAR_SCALE,
            minNeighbors=HAAR_MIN_NEIGHBORS,
            minSize=HAAR_MIN_SIZE,
        )

        if len(faces) == 0:
            return empty

        # Pick the largest face by area
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_crop = frame[y : y + h, x : x + w]
        face_gray = gray[y : y + h, x : x + w]

        name = "Unknown"
        distance = 999.0
        is_known = False

        if self.recognizer is not None and self.label_map:
            try:
                label_id, dist = self.recognizer.predict(face_gray)
                distance = float(dist)
                if distance < LBPH_THRESHOLD:
                    name = self.label_map.get(label_id, "Unknown")
                    is_known = True
            except Exception as e:
                logger.debug("LBPH predict error: %s", e)

        return {
            "found": True,
            "name": name,
            "confidence": distance,
            "face_crop": face_crop,
            "face_gray": face_gray,
            "bbox": (x, y, w, h),
            "is_known": is_known,
        }

    # ------------------------------------------------------------------
    # Training data capture
    # ------------------------------------------------------------------

    def capture_training_images(
        self, robot: Any, name: str, count: int = 5
    ) -> str:
        """Capture `count` face photos from the robot camera and train the model.

        Saves grayscale face crops to known_faces/{name}/.
        Sleeps 0.5 s between shots to get slightly different angles.
        """
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        saved = 0
        for i in range(count):
            try:
                frame = robot.media.get_frame()
                if frame is None:
                    logger.warning("capture_training_images: got None frame (shot %d)", i)
                    time.sleep(0.5)
                    continue

                result = self.identify_face(frame)
                if not result["found"]:
                    logger.debug("capture_training_images: no face detected (shot %d)", i)
                    time.sleep(0.5)
                    continue

                ts_ms = int(time.time() * 1000)
                filename = os.path.join(person_dir, f"{name}_{ts_ms}.jpg")
                cv2.imwrite(filename, result["face_gray"])
                saved += 1
                logger.debug("Saved training image: %s", filename)
            except Exception as e:
                logger.warning("capture_training_images: shot %d error: %s", i, e)
            time.sleep(0.5)

        if saved == 0:
            return f"No face detected during registration for {name}. Please try again facing the camera."

        msg = self.train_model()
        return f"Registered {name} with {saved}/{count} images. {msg}"

    def register_single_frame(self, frame: np.ndarray, name: str) -> str:
        """Register a face from a single already-captured frame."""
        result = self.identify_face(frame)
        if not result["found"] or result["face_gray"] is None:
            return f"No face detected in frame for {name}."

        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        os.makedirs(person_dir, exist_ok=True)
        ts_ms = int(time.time() * 1000)
        filename = os.path.join(person_dir, f"{name}_{ts_ms}.jpg")
        cv2.imwrite(filename, result["face_gray"])

        msg = self.train_model()
        return f"Registered frame for {name}. {msg}"

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def known_people(self) -> list:
        """Return list of known person names."""
        return list(set(self.label_map.values()))
