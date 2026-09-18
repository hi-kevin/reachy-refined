"""FaceIdentifier: ArcFace-based face recognition for Reachy Mini.

  - Detection stays local and fast: Haar cascade, adequate for the 10Hz
    tracking loop.
  - Recognition is ArcFace embeddings (ONNX, 512-dim) compared by cosine
    similarity. Unlike LBPH, this has a meaningful "no match" threshold:
    LBPH's predict() always returned the nearest known label whatever the
    face, so a stranger was identified as whoever they were closest to and
    told that person's private memories.
  - Subdirectory-per-person image storage: known_faces/{name}/*.jpg, kept so
    people can be re-enrolled offline if the embedding model ever changes.
  - Embeddings live in the people table (memory DB), not a side-car file.
  - Always returns a consistent dict from identify_face().
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and tunables
# ---------------------------------------------------------------------------
KNOWN_FACES_DIR = "known_faces"

# ArcFace recognition model (insightface model zoo).
# w600k_mbf (buffalo_s) is used rather than the larger w600k_r50 (buffalo_l):
# measured on this robot's aarch64 CPU, r50 takes ~1460 ms per embedding and
# mbf ~117 ms. Identification runs on the camera thread, so r50 would stall
# face tracking for over a second on every identification cycle.
# Override with ARCFACE_MODEL_PATH to swap in a different model.
ARCFACE_MODEL_PATH = os.getenv("ARCFACE_MODEL_PATH", "models/w600k_mbf.onnx")
ARCFACE_INPUT_SIZE = 112
ARCFACE_THREADS = 2          # leave headroom for the camera + motor threads

# Cosine similarity in [-1, 1]; higher is a better match. A face scoring below
# this against every enrolled person is reported as Unknown - which is the
# safe answer, and the thing LBPH could not express.
MATCH_THRESHOLD = 0.40

# Margin added around the Haar box before embedding. ArcFace expects a little
# context beyond the tight detector box.
CROP_MARGIN = 0.25

HAAR_SCALE = 1.1
HAAR_MIN_NEIGHBORS = 9       # reduces false positives
HAAR_MIN_SIZE = (80, 80)     # ignores tiny spurious detections


class FaceIdentifier:
    """Detect faces with Haar, recognise them with ArcFace embeddings."""

    def __init__(self, memory_server: Any = None) -> None:
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        self.memory_server = memory_server

        # Haar cascade for detection
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            logger.error("FaceIdentifier: failed to load Haar cascade from %s", cascade_path)

        # Runtime-tunable params (settable via MonitorServer)
        self.haar_min_neighbors = HAAR_MIN_NEIGHBORS
        self.haar_min_size = HAAR_MIN_SIZE
        self.match_threshold = MATCH_THRESHOLD

        # ArcFace ONNX session
        self._session = None
        self._input_name: Optional[str] = None
        self._load_recognizer()

        # name -> unit-normalised 512-dim vector, loaded from the memory DB
        self._embeddings: Dict[str, np.ndarray] = {}
        self.reload_embeddings()

    # ------------------------------------------------------------------
    # Recognition model
    # ------------------------------------------------------------------

    def _load_recognizer(self) -> None:
        """Create the ONNX inference session. Recognition is disabled if this fails."""
        if not os.path.exists(ARCFACE_MODEL_PATH):
            logger.error(
                "FaceIdentifier: ArcFace model not found at %s - recognition disabled. "
                "Everyone will be reported as Unknown.",
                ARCFACE_MODEL_PATH,
            )
            return
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = ARCFACE_THREADS
            self._session = ort.InferenceSession(
                ARCFACE_MODEL_PATH,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            logger.info("FaceIdentifier: loaded ArcFace model %s", ARCFACE_MODEL_PATH)
        except Exception as e:
            logger.error("FaceIdentifier: could not load ArcFace model: %s", e)
            self._session = None

    def _preprocess(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """BGR crop -> normalised NCHW float32 tensor for ArcFace."""
        if face_bgr is None or face_bgr.size == 0:
            return None
        try:
            img = cv2.resize(face_bgr, (ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE))
            if img.ndim == 2:                       # grayscale source image
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = (img.astype(np.float32) - 127.5) / 127.5
            return np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
        except Exception as e:
            logger.debug("FaceIdentifier: preprocess error: %s", e)
            return None

    def embed(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return a unit-normalised 512-dim embedding, or None."""
        if self._session is None:
            return None
        tensor = self._preprocess(face_bgr)
        if tensor is None:
            return None
        try:
            vec = self._session.run(None, {self._input_name: tensor})[0][0]
            norm = np.linalg.norm(vec)
            if norm == 0:
                return None
            return (vec / norm).astype(np.float32)
        except Exception as e:
            logger.warning("FaceIdentifier: embedding failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Enrolled embedding store (backed by the people table)
    # ------------------------------------------------------------------

    def reload_embeddings(self) -> None:
        """Refresh the in-memory embedding cache from the memory DB."""
        self._embeddings = {}
        if self.memory_server is None:
            logger.warning(
                "FaceIdentifier: no memory server - nobody can be recognised."
            )
            return
        try:
            for row in self.memory_server.get_face_embeddings():
                name = row.get("display_name") or row.get("face_label")
                blob = row.get("embedding")
                if not name or not blob:
                    continue
                vec = np.frombuffer(blob, dtype=np.float32)
                if vec.size != 512:
                    logger.warning(
                        "FaceIdentifier: skipping '%s' - embedding is %d dims, expected 512.",
                        name, vec.size,
                    )
                    continue
                self._embeddings[name] = vec
            logger.info(
                "FaceIdentifier: loaded %d enrolled face(s): %s",
                len(self._embeddings), list(self._embeddings) or "(none)",
            )
        except Exception as e:
            logger.error("FaceIdentifier: reload_embeddings error: %s", e)

    def _store_embedding(self, name: str, vec: np.ndarray) -> None:
        """Persist an embedding against the person record and refresh the cache."""
        if self.memory_server is None:
            logger.error("FaceIdentifier: cannot store embedding - no memory server.")
            return
        person_id = self.memory_server.get_or_create_person(face_label=name)
        self.memory_server.set_face_embedding(person_id, vec.tobytes())
        self._embeddings[name] = vec

    def _match(self, vec: np.ndarray) -> Tuple[str, float]:
        """Return (best name, cosine similarity). Name is 'Unknown' below threshold."""
        if not self._embeddings:
            return "Unknown", -1.0
        names = list(self._embeddings)
        matrix = np.stack([self._embeddings[n] for n in names])
        sims = matrix @ vec                      # both sides unit-normalised
        best = int(np.argmax(sims))
        score = float(sims[best])
        if score < self.match_threshold:
            return "Unknown", score
        return names[best], score

    # ------------------------------------------------------------------
    # Core identification
    # ------------------------------------------------------------------

    def identify_face(self, frame: np.ndarray) -> Dict:
        """Detect and identify the largest face in a BGR frame.

        Always returns a dict - callers never need to do None checks:
          {
            "found":      bool,
            "name":       str,      # "Unknown" if no match or no face
            "confidence": float,    # cosine similarity, HIGHER is better
                                    #   (-1.0 when there is no face/embedding)
            "face_crop":  ndarray,  # BGR crop; None if no face
            "face_gray":  ndarray,  # Grayscale crop; None if no face
            "bbox":       (x,y,w,h) or None,
            "is_known":   bool,
          }

        Note `confidence` is a similarity, not the LBPH distance it used to
        be - the comparison direction is inverted from the old behaviour.
        """
        empty = {
            "found": False,
            "name": "Unknown",
            "confidence": -1.0,
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
            minNeighbors=self.haar_min_neighbors,
            minSize=self.haar_min_size,
        )

        if len(faces) == 0:
            return empty

        # Pick the largest face by area
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_crop = frame[y : y + h, x : x + w]
        face_gray = gray[y : y + h, x : x + w]

        name = "Unknown"
        score = -1.0
        is_known = False

        vec = self.embed(self._crop_with_margin(frame, x, y, w, h))
        if vec is not None:
            name, score = self._match(vec)
            is_known = name != "Unknown"

        return {
            "found": True,
            "name": name,
            "confidence": score,
            "face_crop": face_crop,
            "face_gray": face_gray,
            "bbox": (x, y, w, h),
            "is_known": is_known,
        }

    def _crop_with_margin(
        self, frame: np.ndarray, x: int, y: int, w: int, h: int
    ) -> np.ndarray:
        """Crop the face box with CROP_MARGIN padding, clamped to the frame."""
        mx, my = int(w * CROP_MARGIN), int(h * CROP_MARGIN)
        fh, fw = frame.shape[:2]
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(fw, x + w + mx), min(fh, y + h + my)
        return frame[y0:y1, x0:x1]

    def get_detection_params(self) -> dict:
        """Return current detection tunables."""
        return {
            "haar_min_neighbors": self.haar_min_neighbors,
            "haar_min_size":      self.haar_min_size[0],  # square assumed
            "match_threshold":    self.match_threshold,
        }

    def set_detection_params(self, **kwargs) -> None:
        """Update detection tunables at runtime. Unknown keys are ignored."""
        if "haar_min_neighbors" in kwargs:
            self.haar_min_neighbors = int(kwargs["haar_min_neighbors"])
        if "haar_min_size" in kwargs:
            s = int(kwargs["haar_min_size"])
            self.haar_min_size = (s, s)
        if "match_threshold" in kwargs:
            self.match_threshold = float(kwargs["match_threshold"])

    # ------------------------------------------------------------------
    # Enrolment
    # ------------------------------------------------------------------

    def capture_training_images(
        self, robot: Any, name: str, count: int = 5
    ) -> str:
        """Capture `count` face photos from the robot camera and enrol `name`.

        Saves colour face crops to known_faces/{name}/ - colour, because
        ArcFace needs RGB input and grayscale crops embed noticeably worse.
        Sleeps 0.5 s between shots to get slightly different angles.
        """
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        vectors: List[np.ndarray] = []
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

                x, y, w, h = result["bbox"]
                crop = self._crop_with_margin(frame, x, y, w, h)

                ts_ms = int(time.time() * 1000)
                filename = os.path.join(person_dir, f"{name}_{ts_ms}.jpg")
                cv2.imwrite(filename, crop)
                saved += 1

                vec = self.embed(crop)
                if vec is not None:
                    vectors.append(vec)
            except Exception as e:
                logger.warning("capture_training_images: shot %d error: %s", i, e)
            time.sleep(0.5)

        if saved == 0:
            return (
                f"No face detected during registration for {name}. "
                "Please try again facing the camera."
            )
        if not vectors:
            return (
                f"Saved {saved} photo(s) of {name}, but could not compute a face "
                "signature - recognition model unavailable."
            )

        self._enrol_vectors(name, vectors)
        return f"Registered {name} from {saved}/{count} photos. I'll recognise them next time."

    def register_single_frame(self, frame: np.ndarray, name: str) -> str:
        """Enrol a face from a single already-captured frame."""
        result = self.identify_face(frame)
        if not result["found"] or result["bbox"] is None:
            return f"No face detected in frame for {name}."

        x, y, w, h = result["bbox"]
        crop = self._crop_with_margin(frame, x, y, w, h)

        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        os.makedirs(person_dir, exist_ok=True)
        ts_ms = int(time.time() * 1000)
        cv2.imwrite(os.path.join(person_dir, f"{name}_{ts_ms}.jpg"), crop)

        vec = self.embed(crop)
        if vec is None:
            return f"Saved a photo of {name} but could not compute a face signature."

        self._enrol_vectors(name, [vec])
        return f"Registered {name}."

    def _enrol_vectors(self, name: str, vectors: List[np.ndarray]) -> None:
        """Average the given vectors with any existing enrolment and persist."""
        stack = list(vectors)
        existing = self._embeddings.get(name)
        if existing is not None:
            stack.append(existing)
        mean = np.mean(np.stack(stack), axis=0)
        norm = np.linalg.norm(mean)
        if norm == 0:
            logger.error("FaceIdentifier: degenerate embedding for '%s'", name)
            return
        self._store_embedding(name, (mean / norm).astype(np.float32))

    def train_model(self) -> str:
        """Re-embed every image under known_faces/ and rewrite all enrolments.

        This is the offline re-enrolment path - run it after changing the
        ArcFace model so stored vectors match the new model's space.
        """
        if self._session is None:
            return "Recognition model unavailable - cannot re-enrol."

        people = 0
        images = 0
        for entry in os.scandir(KNOWN_FACES_DIR):
            if not entry.is_dir():
                continue
            vectors: List[np.ndarray] = []
            for img_entry in os.scandir(entry.path):
                if not img_entry.name.lower().endswith((".jpg", ".png")):
                    continue
                img = cv2.imread(img_entry.path, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                vec = self.embed(img)
                if vec is not None:
                    vectors.append(vec)
                    images += 1
            if not vectors:
                continue
            mean = np.mean(np.stack(vectors), axis=0)
            norm = np.linalg.norm(mean)
            if norm == 0:
                continue
            self._store_embedding(entry.name, (mean / norm).astype(np.float32))
            people += 1

        if people == 0:
            return "No usable training images found in known_faces/."
        summary = f"Re-enrolled {people} people from {images} images."
        logger.info(summary)
        return summary

    def forget(self, name: str) -> str:
        """Delete a person's photos and drop their embedding from the cache.

        The memory DB rows are removed by MemoryServer.forget_person(); this
        handles the on-disk images and the in-process cache.
        """
        self._embeddings.pop(name, None)
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        removed = 0
        if os.path.isdir(person_dir):
            for img_entry in os.scandir(person_dir):
                try:
                    os.remove(img_entry.path)
                    removed += 1
                except OSError as e:
                    logger.warning("forget: could not remove %s: %s", img_entry.path, e)
            try:
                os.rmdir(person_dir)
            except OSError:
                pass
        logger.info("FaceIdentifier: forgot '%s' (%d photos removed)", name, removed)
        return f"Removed {removed} photo(s) for {name}."

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def known_people(self) -> list:
        """Return list of known person names."""
        return list(self._embeddings)
