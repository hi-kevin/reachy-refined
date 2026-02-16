import cv2
import numpy as np
import os
import pickle

# Directory to store known face images
KNOWN_FACES_DIR = "known_faces"
# Path to save trained recognizer model
RECOGNIZER_MODEL_PATH = "face_recognizer.yml"
# Mapping from Label ID (int) to Name (str)
LABEL_MAP_PATH = "label_map.pkl"

class VisionSystem:
    def __init__(self):
        if not os.path.exists(KNOWN_FACES_DIR):
            os.makedirs(KNOWN_FACES_DIR)
            
        # Initialize Face Detector (Haar Cascade)
        # Using default frontal face
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize Face Recognizer (LBPH)
        # Check if contrib module is available
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError:
            print("Error: cv2.face not found. Ensure opencv-contrib-python is installed.")
            self.recognizer = None

        self.label_map = {}
        self.load_model()

    def load_model(self):
        """Loads the trained recognizer model and label map."""
        if os.path.exists(RECOGNIZER_MODEL_PATH):
            self.recognizer.read(RECOGNIZER_MODEL_PATH)
            
        if os.path.exists(LABEL_MAP_PATH):
            with open(LABEL_MAP_PATH, 'rb') as f:
                self.label_map = pickle.load(f)
            print(f"Loaded model with {len(self.label_map)} known people.")

    def save_model(self):
        """Saves the trained recognizer model and label map."""
        self.recognizer.save(RECOGNIZER_MODEL_PATH)
        with open(LABEL_MAP_PATH, 'wb') as f:
            pickle.dump(self.label_map, f)

    def train_model(self):
        """
        Retrains the LBPH model on all images in KNOWN_FACES_DIR.
        """
        if not self.recognizer:
            return "Recognizer not initialized."

        faces = []
        labels = []
        current_id = 0
        new_label_map = {}
        
        # Iterate through people directories or assume filename format Name_Index.jpg
        # Better to have subdirectories per person: known_faces/Dave/*.jpg
        # Or flat: known_faces/Dave_1.jpg
        
        # Let's support flat directory with Name_Index.jpg
        for filename in os.listdir(KNOWN_FACES_DIR):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                path = os.path.join(KNOWN_FACES_DIR, filename)
                name = filename.split('_')[0]
                
                if name not in new_label_map.values():
                    # Assign new ID
                    new_label_map[current_id] = name
                    label_id = current_id
                    current_id += 1
                else:
                    # Find existing ID
                    for lid, lname in new_label_map.items():
                        if lname == name:
                            label_id = lid
                            break
                
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                    
                faces.append(img)
                labels.append(label_id)

        if len(faces) > 0:
            self.recognizer.train(faces, np.array(labels))
            self.label_map = new_label_map
            self.save_model()
            return f"Trained on {len(faces)} images of {len(self.label_map)} people."
        else:
            return "No images found to train."

    def detect_and_identify(self, frame):
        """
        Detects faces and identifies them.
        Returns list of {'name': str, 'location': (y, right, bottom, left), 'distance': float}
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        results = []
        for (x, y, w, h) in faces:
            name = "Unknown"
            confidence = 0.0 # Distance
            
            # Recognize
            if self.recognizer and self.label_map:
                roi_gray = gray[y:y+h, x:x+w]
                try:
                    label_id, distance = self.recognizer.predict(roi_gray)
                    # LBPH distance: 0 is perfect match. Usually < 50 is good match.
                    # Confidence is weirdly named in opencv api, it returns distance.
                    # Let's assume distance < 100 is a match?
                    # "confidence" here is distance. Lower is better.
                    if distance < 80: # Threshold
                        name = self.label_map.get(label_id, "Unknown")
                    confidence = distance
                except Exception as e:
                    print(f"Prediction error: {e}")

            results.append({
                "name": name,
                "location": (y, x+w, y+h, x),
                "distance": confidence,
                "face_crop": frame[y:y+h, x:x+w]
            })
            
        return results

    def register_face(self, frame, name):
        """
        Registers a new face. 
        Expects 'frame' to contain the face crop or full image (will try to detect).
        """
        # Checks if frame is grayscale? No, imwrite handles it.
        # But for training we need grayscale.
        
        # If frame is small, assume it's a crop.
        # Or run detection on it just in case.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, minNeighbors=3)
        
        if len(faces) == 0:
            # Maybe it is already a cropped face?
            face_img = gray
            # But we might want robustness
        else:
            (x, y, w, h) = faces[0] # Take largest?
            face_img = gray[y:y+h, x:x+w]

        # Save to disk
        count = 0
        while True:
            filename = f"{name}_{count}.jpg"
            path = os.path.join(KNOWN_FACES_DIR, filename)
            if not os.path.exists(path):
                break
            count += 1
            
        # Save original BGR crop for reference, but training uses grayscale
        # Actually LBPH needs grayscale.
        # Let's save the grayscale version to reduce space and complexity
        cv2.imwrite(path, face_img)
        
        # Train model
        msg = self.train_model()
        
        return f"Registered {name} as {filename}. {msg}"

if __name__ == "__main__":
    vs = VisionSystem()
    print("Vision System LBPH Initialized")
