import cv2
import numpy as np

class FaceDetector:
    def __init__(self, scale_factor=1.3, min_neighbors=5):
        """Initialize face detector with OpenCV's Haar cascade classifier."""
        # Load pre-trained model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        
    def detect(self, frame):
        """
        Detect faces in the given frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            (x, y, w, h) coordinates of the face or None if no face detected
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(30, 30)
        )
        
        # Return the first face found (assuming single user)
        if len(faces) > 0:
            return faces[0]  # Returns (x, y, w, h)
        
        return None