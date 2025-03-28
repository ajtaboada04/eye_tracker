import cv2
import numpy as np
from utils import image_preprocessing

class EyeDetector:
    def __init__(self, eye_cascade_path=None):
        """
        Initialize the eye detector.
        
        Args:
            eye_cascade_path: Path to eye cascade XML file
                             If None, uses OpenCV's default
        """
        if eye_cascade_path is None:
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        
    def detect(self, frame, face):
        """
        Detect eyes within a face region.
        
        Args:
            frame: Full image frame
            face: Face coordinates as (x, y, w, h)
            
        Returns:
            Dictionary with 'left_eye' and 'right_eye' coordinates
            Each eye is represented as (x, y, w, h) relative to original frame
            Returns None for eyes that couldn't be detected
        """
        if face is None:
            return {'left_eye': None, 'right_eye': None}
        
        x, y, w, h = face
        
        # Extract face region
        face_roi = frame[y:y+h, x:x+w]
        
        # Convert to grayscale and preprocess
        gray_roi = image_preprocessing(face_roi)
        
        # Detect eyes in the face region
        eyes = self.eye_cascade.detectMultiScale(
            gray_roi,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # If no eyes are detected, return None for both eyes
        if len(eyes) == 0:
            return {'left_eye': None, 'right_eye': None}
            
        # If only one eye is detected, use it as the detected eye
        # and return None for the other eye
        if len(eyes) == 1:
            eye_x, eye_y, eye_w, eye_h = eyes[0]
            # Add face coordinates offset to get global image coordinates
            eye = (x + eye_x, y + eye_y, eye_w, eye_h)
            
            # Assuming that if the eye is on the left half of the face, it's the right eye
            # (from the person's perspective, left from camera's view)
            if eye_x < w/2:
                return {'left_eye': None, 'right_eye': eye}
            else:
                return {'left_eye': eye, 'right_eye': None}
        
        # Sort eyes by x-coordinate (left to right)
        eyes = sorted(eyes, key=lambda e: e[0])
        
        # We assume the leftmost eye is the right eye (from the person's perspective)
        # and the rightmost eye is the left eye
        right_eye_x, right_eye_y, right_eye_w, right_eye_h = eyes[0]
        left_eye_x, left_eye_y, left_eye_w, left_eye_h = eyes[1]
        
        # Add face coordinates offset to get global image coordinates
        right_eye = (x + right_eye_x, y + right_eye_y, right_eye_w, right_eye_h)
        left_eye = (x + left_eye_x, y + left_eye_y, left_eye_w, left_eye_h)
        
        return {
            'left_eye': left_eye,
            'right_eye': right_eye
        }
        
    def extract_eye_regions(self, frame, eyes):
        """
        Extract eye images from the frame using eye coordinates.
        
        Args:
            frame: Input image frame
            eyes: Dictionary with 'left_eye' and 'right_eye' coordinates
            
        Returns:
            Dictionary with 'left_eye' and 'right_eye' image regions
            Returns None for eyes that couldn't be extracted
        """
        eye_regions = {'left_eye': None, 'right_eye': None}
        
        for eye_name in ['left_eye', 'right_eye']:
            if eyes[eye_name] is not None:
                x, y, w, h = eyes[eye_name]
                # Extract eye region
                eye_region = frame[y:y+h, x:x+w]
                # Preprocess eye region
                eye_region = image_preprocessing(eye_region)
                eye_regions[eye_name] = eye_region
                
        return eye_regions