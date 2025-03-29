import dlib
import cv2
import numpy as np

class FacialLandmarkDetector:
    def __init__(self, predictor_path=None):
        """
        Initialize the facial landmark detector.
        
        Args:
            predictor_path: Path to dlib's facial landmark predictor model file.
                            If None, it will use a default model.
        """
        # Initialize dlib's face detector
        self.detector = dlib.get_frontal_face_detector()
        
        # Load the facial landmark predictor
        if predictor_path is None:
            # Default path - you may need to download this file separately
            predictor_path = "models/shape_predictor_68_face_landmarks.dat"
        
        try:
            self.predictor = dlib.shape_predictor(predictor_path)
            self.model_loaded = True
        except Exception as e:
            print(f"Error loading facial landmark predictor: {e}")
            self.model_loaded = False
    
    def download_model(self, save_path="models/shape_predictor_68_face_landmarks.dat"):
        """
        Download the facial landmark predictor model if it doesn't exist.
        """
        import os
        import urllib.request
        
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
            
        if not os.path.exists(save_path):
            print("Downloading facial landmark predictor model...")
            url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
            
            # Download compressed file
            compressed_path = save_path + ".bz2"
            urllib.request.urlretrieve(url, compressed_path)
            
            # Extract the file
            import bz2
            with open(save_path, 'wb') as new_file, bz2.BZ2File(compressed_path, 'rb') as file:
                for data in iter(lambda: file.read(100 * 1024), b''):
                    new_file.write(data)
                    
            # Clean up compressed file
            os.remove(compressed_path)
            print("Model downloaded and extracted successfully.")
            
            # Load the predictor
            self.predictor = dlib.shape_predictor(save_path)
            self.model_loaded = True
            
            return True
        return False
    
    def detect_landmarks(self, frame):
        """
        Detect facial landmarks in the given frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Dictionary containing:
                'face': dlib rectangle object of the face
                'landmarks': numpy array of 68 landmark points
                'success': boolean indicating if detection was successful
        """
        if not self.model_loaded:
            print("Facial landmark predictor model not loaded. Attempting to download...")
            if not self.download_model():
                return {'face': None, 'landmarks': None, 'success': False}
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector(gray)
        
        if len(faces) == 0:
            return {'face': None, 'landmarks': None, 'success': False}
        
        # Get the first face
        face = faces[0]
        
        # Detect landmarks
        landmarks = self.predictor(gray, face)
        
        # Convert landmarks to numpy array
        landmarks_points = []
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            landmarks_points.append((x, y))
        
        landmarks_np = np.array(landmarks_points)
        
        return {
            'face': face,
            'landmarks': landmarks_np,
            'success': True
        }
    
    def get_eye_regions(self, landmarks):
        """
        Extract eye regions from facial landmarks.
        
        Args:
            landmarks: Array of facial landmark points
            
        Returns:
            Dictionary with 'left_eye' and 'right_eye' coordinates as (x, y, w, h)
        """
        if landmarks is None:
            return {'left_eye': None, 'right_eye': None}
        
        # Indices for left and right eyes in 68-point model
        # Left eye (from the subject's perspective, right from camera view)
        left_eye_indices = list(range(42, 48))
        # Right eye (from the subject's perspective, left from camera view)
        right_eye_indices = list(range(36, 42))
        
        # Get eye landmarks
        left_eye_points = landmarks[left_eye_indices]
        right_eye_points = landmarks[right_eye_indices]
        
        # Calculate bounding boxes with padding
        def get_eye_box(eye_points, padding=5):
            min_x = np.min(eye_points[:, 0]) - padding
            max_x = np.max(eye_points[:, 0]) + padding
            min_y = np.min(eye_points[:, 1]) - padding
            max_y = np.max(eye_points[:, 1]) + padding
            
            return (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))
        
        left_eye_box = get_eye_box(left_eye_points)
        right_eye_box = get_eye_box(right_eye_points)
        
        return {
            'left_eye': left_eye_box,
            'right_eye': right_eye_box
        }
    
    def visualize_landmarks(self, frame, landmarks):
        """
        Draw facial landmarks on the frame.
        
        Args:
            frame: Input image frame
            landmarks: Array of landmark points
            
        Returns:
            Frame with landmarks drawn
        """
        if landmarks is None:
            return frame
        
        vis_frame = frame.copy()
        
        # Draw all landmarks
        for (x, y) in landmarks:
            cv2.circle(vis_frame, (x, y), 2, (0, 255, 0), -1)
        
        # Draw eye landmarks
        # Left eye (from the subject's perspective)
        for i in range(42, 48):
            x, y = landmarks[i]
            cv2.circle(vis_frame, (x, y), 3, (255, 0, 0), -1)
        
        # Right eye (from the subject's perspective)
        for i in range(36, 42):
            x, y = landmarks[i]
            cv2.circle(vis_frame, (x, y), 3, (0, 0, 255), -1)
        
        return vis_frame 