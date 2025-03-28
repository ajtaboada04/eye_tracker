import cv2
import numpy as np
import time
import os
import json
from utils import create_directory

class Calibrator:
    def __init__(self, screen_width=1920, screen_height=1080, points=9):
        """
        Initialize the calibration system.
        
        Args:
            screen_width: Width of the screen in pixels
            screen_height: Height of the screen in pixels
            points: Number of calibration points (typically 9 or 16)
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.points = points
        
        # Generate calibration points (in a grid pattern)
        self.calibration_points = self._generate_calibration_points()
        
        # Calibration data storage
        self.calibration_data = {
            'points': [],  # List of (x, y) screen coordinates
            'eye_features': []  # List of eye features for each point
        }
        
        # Current calibration state
        self.current_point_index = 0
        self.is_calibrated = False
        self.calibration_model = None
        
    def _generate_calibration_points(self):
        """Generate a grid of calibration points."""
        points = []
        
        # Determine grid size (e.g., 3x3 for 9 points)
        if self.points == 9:
            rows, cols = 3, 3
        elif self.points == 16:
            rows, cols = 4, 4
        else:
            # Default to 3x3 grid
            rows, cols = 3, 3
            
        # Calculate spacing
        margin_x = int(self.screen_width * 0.1)  # 10% margin
        margin_y = int(self.screen_height * 0.1)  # 10% margin
        
        step_x = (self.screen_width - 2 * margin_x) // (cols - 1)
        step_y = (self.screen_height - 2 * margin_y) // (rows - 1)
        
        # Generate points
        for row in range(rows):
            for col in range(cols):
                x = margin_x + col * step_x
                y = margin_y + row * step_y
                points.append((x, y))
                
        return points
    
    def start_calibration(self):
        """Start the calibration process."""
        self.current_point_index = 0
        self.calibration_data = {
            'points': [],
            'eye_features': []
        }
        self.is_calibrated = False
        
        return self.get_current_point()
    
    def get_current_point(self):
        """Get the current calibration point."""
        if self.current_point_index < len(self.calibration_points):
            return self.calibration_points[self.current_point_index]
        return None
    
    def add_calibration_sample(self, eye_features):
        """
        Add a calibration sample for the current point.
        
        Args:
            eye_features: Features extracted from the eye regions
            
        Returns:
            True if there are more points to calibrate, False if calibration is complete
        """
        current_point = self.get_current_point()
        if current_point is None:
            return False
            
        # Store the point and its corresponding eye features
        self.calibration_data['points'].append(current_point)
        self.calibration_data['eye_features'].append(eye_features)
        
        # Move to the next point
        self.current_point_index += 1
        
        # Check if calibration is complete
        if self.current_point_index >= len(self.calibration_points):
            self._build_calibration_model()
            return False
            
        return True
    
    def _build_calibration_model(self):
        """Build a regression model for gaze prediction based on collected data."""
        # This is a simplified placeholder for the actual calibration model
        # In a real implementation, you would use more sophisticated 
        # machine learning techniques (SVR, Neural Networks, etc.)
        
        # For demonstration, we'll use a simple linear regression model
        # X: eye features, y: screen coordinates
        
        # Convert to numpy arrays
        X = np.array(self.calibration_data['eye_features'])
        y = np.array(self.calibration_data['points'])
        
        # PLACEHOLDER: Simple linear model
        # In a real implementation, use sklearn or other ML libraries
        # Example with sklearn:
        # from sklearn.linear_model import LinearRegression
        # model = LinearRegression()
        # model.fit(X, y)
        # self.calibration_model = model
        
        self.is_calibrated = True
        
    def predict_gaze_position(self, eye_features):
        """
        Predict the gaze position based on eye features.
        
        Args:
            eye_features: Features extracted from the eye regions
            
        Returns:
            (x, y) screen coordinates of the predicted gaze position
            Returns None if the system is not calibrated
        """
        if not self.is_calibrated or self.calibration_model is None:
            return None
            
        # Convert eye features to the format expected by the model
        X = np.array([eye_features])
        
        # PLACEHOLDER: Use the model to predict gaze position
        # In a real implementation:
        # predicted_position = self.calibration_model.predict(X)[0]
        # return predicted_position
        
        # For demo purposes, return the center of the screen
        return (self.screen_width // 2, self.screen_height // 2)
    
    def save_calibration(self, filepath="data/calibration_data/calibration.json"):
        """Save calibration data to file."""
        if not self.is_calibrated:
            return False
            
        calibration_dir = os.path.dirname(filepath)
        create_directory(calibration_dir)
        
        # Convert numpy arrays to lists for JSON serialization
        data_to_save = {
            'points': self.calibration_data['points'],
            'eye_features': self.calibration_data['eye_features'],
            'screen_dimensions': {
                'width': self.screen_width,
                'height': self.screen_height
            },
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data_to_save, f)
            
        return True
    
    def load_calibration(self, filepath="data/calibration_data/calibration.json"):
        """Load calibration data from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            self.calibration_data = {
                'points': data['points'],
                'eye_features': data['eye_features']
            }
            
            self.screen_width = data['screen_dimensions']['width']
            self.screen_height = data['screen_dimensions']['height']
            
            # Rebuild the calibration model
            self._build_calibration_model()
            
            return True
            
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Error loading calibration data: {e}")
            return False
    
    def draw_calibration_point(self, frame, point=None, size=15, color=(0, 0, 255)):
        """
        Draw a calibration point on the frame.
        
        Args:
            frame: The image frame to draw on
            point: (x, y) coordinates, if None uses current point
            size: Size of the point
            color: Color as (B, G, R)
            
        Returns:
            Frame with the drawn calibration point
        """
        if point is None:
            point = self.get_current_point()
            
        if point is None:
            return frame
            
        x, y = point
        
        # Scale points from screen coordinates to frame coordinates
        frame_h, frame_w = frame.shape[:2]
        scaled_x = int(x * frame_w / self.screen_width)
        scaled_y = int(y * frame_h / self.screen_height)
        
        # Draw the point (outer and inner circles)
        cv2.circle(frame, (scaled_x, scaled_y), size, color, 2)
        cv2.circle(frame, (scaled_x, scaled_y), 3, color, -1)
        
        return frame