import cv2
import numpy as np
import time
import os
import json
from utils import create_directory
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import pickle

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
        self.feature_scaler = None
        
        # Enhanced calibration parameters
        self.samples_per_point = 3  # Collect multiple samples per point for better accuracy
        self.current_sample_count = 0
        self.current_point_samples = []
        
    def _generate_calibration_points(self):
        """Generate a grid of calibration points with better distribution."""
        points = []
        
        # Determine grid size (e.g., 3x3 for 9 points)
        if self.points == 9:
            rows, cols = 3, 3
        elif self.points == 16:
            rows, cols = 4, 4
        elif self.points == 5:
            # Special case: center and four corners
            margin_x = int(self.screen_width * 0.15)  # 15% margin
            margin_y = int(self.screen_height * 0.15)  # 15% margin
            
            # Center point
            center_x = self.screen_width // 2
            center_y = self.screen_height // 2
            
            # Four corners
            top_left = (margin_x, margin_y)
            top_right = (self.screen_width - margin_x, margin_y)
            bottom_left = (margin_x, self.screen_height - margin_y)
            bottom_right = (self.screen_width - margin_x, self.screen_height - margin_y)
            
            return [center_x, center_y, top_left, top_right, bottom_left, bottom_right]
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
        self.current_sample_count = 0
        self.current_point_samples = []
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
            
        # Store the eye features for the current point
        self.current_point_samples.append(eye_features)
        self.current_sample_count += 1
        
        # Check if we've collected enough samples for this point
        if self.current_sample_count >= self.samples_per_point:
            # Process the samples (average them for stability)
            avg_features = np.mean(self.current_point_samples, axis=0)
            
            # Store the point and its corresponding eye features
            self.calibration_data['points'].append(current_point)
            self.calibration_data['eye_features'].append(avg_features)
            
            # Move to the next point
            self.current_point_index += 1
            self.current_sample_count = 0
            self.current_point_samples = []
            
            # Check if calibration is complete
            if self.current_point_index >= len(self.calibration_points):
                self._build_calibration_model()
                return False
        
        return True
    
    def _build_calibration_model(self):
        """Build a regression model for gaze prediction based on collected data."""
        # Check if we have enough data
        if len(self.calibration_data['points']) < 5:
            print("Not enough calibration points. Calibration failed.")
            return False
        
        try:
            # Convert to numpy arrays
            X = np.array(self.calibration_data['eye_features'])
            y = np.array(self.calibration_data['points'])
            
            # Standardize features
            self.feature_scaler = StandardScaler()
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Create a SVR model with RBF kernel
            base_svr = SVR(kernel='rbf', C=100, gamma='auto')
            
            # Use MultiOutputRegressor for predicting both x and y coordinates
            self.calibration_model = MultiOutputRegressor(base_svr)
            
            # Fit the model
            self.calibration_model.fit(X_scaled, y)
            
            # Evaluate model on training data
            train_predictions = self.calibration_model.predict(X_scaled)
            
            # Calculate average error
            errors = np.sqrt(np.sum((train_predictions - y) ** 2, axis=1))
            avg_error = np.mean(errors)
            
            print(f"Calibration complete. Average error: {avg_error:.2f} pixels")
            
            self.is_calibrated = True
            return True
        
        except Exception as e:
            print(f"Error building calibration model: {e}")
            self.is_calibrated = False
            return False
    
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
        
        try:
            # Convert eye features to the format expected by the model
            X = np.array([eye_features])
            
            # Apply the same scaling used during training
            X_scaled = self.feature_scaler.transform(X)
            
            # Predict gaze position
            predicted_position = self.calibration_model.predict(X_scaled)[0]
            
            # Ensure we're within screen boundaries
            x = max(0, min(int(predicted_position[0]), self.screen_width))
            y = max(0, min(int(predicted_position[1]), self.screen_height))
            
            return (x, y)
        
        except Exception as e:
            print(f"Error predicting gaze position: {e}")
            return None
    
    def save_calibration(self, filepath="data/calibration_data/calibration.json"):
        """Save calibration data and model to file."""
        if not self.is_calibrated:
            return False
            
        calibration_dir = os.path.dirname(filepath)
        create_directory(calibration_dir)
        
        # Save the model separately (using pickle)
        model_path = os.path.join(calibration_dir, "calibration_model.pkl")
        scaler_path = os.path.join(calibration_dir, "feature_scaler.pkl")
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.calibration_model, f)
                
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.feature_scaler, f)
        except Exception as e:
            print(f"Error saving calibration model: {e}")
        
        # Convert numpy arrays to lists for JSON serialization
        data_to_save = {
            'points': self.calibration_data['points'],
            'eye_features': self.calibration_data['eye_features'],
            'screen_dimensions': {
                'width': self.screen_width,
                'height': self.screen_height
            },
            'model_path': model_path,
            'scaler_path': scaler_path,
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data_to_save, f)
            
        return True
    
    def load_calibration(self, filepath="data/calibration_data/calibration.json"):
        """Load calibration data and model from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            self.calibration_data = {
                'points': data['points'],
                'eye_features': data['eye_features']
            }
            
            self.screen_width = data['screen_dimensions']['width']
            self.screen_height = data['screen_dimensions']['height']
            
            # Load the model and scaler
            if 'model_path' in data and 'scaler_path' in data:
                try:
                    with open(data['model_path'], 'rb') as f:
                        self.calibration_model = pickle.load(f)
                        
                    with open(data['scaler_path'], 'rb') as f:
                        self.feature_scaler = pickle.load(f)
                        
                    self.is_calibrated = True
                except Exception as e:
                    print(f"Error loading calibration model: {e}")
                    # If model loading fails, rebuild from the calibration data
                    self._build_calibration_model()
            else:
                # Rebuild the model if paths are not in the data
                self._build_calibration_model()
            
            return self.is_calibrated
            
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
        
        # Draw animated calibration point
        # Outer circle that pulses
        pulse_size = size + int(5 * np.sin(time.time() * 3))
        cv2.circle(frame, (scaled_x, scaled_y), pulse_size, color, 2)
        
        # Inner solid circle
        cv2.circle(frame, (scaled_x, scaled_y), 5, color, -1)
        
        # Show sample count if collecting multiple samples per point
        if self.current_sample_count > 0:
            sample_text = f"{self.current_sample_count}/{self.samples_per_point}"
            cv2.putText(frame, sample_text, (scaled_x + 15, scaled_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
        
    def evaluate_calibration(self):
        """
        Evaluate the quality of the calibration.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_calibrated or self.calibration_model is None:
            return {
                'success': False,
                'message': 'System not calibrated'
            }
            
        try:
            # Use training data for evaluation
            X = np.array(self.calibration_data['eye_features'])
            y = np.array(self.calibration_data['points'])
            
            # Apply scaling
            X_scaled = self.feature_scaler.transform(X)
            
            # Make predictions
            predictions = self.calibration_model.predict(X_scaled)
            
            # Calculate errors
            errors = np.sqrt(np.sum((predictions - y) ** 2, axis=1))
            
            return {
                'success': True,
                'mean_error': float(np.mean(errors)),
                'max_error': float(np.max(errors)),
                'min_error': float(np.min(errors)),
                'std_error': float(np.std(errors)),
                'point_errors': errors.tolist()
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error evaluating calibration: {e}'
            }