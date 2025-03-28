import logging
import os
import json
import cv2
import numpy as np

def setup_logging():
    """Configure and return a logger instance."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("eye_tracker")

def load_config(config_path="config/parameters.json"):
    """Load configuration from JSON file."""
    try:
        create_directory(os.path.dirname(config_path))
        
        # Check if file exists
        if not os.path.exists(config_path):
            # Create default config file
            default_config = {
                "face_detection": {
                    "scale_factor": 1.3,
                    "min_neighbors": 5,
                    "min_size": 30
                },
                "eye_detection": {
                    "scale_factor": 1.1,
                    "min_neighbors": 5,
                    "min_size": 30
                },
                "pupil_detection": {
                    "threshold": 45,
                    "blur_size": 5
                },
                "calibration": {
                    "points": 9,
                    "point_size": 15
                },
                "visualization": {
                    "show_pupils": True,
                    "show_gaze": True,
                    "show_landmarks": True
                }
            }
            
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            
            logging.info(f"Created default config file at {config_path}")
            return default_config
            
        with open(config_path, 'r') as f:
            config_text = f.read().strip()
            if not config_text:  # File is empty
                logging.warning(f"Config file at {config_path} is empty. Using defaults.")
                return {}
            return json.loads(config_text)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.warning(f"Error loading config file: {e}. Using defaults.")
        return {}

def create_directory(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        
def image_preprocessing(image):
    """Basic image preprocessing for eye tracking."""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply histogram equalization
    gray = cv2.equalizeHist(gray)
    
    # Apply Gaussian blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    return gray