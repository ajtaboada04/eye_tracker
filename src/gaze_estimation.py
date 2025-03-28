import cv2
import numpy as np
from utils import image_preprocessing

class GazeEstimator:
    def __init__(self, calibrator=None):
        """
        Initialize the gaze estimator.
        
        Args:
            calibrator: Optional calibrator instance for gaze prediction
        """
        self.calibrator = calibrator
        
    def estimate(self, frame, eyes):
        """
        Estimate gaze direction from detected eyes.
        
        Args:
            frame: Input image frame
            eyes: Dictionary with 'left_eye' and 'right_eye' coordinates
            
        Returns:
            Dictionary containing:
                'gaze_point': (x, y) coordinates of the estimated gaze point on screen
                'pupil_left': (x, y) coordinates of the left pupil relative to the eye region
                'pupil_right': (x, y) coordinates of the right pupil relative to the eye region
                'confidence': Confidence score of the estimation (0-1)
        """
        if eyes['left_eye'] is None and eyes['right_eye'] is None:
            return {
                'gaze_point': None,
                'pupil_left': None,
                'pupil_right': None,
                'confidence': 0.0
            }
        
        # Extract eye regions
        eye_regions = self._extract_eye_regions(frame, eyes)
        
        # Detect pupils in eye regions
        pupils = self._detect_pupils(eye_regions)
        
        # Extract eye features for gaze estimation
        eye_features = self._extract_eye_features(eye_regions, pupils)
        
        # Estimate gaze point using calibration model if available
        gaze_point = None
        confidence = 0.0
        
        if self.calibrator is not None and self.calibrator.is_calibrated:
            gaze_point = self.calibrator.predict_gaze_position(eye_features)
            confidence = 0.8  # Placeholder confidence value
        else:
            # Fallback method without calibration
            gaze_point = self._estimate_gaze_without_calibration(pupils, eyes)
            confidence = 0.4  # Lower confidence for uncalibrated estimation
        
        return {
            'gaze_point': gaze_point,
            'pupil_left': pupils['left_eye'],
            'pupil_right': pupils['right_eye'],
            'confidence': confidence
        }
    
    def _extract_eye_regions(self, frame, eyes):
        """Extract and preprocess eye regions from the frame."""
        eye_regions = {'left_eye': None, 'right_eye': None}
        
        for eye_name in ['left_eye', 'right_eye']:
            if eyes[eye_name] is not None:
                x, y, w, h = eyes[eye_name]
                # Extract eye region
                eye_region = frame[y:y+h, x:x+w].copy()
                
                # Preprocess eye region for better pupil detection
                if eye_region.size > 0:
                    eye_region = image_preprocessing(eye_region)
                    eye_regions[eye_name] = eye_region
                
        return eye_regions
    
    def _detect_pupils(self, eye_regions):
        """
        Detect pupil centers in each eye region.
        
        Returns:
            Dictionary with 'left_eye' and 'right_eye' pupil coordinates,
            each as (x, y) relative to the eye region
        """
        pupils = {'left_eye': None, 'right_eye': None}
        
        for eye_name, eye_region in eye_regions.items():
            if eye_region is not None and eye_region.size > 0:
                # Apply thresholding to isolate the pupil
                _, thresh = cv2.threshold(eye_region, 45, 255, cv2.THRESH_BINARY_INV)
                
                # Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find the largest contour (likely the pupil)
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Calculate the center of the contour
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        pupils[eye_name] = (cx, cy)
                    
                    # Alternative: Minimum enclosing circle
                    # (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
                    # pupils[eye_name] = (int(cx), int(cy))
        
        return pupils
    
    def _extract_eye_features(self, eye_regions, pupils):
        """
        Extract features from eye regions for gaze estimation.
        
        This method extracts relevant features that will be used by the 
        calibration model to predict the gaze point.
        
        Returns:
            A feature vector combining information from both eyes
        """
        # This is a placeholder for the actual feature extraction
        # In a real implementation, you would extract more sophisticated features
        
        features = []
        
        # Process left eye
        if eye_regions['left_eye'] is not None and pupils['left_eye'] is not None:
            left_eye = eye_regions['left_eye']
            px, py = pupils['left_eye']
            
            # Normalize pupil position relative to eye region size
            h, w = left_eye.shape[:2]
            norm_x = px / w if w > 0 else 0
            norm_y = py / h if h > 0 else 0
            
            # Add to features
            features.extend([norm_x, norm_y])
        else:
            # Placeholder values if eye not detected
            features.extend([0.5, 0.5])
            
        # Process right eye
        if eye_regions['right_eye'] is not None and pupils['right_eye'] is not None:
            right_eye = eye_regions['right_eye']
            px, py = pupils['right_eye']
            
            # Normalize pupil position relative to eye region size
            h, w = right_eye.shape[:2]
            norm_x = px / w if w > 0 else 0
            norm_y = py / h if h > 0 else 0
            
            # Add to features
            features.extend([norm_x, norm_y])
        else:
            # Placeholder values if eye not detected
            features.extend([0.5, 0.5])
        
        # Calculate interpupillary features if both pupils detected
        if pupils['left_eye'] is not None and pupils['right_eye'] is not None:
            # Add some relative position feature
            features.append(1.0)  # Placeholder
        else:
            features.append(0.0)
            
        return features
    
    def _estimate_gaze_without_calibration(self, pupils, eyes):
        """
        Fallback method to roughly estimate gaze direction without calibration.
        
        This provides a very rough estimate based on pupil positions within the eyes.
        
        Returns:
            (x, y) coordinates as a rough estimate of gaze direction:
            (-1, -1) = top-left, (1, 1) = bottom-right, (0, 0) = center
        """
        # Default to center
        gaze_x, gaze_y = 0.0, 0.0
        count = 0
        
        # Process left eye
        if pupils['left_eye'] is not None and eyes['left_eye'] is not None:
            px, py = pupils['left_eye']
            x, y, w, h = eyes['left_eye']
            
            # Convert pupil position to -1 to 1 range
            # where 0,0 is the center of the eye
            norm_x = (px / w * 2) - 1 if w > 0 else 0
            norm_y = (py / h * 2) - 1 if h > 0 else 0
            
            gaze_x += norm_x
            gaze_y += norm_y
            count += 1
            
        # Process right eye
        if pupils['right_eye'] is not None and eyes['right_eye'] is not None:
            px, py = pupils['right_eye']
            x, y, w, h = eyes['right_eye']
            
            # Convert pupil position to -1 to 1 range
            norm_x = (px / w * 2) - 1 if w > 0 else 0
            norm_y = (py / h * 2) - 1 if h > 0 else 0
            
            gaze_x += norm_x
            gaze_y += norm_y
            count += 1
            
        # Average if we have data from both eyes
        if count > 0:
            gaze_x /= count
            gaze_y /= count
            
        return (gaze_x, gaze_y)
    
    def refine_pupils(self, eye_regions, initial_pupils):
        """
        Refine pupil detection using gradient-based methods.
        
        This is an optional enhancement that can improve pupil detection
        accuracy in challenging lighting conditions.
        
        Args:
            eye_regions: Dictionary with preprocessed eye images
            initial_pupils: Initial pupil position estimates
            
        Returns:
            Refined pupil positions
        """
        refined_pupils = {'left_eye': None, 'right_eye': None}
        
        for eye_name, eye_region in eye_regions.items():
            if eye_region is None or initial_pupils[eye_name] is None:
                continue
                
            # Generate a gradient map
            sobelx = cv2.Sobel(eye_region, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(eye_region, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # Normalize to 0-255
            magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Threshold to find regions with high gradient (pupil boundaries)
            _, thresh = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
            
            # Optional: Apply morphological operations to clean up
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours in the thresholded gradient map
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get initial pupil position
                init_x, init_y = initial_pupils[eye_name]
                
                # Find the contour closest to the initial pupil estimate
                closest_contour = None
                min_distance = float('inf')
                
                for contour in contours:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        distance = np.sqrt((cx - init_x)**2 + (cy - init_y)**2)
                        
                        if distance < min_distance:
                            min_distance = distance
                            closest_contour = contour
                
                if closest_contour is not None:
                    # Calculate refined center
                    M = cv2.moments(closest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        refined_pupils[eye_name] = (cx, cy)
                    else:
                        refined_pupils[eye_name] = initial_pupils[eye_name]
                else:
                    refined_pupils[eye_name] = initial_pupils[eye_name]
            else:
                refined_pupils[eye_name] = initial_pupils[eye_name]
                
        return refined_pupils