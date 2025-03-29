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
        
        # Detect pupils in eye regions using improved method
        pupils = self._detect_pupils_improved(eye_regions)
        
        # Refine pupil detection 
        refined_pupils = self.refine_pupils(eye_regions, pupils)
        
        # Extract eye features for gaze estimation
        eye_features = self._extract_eye_features(eye_regions, refined_pupils)
        
        # Estimate gaze point using calibration model if available
        gaze_point = None
        confidence = 0.0
        
        if self.calibrator is not None and self.calibrator.is_calibrated:
            gaze_point = self.calibrator.predict_gaze_position(eye_features)
            confidence = 0.85  # Increased confidence with better pupil detection
        else:
            # Fallback method without calibration - improved
            gaze_point = self._estimate_gaze_without_calibration(refined_pupils, eyes)
            confidence = 0.5  # Improved confidence for uncalibrated estimation
        
        return {
            'gaze_point': gaze_point,
            'pupil_left': refined_pupils['left_eye'],
            'pupil_right': refined_pupils['right_eye'],
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
    
    def _detect_pupils_improved(self, eye_regions):
        """
        Improved pupil detection using multiple techniques.
        
        Returns:
            Dictionary with 'left_eye' and 'right_eye' pupil coordinates,
            each as (x, y) relative to the eye region
        """
        pupils = {'left_eye': None, 'right_eye': None}
        
        for eye_name, eye_region in eye_regions.items():
            if eye_region is None or eye_region.size == 0:
                continue
                
            # Create a copy for visualization if needed
            eye_copy = eye_region.copy()
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            eye_clahe = clahe.apply(eye_region)
            
            # Apply bilateral filtering to reduce noise while preserving edges
            eye_filtered = cv2.bilateralFilter(eye_clahe, 10, 15, 15)
            
            # Apply multiple thresholding techniques and combine results
            pupils_candidates = []
            
            # 1. Standard binary thresholding
            _, thresh1 = cv2.threshold(eye_filtered, 35, 255, cv2.THRESH_BINARY_INV)
            
            # 2. Adaptive thresholding
            thresh2 = cv2.adaptiveThreshold(
                eye_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # 3. Otsu's thresholding
            _, thresh3 = cv2.threshold(
                eye_filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            
            # Process each threshold
            for i, thresh in enumerate([thresh1, thresh2, thresh3]):
                # Apply morphological operations
                kernel = np.ones((3, 3), np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
                
                # Find contours
                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                if contours:
                    # Filter contours by area and shape
                    valid_contours = []
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        
                        # Filter by area - pupil should be reasonably sized
                        min_area = 0.01 * eye_region.shape[0] * eye_region.shape[1]
                        max_area = 0.5 * eye_region.shape[0] * eye_region.shape[1]
                        
                        if min_area <= area <= max_area:
                            # Calculate circularity
                            perimeter = cv2.arcLength(contour, True)
                            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                            
                            # Pupil should be reasonably circular
                            if circularity > 0.6:
                                valid_contours.append(contour)
                    
                    if valid_contours:
                        # Choose the most circular contour
                        best_contour = max(valid_contours, key=lambda c: 
                                         4 * np.pi * cv2.contourArea(c) / 
                                         (cv2.arcLength(c, True) ** 2) if cv2.arcLength(c, True) > 0 else 0)
                        
                        # Calculate center
                        M = cv2.moments(best_contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            pupils_candidates.append((cx, cy, circularity))
            
            # Choose the best pupil candidate from all thresholding methods
            if pupils_candidates:
                # Sort by circularity and choose the best one
                best_pupil = max(pupils_candidates, key=lambda p: p[2])
                pupils[eye_name] = (best_pupil[0], best_pupil[1])
                
            # If no valid candidates, attempt to locate dark regions
            if pupils[eye_name] is None and eye_region.size > 0:
                h, w = eye_region.shape
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(eye_filtered)
                
                # The darkest point is likely to be the pupil
                # But we need to check if it's not just a random dark point
                px, py = min_loc
                
                # Check if the darkest point is near the center of the eye
                center_x, center_y = w // 2, h // 2
                dist_from_center = np.sqrt((px - center_x)**2 + (py - center_y)**2)
                
                # The point should be within a reasonable distance from the center
                if dist_from_center < 0.4 * max(w, h):
                    pupils[eye_name] = min_loc
        
        return pupils
    
    def _extract_eye_features(self, eye_regions, pupils):
        """
        Extract features from eye regions for gaze estimation.
        
        This method extracts relevant features that will be used by the 
        calibration model to predict the gaze point.
        
        Returns:
            A feature vector combining information from both eyes
        """
        features = []
        
        # Process left eye
        if eye_regions['left_eye'] is not None and pupils['left_eye'] is not None:
            left_eye = eye_regions['left_eye']
            px, py = pupils['left_eye']
            
            # Normalize pupil position relative to eye region size
            h, w = left_eye.shape[:2]
            norm_x = px / w if w > 0 else 0
            norm_y = py / h if h > 0 else 0
            
            # Calculate distance from center
            center_x, center_y = w / 2, h / 2
            dist_x = (px - center_x) / (w / 2)  # Normalized to [-1, 1]
            dist_y = (py - center_y) / (h / 2)  # Normalized to [-1, 1]
            
            # Add to features
            features.extend([norm_x, norm_y, dist_x, dist_y])
            
            # Extract some intensity features
            if px > 0 and py > 0 and px < w and py < h:
                # Get average intensity around pupil
                pupil_region = left_eye[
                    max(0, py-5):min(h, py+5),
                    max(0, px-5):min(w, px+5)
                ]
                if pupil_region.size > 0:
                    avg_intensity = np.mean(pupil_region)
                    features.append(avg_intensity / 255.0)  # Normalize to [0, 1]
                else:
                    features.append(0.5)
            else:
                features.append(0.5)
        else:
            # Placeholder values if eye not detected
            features.extend([0.5, 0.5, 0.0, 0.0, 0.5])
            
        # Process right eye (similar to left eye)
        if eye_regions['right_eye'] is not None and pupils['right_eye'] is not None:
            right_eye = eye_regions['right_eye']
            px, py = pupils['right_eye']
            
            h, w = right_eye.shape[:2]
            norm_x = px / w if w > 0 else 0
            norm_y = py / h if h > 0 else 0
            
            center_x, center_y = w / 2, h / 2
            dist_x = (px - center_x) / (w / 2)
            dist_y = (py - center_y) / (h / 2)
            
            features.extend([norm_x, norm_y, dist_x, dist_y])
            
            if px > 0 and py > 0 and px < w and py < h:
                pupil_region = right_eye[
                    max(0, py-5):min(h, py+5),
                    max(0, px-5):min(w, px+5)
                ]
                if pupil_region.size > 0:
                    avg_intensity = np.mean(pupil_region)
                    features.append(avg_intensity / 255.0)
                else:
                    features.append(0.5)
            else:
                features.append(0.5)
        else:
            features.extend([0.5, 0.5, 0.0, 0.0, 0.5])
        
        # Calculate interpupillary features if both pupils detected
        if pupils['left_eye'] is not None and pupils['right_eye'] is not None:
            # Add a feature representing the relative vertical alignment of the pupils
            left_norm_y = pupils['left_eye'][1] / eye_regions['left_eye'].shape[0] if eye_regions['left_eye'] is not None else 0.5
            right_norm_y = pupils['right_eye'][1] / eye_regions['right_eye'].shape[0] if eye_regions['right_eye'] is not None else 0.5
            
            # Vertical alignment difference (close to 0 means looking straight)
            v_align_diff = left_norm_y - right_norm_y
            features.append(v_align_diff)
            
            # Horizontal position difference
            left_norm_x = pupils['left_eye'][0] / eye_regions['left_eye'].shape[1] if eye_regions['left_eye'] is not None else 0.5
            right_norm_x = pupils['right_eye'][0] / eye_regions['right_eye'].shape[1] if eye_regions['right_eye'] is not None else 0.5
            
            h_pos_diff = left_norm_x - right_norm_x
            features.append(h_pos_diff)
            
            # Average of horizontal positions (useful for side-to-side gaze)
            h_pos_avg = (left_norm_x + right_norm_x) / 2
            features.append(h_pos_avg - 0.5)  # Centered around 0
            
            # Average of vertical positions (useful for up-down gaze)
            v_pos_avg = (left_norm_y + right_norm_y) / 2
            features.append(v_pos_avg - 0.5)  # Centered around 0
        else:
            # If only one eye is detected, add placeholder features
            features.extend([0.0, 0.0, 0.0, 0.0])
            
        return features
    
    def _estimate_gaze_without_calibration(self, pupils, eyes):
        """
        Improved fallback method to estimate gaze direction without calibration.
        
        This provides a better estimate based on pupil positions within the eyes
        and their relation to the eye centers.
        
        Returns:
            (x, y) coordinates as an estimate of gaze direction:
            (-1, -1) = top-left, (1, 1) = bottom-right, (0, 0) = center
        """
        # Default to center
        gaze_x, gaze_y = 0.0, 0.0
        count = 0
        weights = {'left_eye': 1.0, 'right_eye': 1.0}  # Can be adjusted based on confidence
        
        # Process left eye
        if pupils['left_eye'] is not None and eyes['left_eye'] is not None:
            px, py = pupils['left_eye']
            x, y, w, h = eyes['left_eye']
            
            # Normalize to center of eye (0,0) with range [-1, 1]
            center_x, center_y = w / 2, h / 2
            norm_x = (px - center_x) / center_x if center_x > 0 else 0
            norm_y = (py - center_y) / center_y if center_y > 0 else 0
            
            # Apply non-linear transformation to account for eye curvature
            # This helps improve accuracy at the edges of vision
            norm_x = np.sign(norm_x) * (abs(norm_x) ** 0.7)  # Reduce sensitivity
            norm_y = np.sign(norm_y) * (abs(norm_y) ** 0.7)  # Reduce sensitivity
            
            gaze_x += norm_x * weights['left_eye']
            gaze_y += norm_y * weights['left_eye']
            count += weights['left_eye']
            
        # Process right eye
        if pupils['right_eye'] is not None and eyes['right_eye'] is not None:
            px, py = pupils['right_eye']
            x, y, w, h = eyes['right_eye']
            
            center_x, center_y = w / 2, h / 2
            norm_x = (px - center_x) / center_x if center_x > 0 else 0
            norm_y = (py - center_y) / center_y if center_y > 0 else 0
            
            # Apply non-linear transformation
            norm_x = np.sign(norm_x) * (abs(norm_x) ** 0.7)
            norm_y = np.sign(norm_y) * (abs(norm_y) ** 0.7)
            
            gaze_x += norm_x * weights['right_eye']
            gaze_y += norm_y * weights['right_eye']
            count += weights['right_eye']
            
        # Average if we have data from eyes
        if count > 0:
            gaze_x /= count
            gaze_y /= count
            
            # Apply limits
            gaze_x = max(-1.0, min(1.0, gaze_x))
            gaze_y = max(-1.0, min(1.0, gaze_y))
            
        return (gaze_x, gaze_y)
    
    def refine_pupils(self, eye_regions, initial_pupils):
        """
        Refine pupil detection using gradient-based methods.
        
        This enhances pupil detection accuracy in challenging lighting conditions.
        
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
                
            # Use initial pupil location as starting point
            px, py = initial_pupils[eye_name]
            
            # Make sure we're within bounds
            h, w = eye_region.shape[:2]
            if px < 0 or py < 0 or px >= w or py >= h:
                refined_pupils[eye_name] = initial_pupils[eye_name]
                continue
                
            # Apply Canny edge detection
            edges = cv2.Canny(eye_region, 30, 100)
            
            # Apply Hough Circle Transform
            circles = cv2.HoughCircles(
                eye_region,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=h/2,
                param1=50,
                param2=30,
                minRadius=max(1, min(w, h) // 10),
                maxRadius=max(1, min(w, h) // 3)
            )
            
            # If circles are found
            if circles is not None:
                # Convert to integers
                circles = np.uint16(np.around(circles))
                
                # Find the circle closest to the initial pupil estimate
                min_dist = float('inf')
                best_circle = None
                
                for circle in circles[0, :]:
                    cx, cy, r = circle
                    dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_circle = circle
                
                # If the closest circle is within a reasonable distance
                if best_circle is not None and min_dist < w / 3:
                    refined_pupils[eye_name] = (int(best_circle[0]), int(best_circle[1]))
                else:
                    # Fallback to gradient method if Hough didn't find good circles
                    # Generate a gradient map
                    sobelx = cv2.Sobel(eye_region, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(eye_region, cv2.CV_64F, 0, 1, ksize=3)
                    magnitude = np.sqrt(sobelx**2 + sobely**2)
                    
                    # Define a region of interest around the initial pupil estimate
                    roi_size = min(w, h) // 4
                    roi_x = max(0, px - roi_size)
                    roi_y = max(0, py - roi_size)
                    roi_w = min(w - roi_x, 2 * roi_size)
                    roi_h = min(h - roi_y, 2 * roi_size)
                    
                    if roi_w > 0 and roi_h > 0:
                        # Extract region of interest
                        roi = magnitude[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                        
                        if roi.size > 0:
                            # Find the point with the highest gradient magnitude in the ROI
                            max_y, max_x = np.unravel_index(np.argmax(roi), roi.shape)
                            refined_x = roi_x + max_x
                            refined_y = roi_y + max_y
                            
                            refined_pupils[eye_name] = (refined_x, refined_y)
                        else:
                            refined_pupils[eye_name] = initial_pupils[eye_name]
                    else:
                        refined_pupils[eye_name] = initial_pupils[eye_name]
            else:
                # No circles found, use the initial estimate
                refined_pupils[eye_name] = initial_pupils[eye_name]
                
        return refined_pupils