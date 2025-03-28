import cv2
import numpy as np

class Visualizer:
    def __init__(self, screen_width=1920, screen_height=1080, show_pupils=True, 
                 show_gaze=True, show_landmarks=True, debug_mode=False):
        """
        Initialize the visualization module.
        
        Args:
            screen_width: Width of the screen in pixels
            screen_height: Height of the screen in pixels
            show_pupils: Whether to visualize detected pupils
            show_gaze: Whether to visualize gaze direction
            show_landmarks: Whether to visualize facial landmarks
            debug_mode: Whether to show additional debug information
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.show_pupils = show_pupils
        self.show_gaze = show_gaze
        self.show_landmarks = show_landmarks
        self.debug_mode = debug_mode
        
    def draw(self, frame, face=None, eyes=None, gaze=None):
        """
        Draw visualization elements on the frame.
        
        Args:
            frame: Input image frame
            face: Face coordinates as (x, y, w, h)
            eyes: Dictionary with 'left_eye' and 'right_eye' coordinates
            gaze: Dictionary with gaze estimation results
            
        Returns:
            Frame with visualizations added
        """
        # Create a copy of the frame to avoid modifying the original
        vis_frame = frame.copy()
        
        # Draw face rectangle
        if face is not None:
            self._draw_face(vis_frame, face)
        
        # Draw eye regions
        if eyes is not None:
            self._draw_eyes(vis_frame, eyes)
        
        # Draw gaze information
        if gaze is not None and self.show_gaze:
            self._draw_gaze(vis_frame, gaze, eyes)
        
        # Draw debug information
        if self.debug_mode:
            self._draw_debug_info(vis_frame, face, eyes, gaze)
        
        return vis_frame
    
    def _draw_face(self, frame, face):
        """Draw face rectangle on the frame."""
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add label
        cv2.putText(frame, "Face", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def _draw_eyes(self, frame, eyes):
        """Draw eye regions and pupils on the frame."""
        for eye_name in ['left_eye', 'right_eye']:
            if eyes[eye_name] is not None:
                x, y, w, h = eyes[eye_name]
                
                # Draw eye rectangle
                color = (255, 0, 0) if eye_name == 'left_eye' else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Add label
                label = "Left Eye" if eye_name == 'left_eye' else "Right Eye"
                cv2.putText(frame, label, (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _draw_gaze(self, frame, gaze, eyes):
        """Draw gaze information on the frame."""
        # Draw pupils if available
        if self.show_pupils:
            for eye_name in ['left_eye', 'right_eye']:
                pupil_key = 'pupil_left' if eye_name == 'left_eye' else 'pupil_right'
                
                if gaze[pupil_key] is not None and eyes[eye_name] is not None:
                    # Get pupil coordinates (relative to eye region)
                    px, py = gaze[pupil_key]
                    
                    # Get eye region coordinates
                    ex, ey, ew, eh = eyes[eye_name]
                    
                    # Calculate absolute pupil position in the frame
                    abs_px = ex + px
                    abs_py = ey + py
                    
                    # Draw pupil
                    color = (0, 255, 255)  # Yellow
                    cv2.circle(frame, (abs_px, abs_py), 3, color, -1)
                    
                    # Draw gaze vector only if we have a gaze point
                    if gaze['gaze_point'] is not None:
                        # For the uncalibrated case, gaze_point is in [-1, 1] range
                        # Convert to frame coordinates for visualization
                        gaze_x, gaze_y = gaze['gaze_point']
                        
                        # If the gaze estimator is calibrated, gaze_point will be in screen coordinates
                        # We need to scale them to frame coordinates
                        if abs(gaze_x) > 1 or abs(gaze_y) > 1:
                            # Assume gaze_point is in screen coordinates
                            frame_h, frame_w = frame.shape[:2]
                            gaze_x = int(gaze_x * frame_w / self.screen_width)
                            gaze_y = int(gaze_y * frame_h / self.screen_height)
                            
                            # Draw a point at the gaze position
                            cv2.circle(frame, (gaze_x, gaze_y), 10, (0, 0, 255), -1)
                        else:
                            # For uncalibrated mode, draw a line indicating gaze direction
                            # Scale and shift the direction vector
                            line_length = 50
                            end_x = abs_px + int(gaze_x * line_length)
                            end_y = abs_py + int(gaze_y * line_length)
                            
                            # Draw line from pupil to gaze direction
                            cv2.line(frame, (abs_px, abs_py), (end_x, end_y), (0, 255, 0), 2)
        
        # Display confidence
        if 'confidence' in gaze:
            conf_text = f"Confidence: {gaze['confidence']:.2f}"
            cv2.putText(frame, conf_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def _draw_debug_info(self, frame, face, eyes, gaze):
        """Draw additional debug information on the frame."""
        # Add frame dimensions
        h, w = frame.shape[:2]
        dim_text = f"Frame: {w}x{h}"
        cv2.putText(frame, dim_text, (10, h - 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add face dimensions if available
        if face is not None:
            x, y, w, h = face
            face_text = f"Face: ({x}, {y}, {w}, {h})"
            cv2.putText(frame, face_text, (10, frame.shape[0] - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add gaze point if available
        if gaze is not None and gaze['gaze_point'] is not None:
            gx, gy = gaze['gaze_point']
            gaze_text = f"Gaze: ({gx:.2f}, {gy:.2f})"
            cv2.putText(frame, gaze_text, (10, frame.shape[0] - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        # Add fps counter (placeholder - would need actual timing implementation)
        fps_text = "FPS: --"
        cv2.putText(frame, fps_text, (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def create_heatmap(self, frame, gaze_points, alpha=0.7):
        """
        Create a heatmap visualization of gaze points.
        
        Args:
            frame: Input image frame
            gaze_points: List of (x, y) gaze points
            alpha: Transparency of the heatmap overlay
            
        Returns:
            Frame with heatmap overlay
        """
        if not gaze_points:
            return frame
            
        # Create a blank heatmap
        heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        
        # Add Gaussian for each gaze point
        for gx, gy in gaze_points:
            # Convert screen coordinates to frame coordinates if needed
            if abs(gx) > 1 or abs(gy) > 1:
                # Assume coordinates are in screen space
                frame_h, frame_w = frame.shape[:2]
                x = int(gx * frame_w / self.screen_width)
                y = int(gy * frame_h / self.screen_height)
            else:
                # Assume coordinates are in normalized [-1, 1] space
                frame_h, frame_w = frame.shape[:2]
                x = int((gx + 1) * frame_w / 2)
                y = int((gy + 1) * frame_h / 2)
                
            # Skip if the point is outside the frame
            if x < 0 or y < 0 or x >= frame.shape[1] or y >= frame.shape[0]:
                continue
                
            # Create a Gaussian around this point
            sigma = 50  # Standard deviation of the Gaussian
            for i in range(max(0, x - 3*sigma), min(frame.shape[1], x + 3*sigma)):
                for j in range(max(0, y - 3*sigma), min(frame.shape[0], y + 3*sigma)):
                    heatmap[j, i] += np.exp(-((i - x)**2 + (j - y)**2) / (2 * sigma**2))
        
        # Normalize the heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
            
        # Convert to color heatmap
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Create a mask of non-zero elements
        mask = (heatmap > 0).astype(np.uint8) * 255
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask = mask / 255.0
        
        # Blend the heatmap with the frame
        blend = frame.copy()
        for c in range(3):
            blend[:, :, c] = (1 - mask * alpha) * frame[:, :, c] + mask * alpha * heatmap_colored[:, :, c]
            
        return blend.astype(np.uint8)
    
    def create_dashboard(self, frame, face=None, eyes=None, gaze=None, history=None):
        """
        Create a comprehensive dashboard with eye tracking visualizations.
        
        Args:
            frame: Input image frame
            face: Face coordinates
            eyes: Eye coordinates
            gaze: Gaze information
            history: List of historical gaze points
            
        Returns:
            Dashboard visualization
        """
        # Create a larger canvas for the dashboard
        frame_h, frame_w = frame.shape[:2]
        dashboard = np.zeros((frame_h, frame_w * 2, 3), dtype=np.uint8)
        
        # Draw the main visualization on the left
        vis_frame = self.draw(frame, face, eyes, gaze)
        dashboard[:, :frame_w] = vis_frame
        
        # Right panel for additional visualizations
        right_panel = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        
        # Draw historical heatmap if available
        if history is not None and len(history) > 0:
            heatmap = self.create_heatmap(frame, history)
            right_panel[:, :] = heatmap
        
        # Draw eye regions in detail
        if eyes is not None and gaze is not None:
            # Extract eye regions for detailed view
            for eye_name in ['left_eye', 'right_eye']:
                if eyes[eye_name] is not None:
                    x, y, w, h = eyes[eye_name]
                    eye_region = frame[y:y+h, x:x+w].copy()
                    
                    # Resize for better visibility
                    display_w = 200
                    display_h = int(h * display_w / w)
                    eye_region_resized = cv2.resize(eye_region, (display_w, display_h))
                    
                    # Position in right panel
                    offset_y = 50 if eye_name == 'left_eye' else 50 + display_h + 20
                    
                    # Make sure it fits
                    if offset_y + display_h < right_panel.shape[0]:
                        # Create a region to copy into
                        roi = right_panel[offset_y:offset_y+display_h, 30:30+display_w]
                        
                        # Only copy if shapes match (avoid errors)
                        if roi.shape[:2] == eye_region_resized.shape[:2]:
                            right_panel[offset_y:offset_y+display_h, 30:30+display_w] = eye_region_resized
                    
                    # Add label
                    label = "Left Eye" if eye_name == 'left_eye' else "Right Eye"
                    cv2.putText(right_panel, label, (30, offset_y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add the right panel to the dashboard
        dashboard[:, frame_w:] = right_panel
        
        return dashboard