import cv2
import argparse
import time
import numpy as np
from src.face_detection import FaceDetector
from src.eye_detection import EyeDetector
from src.gaze_estimation import GazeEstimator
from src.calibration import Calibrator
from src.visualization import Visualizer
from utils import setup_logging, load_config, create_directory

def parse_args():
    parser = argparse.ArgumentParser(description='Eye Tracking Application')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--display', action='store_true', help='Display visualization')
    parser.add_argument('--calibrate', action='store_true', help='Run calibration')
    parser.add_argument('--load-calibration', type=str, help='Load calibration data from file')
    parser.add_argument('--heatmap', action='store_true', help='Show gaze heatmap')
    parser.add_argument('--dashboard', action='store_true', help='Show detailed dashboard')
    parser.add_argument('--record', type=str, help='Record video to file')
    parser.add_argument('--screen-width', type=int, default=1920, help='Screen width in pixels')
    parser.add_argument('--screen-height', type=int, default=1080, help='Screen height in pixels')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()

def run_calibration(cap, face_detector, eye_detector, gaze_estimator, calibrator, logger):
    """Run the calibration procedure."""
    logger.info("Starting calibration procedure...")
    logger.info("Look at the calibration points and press SPACE to confirm each point.")
    
    # Start calibration
    calibrator.start_calibration()
    calibration_complete = False
    
    while not calibration_complete:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to capture frame during calibration")
            return False
        
        # Get current calibration point
        current_point = calibrator.get_current_point()
        if current_point is None:
            logger.error("No calibration point available")
            return False
            
        # Process frame to detect eyes
        face = face_detector.detect(frame)
        eyes = None
        eye_features = None
        
        if face is not None:
            eyes = eye_detector.detect(frame, face)
            
            # Extract eye regions
            eye_regions = eye_detector.extract_eye_regions(frame, eyes)
            
            # Detect pupils and extract features for calibration
            if eyes['left_eye'] is not None or eyes['right_eye'] is not None:
                # This is a simplification - you'll need to extract actual features
                # from your gaze estimator
                gaze_info = gaze_estimator.estimate(frame, eyes)
                
                # Extract features for calibration (simplified)
                # In a real implementation, use proper eye features
                eye_features = []
                if gaze_info['pupil_left'] is not None:
                    px, py = gaze_info['pupil_left']
                    x, y, w, h = eyes['left_eye']
                    # Normalize pupil position within eye region
                    norm_x = px / w if w > 0 else 0.5
                    norm_y = py / h if h > 0 else 0.5
                    eye_features.extend([norm_x, norm_y])
                else:
                    eye_features.extend([0.5, 0.5])
                    
                if gaze_info['pupil_right'] is not None:
                    px, py = gaze_info['pupil_right']
                    x, y, w, h = eyes['right_eye']
                    norm_x = px / w if w > 0 else 0.5
                    norm_y = py / h if h > 0 else 0.5
                    eye_features.extend([norm_x, norm_y])
                else:
                    eye_features.extend([0.5, 0.5])
        
        # Draw calibration point on the frame
        frame = calibrator.draw_calibration_point(frame)
        
        # Show detected face and eyes if available
        if face is not None:
            cv2.rectangle(frame, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (0, 255, 0), 2)
            
            if eyes is not None:
                for eye_name in ['left_eye', 'right_eye']:
                    if eyes[eye_name] is not None:
                        ex, ey, ew, eh = eyes[eye_name]
                        color = (255, 0, 0) if eye_name == 'left_eye' else (0, 0, 255)
                        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), color, 2)
        
        # Display instructions
        cv2.putText(frame, "Look at the circle and press SPACE", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        point_number = calibrator.current_point_index + 1
        total_points = len(calibrator.calibration_points)
        cv2.putText(frame, f"Point {point_number}/{total_points}", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow('Calibration', frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # Process key press
        if key == ord('q'):
            logger.info("Calibration aborted by user")
            return False
        elif key == ord(' '):  # Space key
            if eye_features is not None:
                # Add calibration sample
                more_points = calibrator.add_calibration_sample(eye_features)
                if not more_points:
                    # Calibration complete
                    calibration_complete = True
                    logger.info("Calibration completed successfully")
                    
                    # Save calibration data
                    calibrator.save_calibration()
                    logger.info("Calibration data saved")
            else:
                logger.warning("No valid eye features detected. Please try again.")
    
    # Close calibration window
    cv2.destroyWindow('Calibration')
    return True

def main():
    args = parse_args()
    logger = setup_logging()
    config = load_config()
    
    # Initialize components
    face_detector = FaceDetector()
    eye_detector = EyeDetector()
    
    # Initialize screen dimensions (get from args or config)
    screen_width = args.screen_width
    screen_height = args.screen_height
    
    # Initialize calibrator
    calibrator = Calibrator(screen_width=screen_width, screen_height=screen_height)
    
    # Load existing calibration if specified
    if args.load_calibration:
        logger.info(f"Loading calibration from {args.load_calibration}")
        calibrator.load_calibration(args.load_calibration)
    
    # Initialize gaze estimator with calibrator
    gaze_estimator = GazeEstimator(calibrator=calibrator)
    
    # Initialize visualizer
    visualizer = Visualizer(
        screen_width=screen_width, 
        screen_height=screen_height,
        debug_mode=args.debug
    )
    
    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        logger.error(f"Failed to open camera {args.camera}")
        return
    
    # Set up video recording if requested
    video_writer = None
    if args.record:
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(
            args.record, fourcc, fps, (frame_width, frame_height)
        )
        logger.info(f"Recording video to {args.record}")
    
    # Run calibration if requested
    if args.calibrate:
        if run_calibration(cap, face_detector, eye_detector, gaze_estimator, calibrator, logger):
            logger.info("Calibration successful. Starting eye tracking.")
        else:
            logger.warning("Calibration failed or was aborted. Using uncalibrated tracking.")
    
    logger.info("Eye tracker started. Press 'q' to quit.")
    
    # Store historical gaze points for heatmap
    gaze_history = []
    max_history = 100  # Limit history to prevent slowdown
    
    # FPS calculation variables
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to capture frame")
            break
            
        # Process frame
        face = face_detector.detect(frame)
        eyes = None
        gaze = None
        
        if face is not None:
            eyes = eye_detector.detect(frame, face)
            gaze = gaze_estimator.estimate(frame, eyes)
            
            # Store gaze point in history
            if gaze['gaze_point'] is not None and args.heatmap:
                gaze_history.append(gaze['gaze_point'])
                # Limit history size
                if len(gaze_history) > max_history:
                    gaze_history.pop(0)
        
        # Visualize results
        if args.display:
            if args.dashboard:
                # Create dashboard view
                frame = visualizer.create_dashboard(frame, face, eyes, gaze, gaze_history if args.heatmap else None)
            elif args.heatmap and len(gaze_history) > 0:
                # Create heatmap visualization
                frame = visualizer.create_heatmap(frame, gaze_history)
                # Draw face and eyes on top
                if face is not None:
                    cv2.rectangle(frame, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (0, 255, 0), 2)
            else:
                # Basic visualization
                frame = visualizer.draw(frame, face, eyes, gaze)
            
            # Calculate and display FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            
            if elapsed_time > 1:  # Update FPS every second
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
                
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Record video if requested
        if video_writer is not None:
            video_writer.write(frame)
        
        # Display the frame
        cv2.imshow('Eye Tracker', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Handle other key commands
        elif cv2.waitKey(1) & 0xFF == ord('c'):
            # Trigger calibration during runtime
            run_calibration(cap, face_detector, eye_detector, gaze_estimator, calibrator, logger)
        
        elif cv2.waitKey(1) & 0xFF == ord('s'):
            # Save a screenshot
            screenshot_path = f"screenshots/eye_tracker_{int(time.time())}.jpg"
            create_directory("screenshots")
            cv2.imwrite(screenshot_path, frame)
            logger.info(f"Screenshot saved to {screenshot_path}")
    
    # Release resources
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    logger.info("Eye tracker stopped")

if __name__ == "__main__":
    main()