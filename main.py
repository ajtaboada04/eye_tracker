import cv2
import argparse
import time
import numpy as np
from src.face_detection import FaceDetector
from src.eye_detection import EyeDetector
from src.face_landmark import FacialLandmarkDetector
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
    parser.add_argument('--use-landmarks', action='store_true', default=True, 
                        help='Use facial landmarks for more accurate eye detection')
    return parser.parse_args()

def run_calibration(cap, face_detector, eye_detector, landmark_detector, gaze_estimator, calibrator, logger, use_landmarks=True):
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
        face = None
        eyes = None
        landmarks = None
        
        # Use landmarks if available and enabled
        if use_landmarks and landmark_detector is not None:
            landmarks_result = landmark_detector.detect_landmarks(frame)
            if landmarks_result['success']:
                landmarks = landmarks_result['landmarks']
                face = (landmarks_result['face'].left(), 
                       landmarks_result['face'].top(),
                       landmarks_result['face'].width(),
                       landmarks_result['face'].height())
                eyes = landmark_detector.get_eye_regions(landmarks)
        
        # Fall back to basic face detection if landmarks failed
        if face is None:
            face = face_detector.detect(frame)
            if face is not None:
                eyes = eye_detector.detect(frame, face)
        
        eye_features = None
        
        if eyes and (eyes['left_eye'] is not None or eyes['right_eye'] is not None):
            # Extract eye regions
            eye_regions = eye_detector.extract_eye_regions(frame, eyes)
            
            # Get gaze info for calibration
            gaze_info = gaze_estimator.estimate(frame, eyes)
            
            if gaze_info['pupil_left'] is not None or gaze_info['pupil_right'] is not None:
                # Use the features extracted by the gaze estimator
                eye_features = gaze_estimator._extract_eye_features(
                    eye_regions, 
                    {'left_eye': gaze_info['pupil_left'], 'right_eye': gaze_info['pupil_right']}
                )
        
        # Draw calibration point on the frame
        frame = calibrator.draw_calibration_point(frame)
        
        # Show detected face and eyes if available
        vis_frame = frame.copy()
        
        if landmarks is not None:
            # Draw landmarks if available
            vis_frame = landmark_detector.visualize_landmarks(vis_frame, landmarks)
        elif face is not None:
            # Draw face rectangle
            cv2.rectangle(vis_frame, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (0, 255, 0), 2)
            
            # Draw eye rectangles
            if eyes is not None:
                for eye_name in ['left_eye', 'right_eye']:
                    if eyes[eye_name] is not None:
                        ex, ey, ew, eh = eyes[eye_name]
                        color = (255, 0, 0) if eye_name == 'left_eye' else (0, 0, 255)
                        cv2.rectangle(vis_frame, (ex, ey), (ex + ew, ey + eh), color, 2)
        
        # Display instructions
        cv2.putText(vis_frame, "Look at the circle and press SPACE", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        point_number = calibrator.current_point_index + 1
        total_points = len(calibrator.calibration_points)
        cv2.putText(vis_frame, f"Point {point_number}/{total_points}", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show calibration progress
        if calibrator.current_sample_count > 0:
            progress_text = f"Sample: {calibrator.current_sample_count}/{calibrator.samples_per_point}"
            cv2.putText(vis_frame, progress_text, (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow('Calibration', vis_frame)
        
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
                    
                    # Evaluate calibration quality
                    eval_results = calibrator.evaluate_calibration()
                    if eval_results['success']:
                        logger.info(f"Calibration quality: Mean error = {eval_results['mean_error']:.2f} pixels")
                    else:
                        logger.warning(f"Calibration evaluation failed: {eval_results['message']}")
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
    
    # Initialize facial landmark detector if requested
    landmark_detector = None
    if args.use_landmarks:
        logger.info("Initializing facial landmark detector...")
        landmark_detector = FacialLandmarkDetector()
    
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
        if run_calibration(cap, face_detector, eye_detector, landmark_detector, 
                         gaze_estimator, calibrator, logger, args.use_landmarks):
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
        face = None
        eyes = None
        landmarks = None
        gaze = None
        
        # Use landmarks if available and enabled
        if args.use_landmarks and landmark_detector is not None:
            landmarks_result = landmark_detector.detect_landmarks(frame)
            if landmarks_result['success']:
                landmarks = landmarks_result['landmarks']
                face = (landmarks_result['face'].left(), 
                       landmarks_result['face'].top(),
                       landmarks_result['face'].width(),
                       landmarks_result['face'].height())
                eyes = landmark_detector.get_eye_regions(landmarks)
        
        # Fall back to basic face detection if landmarks failed
        if face is None:
            face = face_detector.detect(frame)
            if face is not None:
                eyes = eye_detector.detect(frame, face)
        
        # Estimate gaze if eyes were detected
        if eyes and (eyes['left_eye'] is not None or eyes['right_eye'] is not None):
            gaze = gaze_estimator.estimate(frame, eyes)
            
            # Store gaze point in history for heatmap
            if gaze['gaze_point'] is not None and args.heatmap:
                gaze_history.append(gaze['gaze_point'])
                # Limit history size
                if len(gaze_history) > max_history:
                    gaze_history.pop(0)
        
        # Visualize results
        if args.display:
            if args.dashboard:
                # Create dashboard view with landmarks if available
                vis_frame = visualizer.create_dashboard(
                    frame, face, eyes, gaze, 
                    gaze_history if args.heatmap else None
                )
                
                # Add landmarks visualization if available
                if landmarks is not None and args.use_landmarks:
                    dashboard_h, dashboard_w = vis_frame.shape[:2]
                    frame_w = frame.shape[1]
                    
                    # Draw landmarks on the original frame section of the dashboard
                    landmarks_vis = landmark_detector.visualize_landmarks(
                        vis_frame[:, :frame_w].copy(), landmarks
                    )
                    vis_frame[:, :frame_w] = landmarks_vis
            elif args.heatmap and len(gaze_history) > 0:
                # Create heatmap visualization
                vis_frame = visualizer.create_heatmap(frame, gaze_history)
                
                # Draw landmarks if available on top of heatmap
                if landmarks is not None and args.use_landmarks:
                    vis_frame = landmark_detector.visualize_landmarks(vis_frame, landmarks)
                elif face is not None:
                    cv2.rectangle(vis_frame, (face[0], face[1]), 
                                 (face[0] + face[2], face[1] + face[3]), (0, 255, 0), 2)
            else:
                # Basic visualization
                vis_frame = visualizer.draw(frame, face, eyes, gaze)
                
                # Draw landmarks if available
                if landmarks is not None and args.use_landmarks:
                    vis_frame = landmark_detector.visualize_landmarks(vis_frame, landmarks)
            
            # Calculate and display FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            
            if elapsed_time > 1:  # Update FPS every second
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
                
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
            # Show gaze coordinates on screen if available
            if gaze and gaze['gaze_point'] is not None:
                if calibrator.is_calibrated:
                    # Show screen coordinates if calibrated
                    gx, gy = gaze['gaze_point']
                    cv2.putText(vis_frame, f"Gaze: ({int(gx)}, {int(gy)})", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Show normalized coordinates if not calibrated
                    gx, gy = gaze['gaze_point']
                    cv2.putText(vis_frame, f"Gaze: ({gx:.2f}, {gy:.2f})", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Record video if requested
            if video_writer is not None:
                video_writer.write(vis_frame)
            
            # Display the frame
            cv2.imshow('Eye Tracker', vis_frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Handle other key commands
        elif cv2.waitKey(1) & 0xFF == ord('c'):
            # Trigger calibration during runtime
            run_calibration(cap, face_detector, eye_detector, landmark_detector, 
                          gaze_estimator, calibrator, logger, args.use_landmarks)
        
        elif cv2.waitKey(1) & 0xFF == ord('s'):
            # Save a screenshot
            screenshot_path = f"screenshots/eye_tracker_{int(time.time())}.jpg"
            create_directory("screenshots")
            cv2.imwrite(screenshot_path, vis_frame if 'vis_frame' in locals() else frame)
            logger.info(f"Screenshot saved to {screenshot_path}")
    
    # Release resources
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    logger.info("Eye tracker stopped")

if __name__ == "__main__":
    main()