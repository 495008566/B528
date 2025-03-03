#!/usr/bin/env python3
"""
Demo script for the Dunhuang Dance Evaluation System.
This script demonstrates the core functionality of the system with sample data.
"""

import os
import sys
import time
import cv2
import numpy as np
import argparse
import logging
from PyQt5.QtWidgets import QApplication

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.skeleton_detection.video_processor import VideoProcessor
from src.motion_features.feature_extractor import FeatureExtractor
from src.dtw_comparison.dtw_comparator import DTWComparator
from src.scoring.score_calculator import ScoreCalculator
from src.scoring.feedback_generator import FeedbackGenerator
from src.data_loader import DataLoader
from src.utils.performance_optimizer import PerformanceOptimizer
from src.ui.main_window import MainWindow, run_application

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        args: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Dunhuang Dance Evaluation System Demo')
    
    parser.add_argument('--video', type=str, help='Path to a video file to process')
    parser.add_argument('--camera', type=int, default=0, help='Camera index to use (default: 0)')
    parser.add_argument('--reference', type=str, help='Path to a reference movement file')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no UI)')
    parser.add_argument('--output', type=str, help='Path to save output video')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    return parser.parse_args()

def setup_logging(debug=False):
    """
    Set up logging.
    
    Args:
        debug: Whether to enable debug logging.
    """
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'demo.log')),
            logging.StreamHandler()
        ]
    )

def run_headless_demo(args):
    """
    Run the demo in headless mode.
    
    Args:
        args: The command line arguments.
    """
    logger = logging.getLogger("DunhuangDanceDemo")
    logger.info("Running headless demo")
    
    # Load reference movement if specified
    reference_features = None
    if args.reference:
        logger.info("Loading reference movement: %s", args.reference)
        data_loader = DataLoader()
        reference_features = data_loader.load_reference_movement(args.reference)
        
        if not reference_features:
            logger.error("Failed to load reference movement")
            return
    
    # Create components
    video_processor = VideoProcessor(
        video_path=args.video, 
        camera_index=args.camera if not args.video else None
    )
    feature_extractor = FeatureExtractor()
    dtw_comparator = DTWComparator()
    score_calculator = ScoreCalculator()
    feedback_generator = FeedbackGenerator()
    performance_optimizer = PerformanceOptimizer()
    
    # Open video source
    if not video_processor.open():
        logger.error("Failed to open video source")
        return
    
    # Get video info
    width = int(video_processor.get_property(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_processor.get_property(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_processor.get_property(cv2.CAP_PROP_FPS)
    
    # Create output video writer if specified
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Process video
    frame_count = 0
    landmarks_buffer = []
    
    logger.info("Processing video...")
    
    while True:
        # Start timing
        performance_optimizer.start_timer('frame_processing')
        
        # Process a frame
        frame, landmarks, success = video_processor.process_next_frame()
        
        if not success:
            break
        
        # Extract features if landmarks were detected
        if landmarks:
            # Add landmarks to buffer
            landmarks_buffer.append(landmarks)
            
            # Keep only the last 30 frames
            if len(landmarks_buffer) > 30:
                landmarks_buffer = landmarks_buffer[-30:]
            
            # Extract features from the buffer
            learner_features = feature_extractor.extract_features(landmarks_buffer)
            
            # Compare with reference if available
            if reference_features and len(landmarks_buffer) >= 10:
                # Compare movements
                distance, path, normalized_distance = dtw_comparator.compare(
                    reference_features, learner_features)
                
                # Calculate scores
                scores = score_calculator.calculate_score(
                    reference_features, learner_features)
                
                # Generate feedback
                feedback = feedback_generator.generate_feedback(
                    scores, reference_features, learner_features, path)
                
                # Draw scores and feedback on frame
                cv2.putText(frame, f"Overall Score: {scores['overall']:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                y_pos = 70
                for aspect, score in scores.items():
                    if aspect != 'overall':
                        cv2.putText(frame, f"{aspect.capitalize()}: {score:.1f}", 
                                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        y_pos += 30
        
        # Write frame to output video if specified
        if out:
            out.write(frame)
        
        # Display the frame
        cv2.imshow('Dunhuang Dance Evaluation', frame)
        
        # Stop timing
        performance_optimizer.stop_timer('frame_processing')
        
        # Record frame processing time
        frame_time = performance_optimizer.get_timer_duration('frame_processing')
        performance_optimizer.record_frame_time(frame_time)
        
        # Increment frame count
        frame_count += 1
        
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            logger.info("Processed %d frames, FPS: %.2f", 
                       frame_count, performance_optimizer.get_fps())
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    video_processor.close()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    # Print performance statistics
    logger.info("Demo completed")
    logger.info("Processed %d frames", frame_count)
    logger.info("Average FPS: %.2f", performance_optimizer.get_fps())

def main():
    """
    Main entry point for the demo.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.debug)
    
    # Run in headless mode or with UI
    if args.headless:
        run_headless_demo(args)
    else:
        # Run the application with UI
        run_application(
            video_path=args.video,
            camera_index=args.camera if not args.video else None,
            reference_path=args.reference
        )

if __name__ == "__main__":
    main()
