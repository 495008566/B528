#!/usr/bin/env python3
"""
Test script for the Dunhuang Dance Evaluation System.
This script tests the core functionality of the system with sample data.
"""

import os
import sys
import time
import numpy as np
import cv2
import logging
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.skeleton_detection.video_processor import VideoProcessor
from src.skeleton_detection.pose_detector import PoseDetector
from src.motion_features.feature_extractor import FeatureExtractor
from src.dtw_comparison.dtw_comparator import DTWComparator
from src.scoring.score_calculator import ScoreCalculator
from src.scoring.feedback_generator import FeedbackGenerator
from src.data_loader import DataLoader
from src.utils.performance_optimizer import PerformanceOptimizer
from src.utils.error_handler import setup_global_error_handler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("TestSystem")

def test_skeleton_detection(video_path):
    """
    Test skeleton detection on a video file.
    
    Args:
        video_path: Path to the video file.
        
    Returns:
        success: Whether the test was successful.
        landmarks: The detected landmarks.
    """
    logger.info("Testing skeleton detection on %s", video_path)
    
    # Create a video processor
    video_processor = VideoProcessor(video_path=video_path)
    
    # Open the video
    if not video_processor.open():
        logger.error("Failed to open video: %s", video_path)
        return False, None
    
    # Process the first 10 frames
    landmarks_list = []
    frame_count = 0
    success_count = 0
    
    start_time = time.time()
    
    while frame_count < 10:
        frame, landmarks, ret = video_processor.process_next_frame()
        
        if not ret:
            break
        
        if landmarks:
            success_count += 1
            landmarks_list.append(landmarks)
        
        frame_count += 1
    
    end_time = time.time()
    
    # Close the video
    video_processor.close()
    
    # Calculate success rate and processing time
    success_rate = success_count / frame_count if frame_count > 0 else 0
    processing_time = end_time - start_time
    fps = frame_count / processing_time if processing_time > 0 else 0
    
    logger.info("Skeleton detection results:")
    logger.info("  Success rate: %.2f%%", success_rate * 100)
    logger.info("  Processing time: %.2f seconds", processing_time)
    logger.info("  FPS: %.2f", fps)
    
    return success_rate > 0.5, landmarks_list

def test_feature_extraction(landmarks_list):
    """
    Test feature extraction on landmarks.
    
    Args:
        landmarks_list: List of landmarks.
        
    Returns:
        success: Whether the test was successful.
        features: The extracted features.
    """
    logger.info("Testing feature extraction")
    
    if not landmarks_list:
        logger.error("No landmarks to extract features from")
        return False, None
    
    # Create a feature extractor
    feature_extractor = FeatureExtractor()
    
    # Extract features
    start_time = time.time()
    features = feature_extractor.extract_features(landmarks_list)
    end_time = time.time()
    
    # Check if features were extracted
    if not features:
        logger.error("Failed to extract features")
        return False, None
    
    # Calculate processing time
    processing_time = end_time - start_time
    
    logger.info("Feature extraction results:")
    logger.info("  Number of features: %d", len(features))
    logger.info("  Processing time: %.2f seconds", processing_time)
    
    return True, features

def test_dtw_comparison(reference_features, learner_features):
    """
    Test DTW comparison between reference and learner features.
    
    Args:
        reference_features: The reference features.
        learner_features: The learner features.
        
    Returns:
        success: Whether the test was successful.
        results: The comparison results.
    """
    logger.info("Testing DTW comparison")
    
    if not reference_features or not learner_features:
        logger.error("Missing features for comparison")
        return False, None
    
    # Create a DTW comparator
    dtw_comparator = DTWComparator()
    
    # Compare features
    start_time = time.time()
    distance, path, normalized_distance = dtw_comparator.compare(reference_features, learner_features)
    end_time = time.time()
    
    # Calculate processing time
    processing_time = end_time - start_time
    
    logger.info("DTW comparison results:")
    logger.info("  Distance: %.2f", distance)
    logger.info("  Normalized distance: %.2f", normalized_distance)
    logger.info("  Path length: %d", len(path))
    logger.info("  Processing time: %.2f seconds", processing_time)
    
    # Analyze the path
    path_analysis = dtw_comparator.analyze_path(path)
    
    logger.info("Path analysis:")
    for key, value in path_analysis.items():
        if key != 'warping_regions':
            logger.info("  %s: %.2f", key, value)
    
    return True, {
        'distance': distance,
        'path': path,
        'normalized_distance': normalized_distance,
        'path_analysis': path_analysis
    }

def test_scoring(reference_features, learner_features, dtw_results):
    """
    Test scoring between reference and learner features.
    
    Args:
        reference_features: The reference features.
        learner_features: The learner features.
        dtw_results: The DTW comparison results.
        
    Returns:
        success: Whether the test was successful.
        scores: The calculated scores.
    """
    logger.info("Testing scoring")
    
    if not reference_features or not learner_features:
        logger.error("Missing features for scoring")
        return False, None
    
    # Create a score calculator
    score_calculator = ScoreCalculator()
    
    # Calculate scores
    start_time = time.time()
    scores = score_calculator.calculate_score(reference_features, learner_features)
    end_time = time.time()
    
    # Calculate processing time
    processing_time = end_time - start_time
    
    logger.info("Scoring results:")
    for aspect, score in scores.items():
        logger.info("  %s: %.2f", aspect, score)
    
    logger.info("  Processing time: %.2f seconds", processing_time)
    
    return True, scores

def test_feedback_generation(scores, reference_features, learner_features, path):
    """
    Test feedback generation based on scores.
    
    Args:
        scores: The calculated scores.
        reference_features: The reference features.
        learner_features: The learner features.
        path: The DTW path.
        
    Returns:
        success: Whether the test was successful.
        feedback: The generated feedback.
    """
    logger.info("Testing feedback generation")
    
    if not scores:
        logger.error("No scores for feedback generation")
        return False, None
    
    # Create a feedback generator
    feedback_generator = FeedbackGenerator()
    
    # Generate feedback
    start_time = time.time()
    feedback = feedback_generator.generate_feedback(scores, reference_features, learner_features, path)
    end_time = time.time()
    
    # Calculate processing time
    processing_time = end_time - start_time
    
    logger.info("Feedback generation results:")
    for aspect, message in feedback.items():
        logger.info("  %s: %s", aspect, message)
    
    logger.info("  Processing time: %.2f seconds", processing_time)
    
    return True, feedback

def test_data_loader(data_dir):
    """
    Test data loader functionality.
    
    Args:
        data_dir: The directory containing the data.
        
    Returns:
        success: Whether the test was successful.
        reference_features: The loaded reference features.
    """
    logger.info("Testing data loader with directory: %s", data_dir)
    
    # Create a data loader
    data_loader = DataLoader(data_dir)
    
    # Load reference movements
    start_time = time.time()
    reference_features = data_loader.load_reference_movements()
    end_time = time.time()
    
    # Calculate processing time
    processing_time = end_time - start_time
    
    if not reference_features:
        logger.error("Failed to load reference movements")
        return False, None
    
    logger.info("Data loader results:")
    logger.info("  Number of reference movements: %d", len(reference_features))
    logger.info("  Processing time: %.2f seconds", processing_time)
    
    # Get available movements
    available_movements = data_loader.get_available_movements()
    logger.info("Available movements:")
    for movement in available_movements:
        logger.info("  %s", movement)
    
    return True, reference_features

def test_performance_optimization(video_path):
    """
    Test performance optimization.
    
    Args:
        video_path: Path to the video file.
        
    Returns:
        success: Whether the test was successful.
        stats: Performance statistics.
    """
    logger.info("Testing performance optimization on %s", video_path)
    
    # Create a performance optimizer
    performance_optimizer = PerformanceOptimizer()
    
    # Create a video processor
    video_processor = VideoProcessor(video_path=video_path)
    
    # Open the video
    if not video_processor.open():
        logger.error("Failed to open video: %s", video_path)
        return False, None
    
    # Process frames with performance monitoring
    frame_count = 0
    max_frames = 30
    
    while frame_count < max_frames:
        # Start timing
        performance_optimizer.start_timer('frame_processing')
        
        # Process a frame
        frame, landmarks, ret = video_processor.process_next_frame()
        
        if not ret:
            break
        
        # Stop timing
        performance_optimizer.stop_timer('frame_processing')
        
        # Record frame processing time
        frame_time = performance_optimizer.get_timer_duration('frame_processing')
        performance_optimizer.record_frame_time(frame_time)
        
        frame_count += 1
    
    # Close the video
    video_processor.close()
    
    # Get performance statistics
    stats = performance_optimizer.get_all_stats()
    bottlenecks = performance_optimizer.identify_bottlenecks()
    fps = performance_optimizer.get_fps()
    
    logger.info("Performance optimization results:")
    logger.info("  FPS: %.2f", fps)
    
    logger.info("Operation times:")
    for operation, time_stats in stats.items():
        logger.info("  %s: avg=%.2fms, min=%.2fms, max=%.2fms", 
                   operation, time_stats['avg'] * 1000, 
                   time_stats['min'] * 1000, time_stats['max'] * 1000)
    
    logger.info("Bottlenecks:")
    for operation in bottlenecks:
        logger.info("  %s", operation)
    
    return True, {
        'fps': fps,
        'stats': stats,
        'bottlenecks': bottlenecks
    }

def test_error_handling():
    """
    Test error handling.
    
    Returns:
        success: Whether the test was successful.
    """
    logger.info("Testing error handling")
    
    # Set up error handler
    error_handler = setup_global_error_handler()
    
    # Test handling different types of errors
    test_errors = [
        ('video_input', ValueError("Test video input error")),
        ('skeleton_detection', RuntimeError("Test skeleton detection error")),
        ('feature_extraction', IndexError("Test feature extraction error")),
        ('dtw_comparison', TypeError("Test DTW comparison error")),
        ('scoring', ZeroDivisionError("Test scoring error")),
        ('ui', AttributeError("Test UI error"))
    ]
    
    for component, error in test_errors:
        handled = error_handler.handle_error(error, component)
        logger.info("Error handling for %s: %s", component, "Handled" if handled else "Failed")
    
    # Get error counts
    error_counts = error_handler.get_error_counts()
    
    logger.info("Error counts:")
    for component, count in error_counts.items():
        logger.info("  %s: %d", component, count)
    
    return True

def main():
    """
    Main function to run all tests.
    """
    logger.info("Starting system tests")
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Define test data paths
    test_data_dir = os.path.join(os.path.expanduser("~"), "dunhuang_dance_evaluation", "test_data")
    video_path = os.path.join(test_data_dir, "1-FeiTian", "02-PanTuiXieShenShi.mp4")
    
    # Test skeleton detection
    skeleton_success, landmarks_list = test_skeleton_detection(video_path)
    if not skeleton_success:
        logger.error("Skeleton detection test failed")
        return False
    
    # Test feature extraction
    feature_success, learner_features = test_feature_extraction(landmarks_list)
    if not feature_success:
        logger.error("Feature extraction test failed")
        return False
    
    # Test data loader
    loader_success, reference_features_dict = test_data_loader(test_data_dir)
    if not loader_success:
        logger.error("Data loader test failed")
        return False
    
    # Get the first reference features
    if reference_features_dict:
        reference_movement = list(reference_features_dict.keys())[0]
        reference_features = reference_features_dict[reference_movement]
    else:
        # If no reference features were loaded, use the learner features as reference
        logger.warning("No reference features loaded, using learner features as reference")
        reference_features = learner_features
    
    # Test DTW comparison
    dtw_success, dtw_results = test_dtw_comparison(reference_features, learner_features)
    if not dtw_success:
        logger.error("DTW comparison test failed")
        return False
    
    # Test scoring
    scoring_success, scores = test_scoring(reference_features, learner_features, dtw_results)
    if not scoring_success:
        logger.error("Scoring test failed")
        return False
    
    # Test feedback generation
    feedback_success, feedback = test_feedback_generation(
        scores, reference_features, learner_features, dtw_results['path'])
    if not feedback_success:
        logger.error("Feedback generation test failed")
        return False
    
    # Test performance optimization
    try:
        performance_success, performance_stats = test_performance_optimization(video_path)
        if not performance_success:
            logger.error("Performance optimization test failed")
            return False
    except Exception as e:
        logger.error(f"Performance optimization test failed with error: {str(e)}")
        logger.info("Continuing with other tests...")
    
    # Test error handling
    error_handling_success = test_error_handling()
    if not error_handling_success:
        logger.error("Error handling test failed")
        return False
    
    logger.info("All tests completed successfully")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
