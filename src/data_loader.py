import os
import glob
import numpy as np
import cv2
import logging
from .skeleton_detection.bvh_parser import BVHParser
from .skeleton_detection.skeleton_extractor import SkeletonExtractor
from .motion_features.feature_extractor import FeatureExtractor

class DataLoader:
    """
    A class for loading and processing dance movement data.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the data loader.
        
        Args:
            data_dir: The directory containing the dance movement data.
        """
        self.data_dir = data_dir or os.path.join(os.path.expanduser("~"), "dunhuang_dance_evaluation", "data")
        self.skeleton_extractor = SkeletonExtractor()
        self.feature_extractor = FeatureExtractor()
        self.logger = logging.getLogger("DunhuangDanceEvaluation.DataLoader")
        
        # Create the data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize dictionaries to store loaded data
        self.reference_movements = {}
        self.reference_features = {}
    
    def load_reference_movements(self):
        """
        Load all reference movements from the data directory.
        
        Returns:
            reference_movements: A dictionary mapping movement names to their features.
        """
        self.logger.info("Loading reference movements from %s", self.data_dir)
        
        # Get all subdirectories (categories)
        categories = [d for d in os.listdir(self.data_dir) 
                     if os.path.isdir(os.path.join(self.data_dir, d))]
        
        for category in categories:
            category_dir = os.path.join(self.data_dir, category)
            
            # Load BVH files
            bvh_files = glob.glob(os.path.join(category_dir, "*.bvh"))
            for bvh_file in bvh_files:
                movement_name = os.path.splitext(os.path.basename(bvh_file))[0]
                full_name = f"{category}/{movement_name}"
                
                try:
                    # Load the BVH file
                    self.logger.info("Loading BVH file: %s", bvh_file)
                    skeleton_data = self._load_bvh_file(bvh_file)
                    
                    # Extract features
                    features = self.feature_extractor.extract_features(skeleton_data)
                    
                    # Store the features
                    self.reference_movements[full_name] = skeleton_data
                    self.reference_features[full_name] = features
                    
                    self.logger.info("Loaded reference movement: %s", full_name)
                except Exception as e:
                    self.logger.error("Error loading BVH file %s: %s", bvh_file, str(e))
        
        self.logger.info("Loaded %d reference movements", len(self.reference_features))
        
        return self.reference_features
    
    def load_reference_movement(self, movement_path):
        """
        Load a specific reference movement.
        
        Args:
            movement_path: The path to the reference movement file.
            
        Returns:
            features: The features of the reference movement.
        """
        try:
            # Check if the file exists
            if not os.path.exists(movement_path):
                self.logger.error("Reference movement file not found: %s", movement_path)
                return None
            
            # Get the file extension
            _, ext = os.path.splitext(movement_path)
            
            # Load the file based on its extension
            if ext.lower() == '.bvh':
                # Load the BVH file
                self.logger.info("Loading BVH file: %s", movement_path)
                skeleton_data = self._load_bvh_file(movement_path)
            elif ext.lower() in ['.mp4', '.avi', '.mov']:
                # Load the video file
                self.logger.info("Loading video file: %s", movement_path)
                skeleton_data = self._load_video_file(movement_path)
            else:
                self.logger.error("Unsupported file format: %s", ext)
                return None
            
            # Extract features
            features = self.feature_extractor.extract_features(skeleton_data)
            
            # Store the features
            movement_name = os.path.splitext(os.path.basename(movement_path))[0]
            self.reference_movements[movement_name] = skeleton_data
            self.reference_features[movement_name] = features
            
            self.logger.info("Loaded reference movement: %s", movement_name)
            
            return features
        except Exception as e:
            self.logger.error("Error loading reference movement %s: %s", movement_path, str(e))
            return None
    
    def _load_bvh_file(self, bvh_file):
        """
        Load a BVH file and extract skeleton data.
        
        Args:
            bvh_file: The path to the BVH file.
            
        Returns:
            skeleton_data: A list of skeleton data for each frame.
        """
        # Parse the BVH file
        bvh_parser = BVHParser(bvh_file)
        if not bvh_parser.parse():
            raise ValueError(f"Failed to parse BVH file: {bvh_file}")
        
        # Extract skeleton data for each frame
        skeleton_data = []
        for frame_idx in range(bvh_parser.get_num_frames()):
            joint_positions = bvh_parser.get_joint_positions(frame_idx)
            
            # Convert joint positions to skeleton format
            skeleton = {}
            for joint_name, position in joint_positions.items():
                skeleton[joint_name] = {
                    'position': position,
                    'visibility': 1.0  # BVH files have full visibility
                }
            
            skeleton_data.append(skeleton)
        
        return skeleton_data
    
    def _load_video_file(self, video_file):
        """
        Load a video file and extract skeleton data.
        
        Args:
            video_file: The path to the video file.
            
        Returns:
            skeleton_data: A list of skeleton data for each frame.
        """
        # Open the video file
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_file}")
        
        # Extract skeleton data for each frame
        skeleton_data = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract skeleton from the frame
            skeleton = self.skeleton_extractor.extract_from_image(frame)
            if skeleton:
                skeleton_data.append(skeleton)
        
        # Release the video capture
        cap.release()
        
        return skeleton_data
    
    def get_reference_features(self, movement_name=None):
        """
        Get the features of a reference movement.
        
        Args:
            movement_name: The name of the reference movement. If None, return all reference features.
            
        Returns:
            features: The features of the reference movement, or a dictionary of all reference features.
        """
        if movement_name is None:
            return self.reference_features
        
        return self.reference_features.get(movement_name, None)
    
    def get_reference_movements(self, movement_name=None):
        """
        Get the skeleton data of a reference movement.
        
        Args:
            movement_name: The name of the reference movement. If None, return all reference movements.
            
        Returns:
            skeleton_data: The skeleton data of the reference movement, or a dictionary of all reference movements.
        """
        if movement_name is None:
            return self.reference_movements
        
        return self.reference_movements.get(movement_name, None)
    
    def get_available_movements(self):
        """
        Get a list of available reference movements.
        
        Returns:
            movements: A list of available reference movement names.
        """
        return list(self.reference_features.keys())
