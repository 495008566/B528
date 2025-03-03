import numpy as np
from .dtw_comparator import DTWComparator
from ..utils.performance_optimizer import PerformanceOptimizer

class RealTimeComparator:
    """
    A class for real-time comparison of dance movements using DTW.
    """
    
    def __init__(self, reference_features=None, buffer_size=30, overlap=15, 
                 similarity_threshold=0.7, feature_weights=None):
        """
        Initialize the real-time comparator.
        
        Args:
            reference_features: The reference movement features.
            buffer_size: The size of the buffer for real-time comparison.
            overlap: The overlap between consecutive buffers.
            similarity_threshold: The threshold for considering a match.
            feature_weights: Optional weights for different features.
        """
        self.reference_features = reference_features
        self.buffer_size = buffer_size
        self.overlap = overlap
        self.similarity_threshold = similarity_threshold
        self.feature_weights = feature_weights
        
        self.dtw_comparator = DTWComparator()
        self.performance_optimizer = PerformanceOptimizer()
        
        self.buffer = []
        self.last_distance = None
        self.last_path = None
        self.last_normalized_distance = None
        self.last_similarity_score = None
    
    def add_frame(self, frame_features):
        """
        Add a frame to the buffer.
        
        Args:
            frame_features: The features of the frame.
            
        Returns:
            is_buffer_full: Whether the buffer is full.
        """
        # Add the frame features to the buffer
        self.buffer.append(frame_features)
        
        # If the buffer is full, remove the oldest frame
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
        
        # Return whether the buffer is full
        return len(self.buffer) >= self.buffer_size
    
    def compare(self):
        """
        Compare the buffer with the reference features.
        
        Returns:
            similarity_score: The similarity score between the buffer and the reference features.
            is_match: Whether the similarity score is above the threshold.
            comparison_time: The time taken for comparison in seconds.
        """
        if not self.reference_features or len(self.buffer) < self.buffer_size:
            return 0.0, False, 0.0
        
        # Start timing
        self.performance_optimizer.start_timer('dtw_comparison')
        
        # Convert buffer to features
        buffer_features = self._buffer_to_features()
        
        # Compare the buffer with the reference features
        distance, path, normalized_distance = self.dtw_comparator.compare(
            self.reference_features, buffer_features, self.feature_weights)
        
        # Calculate similarity score (0-1)
        similarity_score = 1.0 / (1.0 + normalized_distance)
        
        # Store the results
        self.last_distance = distance
        self.last_path = path
        self.last_normalized_distance = normalized_distance
        self.last_similarity_score = similarity_score
        
        # Stop timing
        self.performance_optimizer.stop_timer('dtw_comparison')
        comparison_time = self.performance_optimizer.get_timer_duration('dtw_comparison')
        
        # Check if the similarity score is above the threshold
        is_match = similarity_score >= self.similarity_threshold
        
        return similarity_score, is_match, comparison_time
    
    def _buffer_to_features(self):
        """
        Convert the buffer to features.
        
        Returns:
            features: The features of the buffer.
        """
        # Combine the features from all frames in the buffer
        combined_features = {}
        
        for frame_idx, frame_features in enumerate(self.buffer):
            for feature_name, feature_data in frame_features.items():
                if feature_name not in combined_features:
                    combined_features[feature_name] = {}
                
                if isinstance(feature_data, dict):
                    # Handle dictionary features (e.g., joint_positions)
                    for key, value in feature_data.items():
                        if key not in combined_features[feature_name]:
                            combined_features[feature_name][key] = []
                        
                        combined_features[feature_name][key].append(value)
                else:
                    # Handle list features (e.g., center_of_mass)
                    if 'values' not in combined_features[feature_name]:
                        combined_features[feature_name]['values'] = []
                    
                    combined_features[feature_name]['values'].append(feature_data)
        
        return combined_features
    
    def get_last_distance(self):
        """
        Get the last DTW distance.
        
        Returns:
            last_distance: The last DTW distance.
        """
        return self.last_distance
    
    def get_last_path(self):
        """
        Get the last DTW path.
        
        Returns:
            last_path: The last DTW path.
        """
        return self.last_path
    
    def get_last_normalized_distance(self):
        """
        Get the last normalized DTW distance.
        
        Returns:
            last_normalized_distance: The last normalized DTW distance.
        """
        return self.last_normalized_distance
    
    def get_last_similarity_score(self):
        """
        Get the last similarity score.
        
        Returns:
            last_similarity_score: The last similarity score.
        """
        return self.last_similarity_score
    
    def set_reference_features(self, reference_features):
        """
        Set the reference features.
        
        Args:
            reference_features: The reference movement features.
        """
        self.reference_features = reference_features
    
    def get_reference_features(self):
        """
        Get the reference features.
        
        Returns:
            reference_features: The reference movement features.
        """
        return self.reference_features
    
    def set_buffer_size(self, buffer_size):
        """
        Set the buffer size.
        
        Args:
            buffer_size: The size of the buffer for real-time comparison.
        """
        self.buffer_size = buffer_size
        
        # Adjust the buffer if necessary
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
    
    def get_buffer_size(self):
        """
        Get the buffer size.
        
        Returns:
            buffer_size: The size of the buffer for real-time comparison.
        """
        return self.buffer_size
    
    def set_overlap(self, overlap):
        """
        Set the overlap between consecutive buffers.
        
        Args:
            overlap: The overlap between consecutive buffers.
        """
        self.overlap = overlap
    
    def get_overlap(self):
        """
        Get the overlap between consecutive buffers.
        
        Returns:
            overlap: The overlap between consecutive buffers.
        """
        return self.overlap
    
    def set_similarity_threshold(self, similarity_threshold):
        """
        Set the similarity threshold.
        
        Args:
            similarity_threshold: The threshold for considering a match.
        """
        self.similarity_threshold = similarity_threshold
    
    def get_similarity_threshold(self):
        """
        Get the similarity threshold.
        
        Returns:
            similarity_threshold: The threshold for considering a match.
        """
        return self.similarity_threshold
    
    def set_feature_weights(self, feature_weights):
        """
        Set the feature weights.
        
        Args:
            feature_weights: Weights for different features.
        """
        self.feature_weights = feature_weights
    
    def get_feature_weights(self):
        """
        Get the feature weights.
        
        Returns:
            feature_weights: Weights for different features.
        """
        return self.feature_weights
    
    def clear_buffer(self):
        """
        Clear the buffer.
        """
        self.buffer = []
