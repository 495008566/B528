import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import time

class DTWComparator:
    """
    A class for comparing dance movements using Dynamic Time Warping (DTW) algorithm.
    """
    
    def __init__(self, distance_metric=euclidean, radius=1):
        """
        Initialize the DTW comparator.
        
        Args:
            distance_metric: The distance metric to use for DTW.
            radius: The radius constraint for FastDTW.
        """
        self.distance_metric = distance_metric
        self.radius = radius
        self.last_comparison_time = 0
        self.last_path = None
        
    def compare(self, reference_features, learner_features, feature_weights=None):
        """
        Compare reference and learner features using DTW.
        
        Args:
            reference_features: The reference movement features.
            learner_features: The learner movement features.
            feature_weights: Optional weights for different features.
            
        Returns:
            distance: The DTW distance between the features.
            path: The optimal warping path.
            normalized_distance: The normalized DTW distance.
        """
        start_time = time.time()
        
        # Flatten and normalize features
        reference_flat = self._flatten_features(reference_features)
        learner_flat = self._flatten_features(learner_features)
        
        # Apply feature weights if provided
        if feature_weights is not None:
            reference_flat = self._apply_weights(reference_flat, feature_weights)
            learner_flat = self._apply_weights(learner_flat, feature_weights)
        
        # Ensure both sequences have the same number of features
        min_features = min(reference_flat.shape[1], learner_flat.shape[1])
        reference_flat = reference_flat[:, :min_features]
        learner_flat = learner_flat[:, :min_features]
        
        # Compute DTW distance
        distance, path = fastdtw(reference_flat, learner_flat, 
                                 dist=self.distance_metric, radius=self.radius)
        
        # Normalize the distance by the path length
        normalized_distance = distance / len(path)
        
        # Store the path for later analysis
        self.last_path = path
        
        # Record the comparison time
        self.last_comparison_time = time.time() - start_time
        
        return distance, path, normalized_distance
    
    def _flatten_features(self, features):
        """
        Flatten features into a 2D numpy array.
        
        Args:
            features: A dictionary of extracted features.
            
        Returns:
            flat_features: A 2D numpy array of flattened features.
        """
        flat_features = []
        
        # Get the number of frames
        num_frames = 0
        for feature_name, feature_data in features.items():
            if isinstance(feature_data, dict):
                # Handle dictionary features (e.g., joint_positions)
                for key, values in feature_data.items():
                    num_frames = max(num_frames, len(values))
            else:
                # Handle list features (e.g., center_of_mass)
                num_frames = max(num_frames, len(feature_data))
        
        # Flatten features for each frame
        for i in range(num_frames):
            frame_features = []
            
            for feature_name, feature_data in features.items():
                if isinstance(feature_data, dict):
                    # Handle dictionary features (e.g., joint_positions)
                    for key, values in feature_data.items():
                        if i < len(values):
                            # Flatten the values for this frame
                            if isinstance(values[i], list):
                                frame_features.extend(values[i])
                            else:
                                frame_features.append(values[i])
                else:
                    # Handle list features (e.g., center_of_mass)
                    if i < len(feature_data):
                        # Flatten the values for this frame
                        if isinstance(feature_data[i], list):
                            frame_features.extend(feature_data[i])
                        else:
                            frame_features.append(feature_data[i])
            
            flat_features.append(frame_features)
        
        return np.array(flat_features)
    
    def _apply_weights(self, features, weights):
        """
        Apply weights to features.
        
        Args:
            features: A 2D numpy array of features.
            weights: A list of weights for each feature.
            
        Returns:
            weighted_features: A 2D numpy array of weighted features.
        """
        # Ensure weights has the same length as features.shape[1]
        if len(weights) != features.shape[1]:
            # Truncate or pad weights as needed
            if len(weights) > features.shape[1]:
                weights = weights[:features.shape[1]]
            else:
                weights = weights + [1.0] * (features.shape[1] - len(weights))
        
        # Apply weights
        weighted_features = features * np.array(weights)
        
        return weighted_features
    
    def get_last_comparison_time(self):
        """
        Get the time taken for the last comparison.
        
        Returns:
            last_comparison_time: The time taken for the last comparison in seconds.
        """
        return self.last_comparison_time
    
    def get_last_path(self):
        """
        Get the optimal warping path from the last comparison.
        
        Returns:
            last_path: The optimal warping path from the last comparison.
        """
        return self.last_path
    
    def analyze_path(self, path):
        """
        Analyze the optimal warping path.
        
        Args:
            path: The optimal warping path.
            
        Returns:
            analysis: A dictionary containing path analysis results.
        """
        if path is None:
            return {}
        
        # Convert path to numpy array for easier analysis
        path_array = np.array(path)
        
        # Extract reference and learner indices
        reference_indices = path_array[:, 0]
        learner_indices = path_array[:, 1]
        
        # Calculate path length
        path_length = len(path)
        
        # Calculate path diagonal ratio (ideal path would be diagonal)
        diagonal_ratio = abs(reference_indices[-1] - learner_indices[-1]) / path_length
        
        # Calculate path monotonicity (how strictly the path follows a monotonic increase)
        reference_diffs = np.diff(reference_indices)
        learner_diffs = np.diff(learner_indices)
        monotonicity = (np.sum(reference_diffs > 0) / len(reference_diffs) + 
                        np.sum(learner_diffs > 0) / len(learner_diffs)) / 2
        
        # Calculate path smoothness (how smooth the path is)
        reference_smoothness = 1 - np.std(reference_diffs) / np.mean(reference_diffs) if np.mean(reference_diffs) > 0 else 0
        learner_smoothness = 1 - np.std(learner_diffs) / np.mean(learner_diffs) if np.mean(learner_diffs) > 0 else 0
        smoothness = (reference_smoothness + learner_smoothness) / 2
        
        # Calculate warping regions (where the path deviates significantly from the diagonal)
        diagonal = np.linspace(0, min(reference_indices[-1], learner_indices[-1]), path_length)
        reference_warping = np.abs(reference_indices - diagonal)
        learner_warping = np.abs(learner_indices - diagonal)
        warping_regions = np.where(reference_warping > np.mean(reference_warping) + np.std(reference_warping))[0]
        
        return {
            'path_length': path_length,
            'diagonal_ratio': diagonal_ratio,
            'monotonicity': monotonicity,
            'smoothness': smoothness,
            'warping_regions': warping_regions
        }
    
    def set_radius(self, radius):
        """
        Set the radius constraint for FastDTW.
        
        Args:
            radius: The radius constraint for FastDTW.
        """
        self.radius = radius
    
    def set_distance_metric(self, distance_metric):
        """
        Set the distance metric for DTW.
        
        Args:
            distance_metric: The distance metric to use for DTW.
        """
        self.distance_metric = distance_metric
