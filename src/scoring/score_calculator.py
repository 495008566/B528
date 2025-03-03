import numpy as np
from ..dtw_comparison.dtw_comparator import DTWComparator

class ScoreCalculator:
    """
    A class for calculating similarity scores between learner and reference movements.
    """
    
    def __init__(self, dtw_comparator=None):
        """
        Initialize the score calculator.
        
        Args:
            dtw_comparator: The DTW comparator to use for movement comparison.
        """
        self.dtw_comparator = dtw_comparator or DTWComparator()
        self.last_score = None
        self.last_normalized_distance = None
        self.last_path = None
        self.score_weights = {
            'overall': 1.0,
            'timing': 0.3,
            'posture': 0.3,
            'fluidity': 0.2,
            'stability': 0.2
        }
    
    def calculate_score(self, reference_features, learner_features, feature_weights=None):
        """
        Calculate the similarity score between reference and learner features.
        
        Args:
            reference_features: The reference movement features.
            learner_features: The learner movement features.
            feature_weights: Optional weights for different features.
            
        Returns:
            scores: A dictionary of scores for different aspects of the movement.
        """
        # Compare the features using DTW
        distance, path, normalized_distance = self.dtw_comparator.compare(
            reference_features, learner_features, feature_weights)
        
        # Store the results for later analysis
        self.last_normalized_distance = normalized_distance
        self.last_path = path
        
        # Calculate the overall score (0-100)
        overall_score = 100 * (1.0 / (1.0 + normalized_distance))
        
        # Analyze the path to get more detailed scores
        path_analysis = self.dtw_comparator.analyze_path(path)
        
        # Calculate timing score based on path monotonicity
        timing_score = 100 * path_analysis.get('monotonicity', 0.5)
        
        # Calculate posture score based on the normalized distance
        posture_score = 100 * (1.0 / (1.0 + normalized_distance * 2))
        
        # Calculate fluidity score based on path smoothness
        fluidity_score = 100 * path_analysis.get('smoothness', 0.5)
        
        # Calculate stability score based on diagonal ratio
        stability_score = 100 * (1.0 - path_analysis.get('diagonal_ratio', 0.5))
        
        # Ensure all scores are in the range [0, 100]
        overall_score = max(0, min(100, overall_score))
        timing_score = max(0, min(100, timing_score))
        posture_score = max(0, min(100, posture_score))
        fluidity_score = max(0, min(100, fluidity_score))
        stability_score = max(0, min(100, stability_score))
        
        # Create a dictionary of scores
        scores = {
            'overall': overall_score,
            'timing': timing_score,
            'posture': posture_score,
            'fluidity': fluidity_score,
            'stability': stability_score
        }
        
        # Store the scores
        self.last_score = scores
        
        return scores
    
    def calculate_weighted_score(self, scores=None):
        """
        Calculate the weighted score based on the score weights.
        
        Args:
            scores: A dictionary of scores. If None, the last calculated scores are used.
            
        Returns:
            weighted_score: The weighted score.
        """
        if scores is None:
            scores = self.last_score
            
        if scores is None:
            return 0.0
            
        weighted_score = 0.0
        total_weight = 0.0
        
        for aspect, score in scores.items():
            if aspect in self.score_weights:
                weight = self.score_weights[aspect]
                weighted_score += score * weight
                total_weight += weight
        
        if total_weight > 0:
            weighted_score /= total_weight
            
        return weighted_score
    
    def set_score_weights(self, weights):
        """
        Set the weights for different aspects of the score.
        
        Args:
            weights: A dictionary mapping score aspects to weights.
        """
        self.score_weights.update(weights)
    
    def get_score_weights(self):
        """
        Get the weights for different aspects of the score.
        
        Returns:
            score_weights: A dictionary mapping score aspects to weights.
        """
        return self.score_weights
    
    def get_last_score(self):
        """
        Get the last calculated score.
        
        Returns:
            last_score: The last calculated score.
        """
        return self.last_score
    
    def get_last_normalized_distance(self):
        """
        Get the last calculated normalized distance.
        
        Returns:
            last_normalized_distance: The last calculated normalized distance.
        """
        return self.last_normalized_distance
    
    def get_last_path(self):
        """
        Get the last calculated path.
        
        Returns:
            last_path: The last calculated path.
        """
        return self.last_path
    
    def get_score_description(self, score_value):
        """
        Get a description of a score value.
        
        Args:
            score_value: The score value.
            
        Returns:
            description: A description of the score.
        """
        if score_value >= 90:
            return "Excellent"
        elif score_value >= 80:
            return "Very Good"
        elif score_value >= 70:
            return "Good"
        elif score_value >= 60:
            return "Satisfactory"
        elif score_value >= 50:
            return "Needs Improvement"
        else:
            return "Poor"
