import numpy as np
import time
from .dtw_comparator import DTWComparator
from scipy.spatial.distance import euclidean, cosine, cityblock

class ParameterTuner:
    """
    A class for tuning DTW algorithm parameters.
    """
    
    def __init__(self):
        """
        Initialize the parameter tuner.
        """
        self.dtw_comparator = DTWComparator()
        self.best_parameters = {}
        self.best_score = float('inf')
        
    def tune_radius(self, reference_features, learner_features, radius_values=None):
        """
        Tune the radius parameter for FastDTW.
        
        Args:
            reference_features: The reference movement features.
            learner_features: The learner movement features.
            radius_values: The radius values to try.
            
        Returns:
            best_radius: The best radius value.
            results: A dictionary mapping radius values to (distance, time) tuples.
        """
        if radius_values is None:
            radius_values = [1, 5, 10, 20, 50]
        
        results = {}
        best_radius = 1
        best_score = float('inf')
        
        for radius in radius_values:
            # Set the radius
            self.dtw_comparator.set_radius(radius)
            
            # Compare the features
            distance, _, normalized_distance = self.dtw_comparator.compare(reference_features, learner_features)
            
            # Get the comparison time
            comparison_time = self.dtw_comparator.get_last_comparison_time()
            
            # Store the results
            results[radius] = (normalized_distance, comparison_time)
            
            # Update best radius if the score improves
            # Score is a combination of distance and time (lower is better)
            score = normalized_distance + 0.1 * comparison_time
            if score < best_score:
                best_score = score
                best_radius = radius
        
        # Set the best radius
        self.dtw_comparator.set_radius(best_radius)
        
        # Store the best parameters
        self.best_parameters['radius'] = best_radius
        self.best_score = best_score
        
        return best_radius, results
    
    def tune_distance_metric(self, reference_features, learner_features, metrics=None):
        """
        Tune the distance metric for DTW.
        
        Args:
            reference_features: The reference movement features.
            learner_features: The learner movement features.
            metrics: The distance metrics to try.
            
        Returns:
            best_metric: The best distance metric.
            results: A dictionary mapping metric names to (distance, time) tuples.
        """
        if metrics is None:
            metrics = {
                'euclidean': euclidean,
                'cosine': cosine,
                'cityblock': cityblock
            }
        
        results = {}
        best_metric_name = 'euclidean'
        best_score = float('inf')
        
        for metric_name, metric_func in metrics.items():
            # Set the distance metric
            self.dtw_comparator.set_distance_metric(metric_func)
            
            # Compare the features
            distance, _, normalized_distance = self.dtw_comparator.compare(reference_features, learner_features)
            
            # Get the comparison time
            comparison_time = self.dtw_comparator.get_last_comparison_time()
            
            # Store the results
            results[metric_name] = (normalized_distance, comparison_time)
            
            # Update best metric if the score improves
            # Score is a combination of distance and time (lower is better)
            score = normalized_distance + 0.1 * comparison_time
            if score < best_score:
                best_score = score
                best_metric_name = metric_name
        
        # Set the best distance metric
        self.dtw_comparator.set_distance_metric(metrics[best_metric_name])
        
        # Store the best parameters
        self.best_parameters['distance_metric'] = best_metric_name
        self.best_score = best_score
        
        return best_metric_name, results
    
    def tune_all_parameters(self, reference_features, learner_features, radius_values=None, metrics=None):
        """
        Tune all DTW parameters.
        
        Args:
            reference_features: The reference movement features.
            learner_features: The learner movement features.
            radius_values: The radius values to try.
            metrics: The distance metrics to try.
            
        Returns:
            best_parameters: A dictionary of the best parameters.
            best_score: The best score achieved.
        """
        # Tune distance metric first
        best_metric_name, metric_results = self.tune_distance_metric(reference_features, learner_features, metrics)
        
        # Tune radius with the best distance metric
        best_radius, radius_results = self.tune_radius(reference_features, learner_features, radius_values)
        
        return self.best_parameters, self.best_score
    
    def get_best_parameters(self):
        """
        Get the best parameters.
        
        Returns:
            best_parameters: A dictionary of the best parameters.
        """
        return self.best_parameters
    
    def get_best_score(self):
        """
        Get the best score.
        
        Returns:
            best_score: The best score achieved.
        """
        return self.best_score
    
    def apply_best_parameters(self, dtw_comparator):
        """
        Apply the best parameters to a DTW comparator.
        
        Args:
            dtw_comparator: The DTW comparator to apply the parameters to.
        """
        if 'radius' in self.best_parameters:
            dtw_comparator.set_radius(self.best_parameters['radius'])
        
        if 'distance_metric' in self.best_parameters:
            metrics = {
                'euclidean': euclidean,
                'cosine': cosine,
                'cityblock': cityblock
            }
            dtw_comparator.set_distance_metric(metrics[self.best_parameters['distance_metric']])
