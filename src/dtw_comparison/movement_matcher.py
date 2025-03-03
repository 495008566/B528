import numpy as np
import time
from .dtw_comparator import DTWComparator

class MovementMatcher:
    """
    A class for matching dance movements using DTW comparison.
    """
    
    def __init__(self, reference_movements=None):
        """
        Initialize the movement matcher.
        
        Args:
            reference_movements: A dictionary mapping movement names to their features.
        """
        self.reference_movements = reference_movements or {}
        self.dtw_comparator = DTWComparator()
        self.last_match_time = 0
        self.feature_weights = None
        
    def add_reference_movement(self, name, features):
        """
        Add a reference movement.
        
        Args:
            name: The name of the movement.
            features: The features of the movement.
        """
        self.reference_movements[name] = features
    
    def remove_reference_movement(self, name):
        """
        Remove a reference movement.
        
        Args:
            name: The name of the movement to remove.
        """
        if name in self.reference_movements:
            del self.reference_movements[name]
    
    def match_movement(self, learner_features, top_k=1):
        """
        Match a learner movement to the reference movements.
        
        Args:
            learner_features: The features of the learner movement.
            top_k: The number of top matches to return.
            
        Returns:
            matches: A list of tuples (movement_name, similarity_score) for the top matches.
            match_time: The time taken for matching in seconds.
        """
        start_time = time.time()
        
        matches = []
        
        for name, reference_features in self.reference_movements.items():
            # Compare the learner features with the reference features
            distance, path, normalized_distance = self.dtw_comparator.compare(
                reference_features, learner_features, self.feature_weights)
            
            # Convert distance to similarity score (higher is better)
            similarity_score = 1.0 / (1.0 + normalized_distance)
            
            matches.append((name, similarity_score, path))
        
        # Sort matches by similarity score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # Keep only the top k matches
        top_matches = matches[:top_k]
        
        # Record the match time
        self.last_match_time = time.time() - start_time
        
        return top_matches, self.last_match_time
    
    def get_last_match_time(self):
        """
        Get the time taken for the last match.
        
        Returns:
            last_match_time: The time taken for the last match in seconds.
        """
        return self.last_match_time
    
    def set_feature_weights(self, weights):
        """
        Set weights for different features.
        
        Args:
            weights: A list of weights for each feature.
        """
        self.feature_weights = weights
    
    def get_feature_weights(self):
        """
        Get the feature weights.
        
        Returns:
            feature_weights: The feature weights.
        """
        return self.feature_weights
    
    def optimize_feature_weights(self, validation_data, method='grid_search'):
        """
        Optimize feature weights using validation data.
        
        Args:
            validation_data: A list of tuples (learner_features, true_movement_name).
            method: The optimization method ('grid_search' or 'random_search').
            
        Returns:
            optimal_weights: The optimal feature weights.
            best_accuracy: The best accuracy achieved.
        """
        if method == 'grid_search':
            return self._optimize_weights_grid_search(validation_data)
        elif method == 'random_search':
            return self._optimize_weights_random_search(validation_data)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _optimize_weights_grid_search(self, validation_data, num_weights=5, weight_values=None):
        """
        Optimize feature weights using grid search.
        
        Args:
            validation_data: A list of tuples (learner_features, true_movement_name).
            num_weights: The number of weights to optimize.
            weight_values: The possible values for each weight.
            
        Returns:
            optimal_weights: The optimal feature weights.
            best_accuracy: The best accuracy achieved.
        """
        if weight_values is None:
            weight_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        # Get the first learner features to determine the number of features
        first_learner_features = validation_data[0][0]
        flat_features = self.dtw_comparator._flatten_features(first_learner_features)
        num_features = flat_features.shape[1]
        
        # Initialize optimal weights and best accuracy
        optimal_weights = [1.0] * num_features
        best_accuracy = 0.0
        
        # Perform grid search for the first num_weights features
        # (optimizing all features would be too computationally expensive)
        for i in range(min(num_weights, num_features)):
            for weight in weight_values:
                # Create a copy of the current optimal weights
                weights = optimal_weights.copy()
                
                # Set the weight for the current feature
                weights[i] = weight
                
                # Set the feature weights
                self.set_feature_weights(weights)
                
                # Evaluate the weights
                accuracy = self._evaluate_weights(validation_data)
                
                # Update optimal weights if accuracy improves
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    optimal_weights = weights.copy()
        
        # Set the optimal weights
        self.set_feature_weights(optimal_weights)
        
        return optimal_weights, best_accuracy
    
    def _optimize_weights_random_search(self, validation_data, num_iterations=100, num_weights=5):
        """
        Optimize feature weights using random search.
        
        Args:
            validation_data: A list of tuples (learner_features, true_movement_name).
            num_iterations: The number of random weight configurations to try.
            num_weights: The number of weights to optimize.
            
        Returns:
            optimal_weights: The optimal feature weights.
            best_accuracy: The best accuracy achieved.
        """
        # Get the first learner features to determine the number of features
        first_learner_features = validation_data[0][0]
        flat_features = self.dtw_comparator._flatten_features(first_learner_features)
        num_features = flat_features.shape[1]
        
        # Initialize optimal weights and best accuracy
        optimal_weights = [1.0] * num_features
        best_accuracy = 0.0
        
        # Perform random search
        for _ in range(num_iterations):
            # Generate random weights for the first num_weights features
            weights = [1.0] * num_features
            for i in range(min(num_weights, num_features)):
                weights[i] = np.random.uniform(0.1, 5.0)
            
            # Set the feature weights
            self.set_feature_weights(weights)
            
            # Evaluate the weights
            accuracy = self._evaluate_weights(validation_data)
            
            # Update optimal weights if accuracy improves
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                optimal_weights = weights.copy()
        
        # Set the optimal weights
        self.set_feature_weights(optimal_weights)
        
        return optimal_weights, best_accuracy
    
    def _evaluate_weights(self, validation_data):
        """
        Evaluate feature weights using validation data.
        
        Args:
            validation_data: A list of tuples (learner_features, true_movement_name).
            
        Returns:
            accuracy: The accuracy of the movement matcher with the current weights.
        """
        correct_matches = 0
        
        for learner_features, true_movement_name in validation_data:
            # Match the learner movement
            matches, _ = self.match_movement(learner_features, top_k=1)
            
            # Check if the top match is correct
            if matches and matches[0][0] == true_movement_name:
                correct_matches += 1
        
        # Calculate accuracy
        accuracy = correct_matches / len(validation_data) if validation_data else 0.0
        
        return accuracy
