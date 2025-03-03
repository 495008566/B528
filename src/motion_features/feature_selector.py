import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

class FeatureSelector:
    """
    A class for selecting the most relevant features for dance movement analysis.
    """
    
    def __init__(self, method='pca', n_components=10):
        """
        Initialize the feature selector.
        
        Args:
            method: The feature selection method ('pca' or 'kbest').
            n_components: The number of components to select.
        """
        self.method = method
        self.n_components = n_components
        self.selector = None
        
    def fit(self, features, labels=None):
        """
        Fit the feature selector to the data.
        
        Args:
            features: The input features.
            labels: The target labels (only used for 'kbest' method).
            
        Returns:
            self: The fitted feature selector.
        """
        # Convert features to a flat numpy array
        flat_features = self._flatten_features(features)
        
        if self.method == 'pca':
            # Use PCA for feature selection
            self.selector = PCA(n_components=self.n_components)
            self.selector.fit(flat_features)
        elif self.method == 'kbest':
            # Use SelectKBest for feature selection
            if labels is None:
                raise ValueError("Labels are required for 'kbest' method.")
            
            self.selector = SelectKBest(f_classif, k=self.n_components)
            self.selector.fit(flat_features, labels)
        
        return self
    
    def transform(self, features):
        """
        Transform the features using the fitted selector.
        
        Args:
            features: The input features.
            
        Returns:
            selected_features: The selected features.
        """
        if self.selector is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        
        # Convert features to a flat numpy array
        flat_features = self._flatten_features(features)
        
        # Transform the features
        transformed_features = self.selector.transform(flat_features)
        
        return transformed_features
    
    def fit_transform(self, features, labels=None):
        """
        Fit the feature selector to the data and transform the features.
        
        Args:
            features: The input features.
            labels: The target labels (only used for 'kbest' method).
            
        Returns:
            selected_features: The selected features.
        """
        self.fit(features, labels)
        return self.transform(features)
    
    def _flatten_features(self, features):
        """
        Flatten the features into a 2D numpy array.
        
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
    
    def get_feature_importance(self):
        """
        Get the importance of each feature.
        
        Returns:
            feature_importance: The importance of each feature.
        """
        if self.selector is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        
        if self.method == 'pca':
            # For PCA, return the explained variance ratio
            return self.selector.explained_variance_ratio_
        elif self.method == 'kbest':
            # For SelectKBest, return the scores
            return self.selector.scores_
        
        return None
    
    def get_n_components(self):
        """
        Get the number of components.
        
        Returns:
            n_components: The number of components.
        """
        return self.n_components
    
    def set_n_components(self, n_components):
        """
        Set the number of components.
        
        Args:
            n_components: The number of components.
        """
        self.n_components = n_components
        
        # Reset the selector
        self.selector = None
