import time
import threading
from queue import Queue
from .score_calculator import ScoreCalculator
from .feedback_generator import FeedbackGenerator
from ..dtw_comparison.real_time_comparator import RealTimeComparator

class RealTimeEvaluator:
    """
    A class for real-time evaluation of dance movements.
    """
    
    def __init__(self, reference_features=None, buffer_size=30, overlap=15, 
                 similarity_threshold=0.7, feature_weights=None):
        """
        Initialize the real-time evaluator.
        
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
        
        self.real_time_comparator = RealTimeComparator(
            reference_features, buffer_size, overlap, similarity_threshold, feature_weights)
        self.score_calculator = ScoreCalculator()
        self.feedback_generator = FeedbackGenerator()
        
        self.buffer = []
        self.last_evaluation_result = None
        self.last_evaluation_time = 0
        
        # For asynchronous evaluation
        self.evaluation_queue = Queue()
        self.result_queue = Queue()
        self.is_running = False
        self.evaluation_thread = None
    
    def add_frame(self, frame_features):
        """
        Add a frame to the buffer.
        
        Args:
            frame_features: The features of the frame.
            
        Returns:
            is_buffer_full: Whether the buffer is full.
        """
        return self.real_time_comparator.add_frame(frame_features)
    
    def evaluate(self):
        """
        Evaluate the buffer against the reference features.
        
        Returns:
            evaluation_result: A dictionary containing evaluation results.
            evaluation_time: The time taken for evaluation in seconds.
        """
        start_time = time.time()
        
        # Compare the buffer with the reference features
        similarity_score, is_match, comparison_time = self.real_time_comparator.compare()
        
        # If the similarity score is below the threshold, return minimal results
        if not is_match:
            evaluation_result = {
                'similarity_score': similarity_score,
                'is_match': is_match,
                'scores': {'overall': similarity_score * 100},
                'feedback': {'overall': "Movement not recognized. Try again."}
            }
            
            self.last_evaluation_result = evaluation_result
            self.last_evaluation_time = time.time() - start_time
            
            return evaluation_result, self.last_evaluation_time
        
        # Convert buffer to features
        buffer_features = self.real_time_comparator._buffer_to_features()
        
        # Calculate scores
        scores = self.score_calculator.calculate_score(
            self.reference_features, buffer_features, self.feature_weights)
        
        # Generate feedback
        feedback = self.feedback_generator.generate_feedback(
            scores, self.reference_features, buffer_features, 
            self.real_time_comparator.get_last_path())
        
        # Create evaluation result
        evaluation_result = {
            'similarity_score': similarity_score,
            'is_match': is_match,
            'scores': scores,
            'feedback': feedback,
            'weighted_score': self.score_calculator.calculate_weighted_score(scores)
        }
        
        # Store the evaluation result
        self.last_evaluation_result = evaluation_result
        
        # Record the evaluation time
        self.last_evaluation_time = time.time() - start_time
        
        return evaluation_result, self.last_evaluation_time
    
    def start_async_evaluation(self):
        """
        Start asynchronous evaluation.
        """
        if self.is_running:
            return
        
        self.is_running = True
        self.evaluation_thread = threading.Thread(target=self._async_evaluation_worker)
        self.evaluation_thread.daemon = True
        self.evaluation_thread.start()
    
    def stop_async_evaluation(self):
        """
        Stop asynchronous evaluation.
        """
        self.is_running = False
        if self.evaluation_thread:
            self.evaluation_thread.join()
            self.evaluation_thread = None
    
    def _async_evaluation_worker(self):
        """
        Worker function for asynchronous evaluation.
        """
        while self.is_running:
            # Check if there's an evaluation request
            if not self.evaluation_queue.empty():
                # Get the buffer from the queue
                buffer = self.evaluation_queue.get()
                
                # Set the buffer
                self.buffer = buffer
                self.real_time_comparator.buffer = buffer
                
                # Perform evaluation
                evaluation_result, evaluation_time = self.evaluate()
                
                # Put the result in the result queue
                self.result_queue.put((evaluation_result, evaluation_time))
            
            # Sleep to avoid busy waiting
            time.sleep(0.01)
    
    def request_async_evaluation(self, buffer=None):
        """
        Request an asynchronous evaluation.
        
        Args:
            buffer: The buffer to evaluate. If None, the current buffer is used.
        """
        if not self.is_running:
            self.start_async_evaluation()
        
        # Use the provided buffer or the current buffer
        buffer_to_evaluate = buffer if buffer is not None else self.buffer
        
        # Put the buffer in the evaluation queue
        self.evaluation_queue.put(buffer_to_evaluate)
    
    def get_async_evaluation_result(self, block=False, timeout=None):
        """
        Get the result of an asynchronous evaluation.
        
        Args:
            block: Whether to block until a result is available.
            timeout: The timeout for blocking.
            
        Returns:
            result: The evaluation result, or None if no result is available.
        """
        try:
            return self.result_queue.get(block=block, timeout=timeout)
        except:
            return None
    
    def get_last_evaluation_result(self):
        """
        Get the result of the last evaluation.
        
        Returns:
            last_evaluation_result: The result of the last evaluation.
        """
        return self.last_evaluation_result
    
    def get_last_evaluation_time(self):
        """
        Get the time taken for the last evaluation.
        
        Returns:
            last_evaluation_time: The time taken for the last evaluation in seconds.
        """
        return self.last_evaluation_time
    
    def set_reference_features(self, reference_features):
        """
        Set the reference features.
        
        Args:
            reference_features: The reference movement features.
        """
        self.reference_features = reference_features
        self.real_time_comparator.reference_features = reference_features
    
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
        self.real_time_comparator.buffer_size = buffer_size
    
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
        self.real_time_comparator.overlap = overlap
    
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
        self.real_time_comparator.similarity_threshold = similarity_threshold
    
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
        self.real_time_comparator.feature_weights = feature_weights
    
    def get_feature_weights(self):
        """
        Get the feature weights.
        
        Returns:
            feature_weights: Weights for different features.
        """
        return self.feature_weights
