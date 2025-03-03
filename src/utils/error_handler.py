import logging
import traceback
import sys
import os
from datetime import datetime

class ErrorHandler:
    """
    A class for handling errors in the dance evaluation system.
    """
    
    def __init__(self, log_file=None, log_level=logging.INFO):
        """
        Initialize the error handler.
        
        Args:
            log_file: The path to the log file. If None, a default log file is created.
            log_level: The logging level.
        """
        # Set up logging
        self.logger = logging.getLogger("DunhuangDanceEvaluation")
        self.logger.setLevel(log_level)
        
        # Create a default log file if none is provided
        if log_file is None:
            log_dir = os.path.join(os.path.expanduser("~"), "dunhuang_dance_evaluation", "logs")
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"error_log_{timestamp}.txt")
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Store error counts
        self.error_counts = {
            'video_input': 0,
            'skeleton_detection': 0,
            'feature_extraction': 0,
            'dtw_comparison': 0,
            'scoring': 0,
            'ui': 0,
            'other': 0
        }
        
        # Set error thresholds for different components
        self.error_thresholds = {
            'video_input': 5,
            'skeleton_detection': 10,
            'feature_extraction': 5,
            'dtw_comparison': 5,
            'scoring': 5,
            'ui': 3,
            'other': 10
        }
        
        # Store the last error message for each component
        self.last_error_messages = {component: None for component in self.error_counts}
    
    def handle_error(self, error, component='other', critical=False):
        """
        Handle an error.
        
        Args:
            error: The error to handle.
            component: The component where the error occurred.
            critical: Whether the error is critical.
            
        Returns:
            handled: Whether the error was handled successfully.
        """
        # Get the error message and traceback
        error_message = str(error)
        error_traceback = traceback.format_exc()
        
        # Log the error
        if critical:
            self.logger.critical(f"{component} error: {error_message}\n{error_traceback}")
        else:
            self.logger.error(f"{component} error: {error_message}\n{error_traceback}")
        
        # Increment the error count for the component
        if component in self.error_counts:
            self.error_counts[component] += 1
        else:
            self.error_counts['other'] += 1
        
        # Store the last error message
        self.last_error_messages[component] = error_message
        
        # Check if the error count exceeds the threshold
        if component in self.error_thresholds and self.error_counts[component] > self.error_thresholds[component]:
            self.logger.warning(f"{component} error count exceeds threshold ({self.error_thresholds[component]})")
            
            # Reset the error count
            self.error_counts[component] = 0
            
            # Return False to indicate that the error was not handled successfully
            return False
        
        # Return True to indicate that the error was handled successfully
        return True
    
    def handle_video_input_error(self, error, critical=False):
        """
        Handle a video input error.
        
        Args:
            error: The error to handle.
            critical: Whether the error is critical.
            
        Returns:
            handled: Whether the error was handled successfully.
        """
        return self.handle_error(error, 'video_input', critical)
    
    def handle_skeleton_detection_error(self, error, critical=False):
        """
        Handle a skeleton detection error.
        
        Args:
            error: The error to handle.
            critical: Whether the error is critical.
            
        Returns:
            handled: Whether the error was handled successfully.
        """
        return self.handle_error(error, 'skeleton_detection', critical)
    
    def handle_feature_extraction_error(self, error, critical=False):
        """
        Handle a feature extraction error.
        
        Args:
            error: The error to handle.
            critical: Whether the error is critical.
            
        Returns:
            handled: Whether the error was handled successfully.
        """
        return self.handle_error(error, 'feature_extraction', critical)
    
    def handle_dtw_comparison_error(self, error, critical=False):
        """
        Handle a DTW comparison error.
        
        Args:
            error: The error to handle.
            critical: Whether the error is critical.
            
        Returns:
            handled: Whether the error was handled successfully.
        """
        return self.handle_error(error, 'dtw_comparison', critical)
    
    def handle_scoring_error(self, error, critical=False):
        """
        Handle a scoring error.
        
        Args:
            error: The error to handle.
            critical: Whether the error is critical.
            
        Returns:
            handled: Whether the error was handled successfully.
        """
        return self.handle_error(error, 'scoring', critical)
    
    def handle_ui_error(self, error, critical=False):
        """
        Handle a UI error.
        
        Args:
            error: The error to handle.
            critical: Whether the error is critical.
            
        Returns:
            handled: Whether the error was handled successfully.
        """
        return self.handle_error(error, 'ui', critical)
    
    def get_error_counts(self):
        """
        Get the error counts.
        
        Returns:
            error_counts: A dictionary mapping components to their error counts.
        """
        return self.error_counts
    
    def get_last_error_message(self, component='other'):
        """
        Get the last error message for a component.
        
        Args:
            component: The component to get the last error message for.
            
        Returns:
            last_error_message: The last error message for the component.
        """
        return self.last_error_messages.get(component, None)
    
    def reset_error_counts(self):
        """
        Reset all error counts.
        """
        for component in self.error_counts:
            self.error_counts[component] = 0
    
    def set_error_threshold(self, component, threshold):
        """
        Set the error threshold for a component.
        
        Args:
            component: The component to set the error threshold for.
            threshold: The error threshold.
        """
        if component in self.error_thresholds:
            self.error_thresholds[component] = threshold
    
    def get_error_threshold(self, component):
        """
        Get the error threshold for a component.
        
        Args:
            component: The component to get the error threshold for.
            
        Returns:
            threshold: The error threshold for the component.
        """
        return self.error_thresholds.get(component, 0)


class ErrorHandlingDecorator:
    """
    A decorator for handling errors in functions.
    """
    
    def __init__(self, error_handler, component='other', critical=False, default_return=None):
        """
        Initialize the error handling decorator.
        
        Args:
            error_handler: The error handler to use.
            component: The component where the error occurred.
            critical: Whether errors are critical.
            default_return: The default return value if an error occurs.
        """
        self.error_handler = error_handler
        self.component = component
        self.critical = critical
        self.default_return = default_return
    
    def __call__(self, func):
        """
        Call the decorator.
        
        Args:
            func: The function to decorate.
            
        Returns:
            wrapper: The decorated function.
        """
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.error_handler.handle_error(e, self.component, self.critical)
                return self.default_return
        
        return wrapper


def setup_global_error_handler():
    """
    Set up a global error handler for uncaught exceptions.
    
    Returns:
        error_handler: The global error handler.
    """
    error_handler = ErrorHandler()
    
    def global_exception_handler(exctype, value, tb):
        error_handler.handle_error(value, 'global', True)
        sys.__excepthook__(exctype, value, tb)
    
    sys.excepthook = global_exception_handler
    
    return error_handler
