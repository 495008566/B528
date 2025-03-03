import sys
import os
import argparse
import logging
from PyQt5.QtWidgets import QApplication

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ui.main_window import MainWindow, run_application
from src.utils.error_handler import setup_global_error_handler

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        args: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Dunhuang Dance Evaluation System')
    
    parser.add_argument('--video', type=str, help='Path to a video file to process')
    parser.add_argument('--camera', type=int, default=0, help='Camera index to use (default: 0)')
    parser.add_argument('--reference', type=str, help='Path to a reference movement file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    return parser.parse_args()

def setup_logging(debug=False):
    """
    Set up logging.
    
    Args:
        debug: Whether to enable debug logging.
    """
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.expanduser("~"), "dunhuang_dance_evaluation", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'app.log')),
            logging.StreamHandler()
        ]
    )

def main():
    """
    Main entry point for the application.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.debug)
    
    # Set up error handling
    error_handler = setup_global_error_handler()
    
    # Run the application
    run_application()

if __name__ == "__main__":
    main()
