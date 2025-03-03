import cv2
import mediapipe as mp
import numpy as np
import time
from ..utils.error_handler import ErrorHandlingDecorator

class PoseDetector:
    """
    A class for detecting human poses in images and videos using MediaPipe.
    This is used as an alternative to OpenPose for skeleton point detection.
    """
    
    def __init__(self, static_image_mode=False, model_complexity=1, 
                 smooth_landmarks=True, min_detection_confidence=0.5, 
                 min_tracking_confidence=0.5):
        """
        Initialize the pose detector with MediaPipe.
        
        Args:
            static_image_mode: Whether to treat the input images as a batch of static images.
            model_complexity: Complexity of the pose landmark model (0, 1, or 2).
            smooth_landmarks: Whether to filter landmarks across frames.
            min_detection_confidence: Minimum confidence for person detection.
            min_tracking_confidence: Minimum confidence for pose tracking.
        """
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        # Define the connections between landmarks for drawing
        self.connections = self.mp_pose.POSE_CONNECTIONS
        
        # Performance metrics
        self.prev_time = 0
        self.curr_time = 0
        self.fps = 0
        
    def find_pose(self, img, draw=True):
        """
        Find the pose in an image.
        
        Args:
            img: The input image.
            draw: Whether to draw the pose landmarks on the image.
            
        Returns:
            img: The image with pose landmarks drawn if draw=True.
            results: The pose detection results.
        """
        try:
            # Check if the image is valid
            if img is None or img.size == 0:
                raise ValueError("Invalid image: image is None or empty")
            
            # Convert the BGR image to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process the image and find poses
            self.results = self.pose.process(img_rgb)
            
            # Draw the pose landmarks on the image if requested
            if draw and self.results and self.results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    img,
                    self.results.pose_landmarks,
                    self.connections,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
            return img, self.results
        except Exception as e:
            if hasattr(self, 'error_handler'):
                self.error_handler.handle_skeleton_detection_error(e)
            else:
                print(f"Error finding pose: {e}")
            
            # Return the original image and None for results
            return img, None
    
    def find_position(self, img, draw=True):
        """
        Find the positions of the pose landmarks.
        
        Args:
            img: The input image.
            draw: Whether to draw circles at the landmark positions.
            
        Returns:
            landmarks: A list of landmark positions (x, y, z, visibility).
        """
        height, width, _ = img.shape
        landmarks = []
        
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                # Convert normalized coordinates to pixel coordinates
                cx, cy = int(lm.x * width), int(lm.y * height)
                # Add the landmark position and visibility
                landmarks.append([id, cx, cy, lm.z, lm.visibility])
                
                # Draw a circle at the landmark position if requested
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                    
        return landmarks
    
    def calculate_fps(self):
        """
        Calculate the frames per second.
        
        Returns:
            fps: The frames per second.
        """
        self.curr_time = time.time()
        self.fps = 1 / (self.curr_time - self.prev_time) if (self.curr_time - self.prev_time) > 0 else 0
        self.prev_time = self.curr_time
        return self.fps
    
    def get_pose_landmarks(self):
        """
        Get the pose landmarks.
        
        Returns:
            pose_landmarks: The pose landmarks if detected, None otherwise.
        """
        if self.results.pose_landmarks:
            return self.results.pose_landmarks
        return None
    
    def get_pose_world_landmarks(self):
        """
        Get the pose world landmarks.
        
        Returns:
            pose_world_landmarks: The pose world landmarks if detected, None otherwise.
        """
        if self.results.pose_world_landmarks:
            return self.results.pose_world_landmarks
        return None
    
    def handle_detection_failure(self, img):
        """
        Handle the case where pose detection fails.
        
        Args:
            img: The input image.
            
        Returns:
            img: The image with a failure message.
        """
        try:
            if img is None:
                # Create a blank image if the input is None
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                height, width = 480, 640
            else:
                height, width, _ = img.shape
            
            # Add a red background rectangle for better visibility
            cv2.rectangle(img, (int(width/2) - 150, int(height/2) - 30), 
                         (int(width/2) + 150, int(height/2) + 10), (0, 0, 150), -1)
            
            # Add the failure message
            cv2.putText(img, "No pose detected", (int(width/2) - 140, int(height/2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Log the failure if we have an error handler
            if hasattr(self, 'error_handler'):
                self.error_handler.handle_skeleton_detection_error(
                    Exception("Pose detection failed"), critical=False)
            
            return img
        except Exception as e:
            if hasattr(self, 'error_handler'):
                self.error_handler.handle_skeleton_detection_error(e)
            else:
                print(f"Error handling detection failure: {e}")
            
            # Return a simple blank image as a last resort
            return np.zeros((480, 640, 3), dtype=np.uint8)
