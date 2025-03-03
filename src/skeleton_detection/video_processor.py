import cv2
import numpy as np
import time
import os
from .pose_detector import PoseDetector
from ..utils.error_handler import ErrorHandlingDecorator

class VideoProcessor:
    """
    A class for processing videos and extracting pose information.
    """
    
    def __init__(self, video_path=None, camera_id=0, use_camera=False):
        """
        Initialize the video processor.
        
        Args:
            video_path: Path to the video file.
            camera_id: Camera ID for webcam capture.
            use_camera: Whether to use the camera instead of a video file.
        """
        self.video_path = video_path
        self.camera_id = camera_id
        self.use_camera = use_camera
        self.cap = None
        self.pose_detector = PoseDetector()
        self.frame_count = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        self.is_opened = False
        
    def open(self):
        """
        Open the video capture.
        
        Returns:
            success: True if the video capture was opened successfully, False otherwise.
        """
        try:
            if self.use_camera:
                self.cap = cv2.VideoCapture(self.camera_id)
            else:
                # Check if the video file exists
                if not os.path.exists(self.video_path):
                    raise FileNotFoundError(f"Video file not found: {self.video_path}")
                
                self.cap = cv2.VideoCapture(self.video_path)
                
            if not self.cap.isOpened():
                raise IOError("Could not open video capture.")
                
            self.is_opened = True
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            return True
        except FileNotFoundError as e:
            if hasattr(self, 'error_handler'):
                self.error_handler.handle_video_input_error(e, critical=True)
            else:
                print(f"Error opening video file: {e}")
            return False
        except IOError as e:
            if hasattr(self, 'error_handler'):
                self.error_handler.handle_video_input_error(e, critical=True)
            else:
                print(f"Error opening video capture: {e}")
            return False
        except Exception as e:
            if hasattr(self, 'error_handler'):
                self.error_handler.handle_video_input_error(e)
            else:
                print(f"Unexpected error opening video capture: {e}")
            return False
    
    def close(self):
        """
        Close the video capture.
        """
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
    
    def process_frame(self, frame, draw=True):
        """
        Process a single frame to detect poses.
        
        Args:
            frame: The input frame.
            draw: Whether to draw the pose landmarks on the frame.
            
        Returns:
            frame: The processed frame with pose landmarks drawn if draw=True.
            landmarks: A list of landmark positions.
        """
        if frame is None:
            return None, []
            
        # Find pose in the frame
        frame, results = self.pose_detector.find_pose(frame, draw)
        
        # Find the positions of the landmarks
        landmarks = self.pose_detector.find_position(frame, draw)
        
        # Calculate FPS
        fps = self.pose_detector.calculate_fps()
        
        # Display FPS on the frame
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, landmarks
    
    def process_video(self, output_path=None, display=True, max_frames=None):
        """
        Process the entire video.
        
        Args:
            output_path: Path to save the processed video.
            display: Whether to display the processed frames.
            max_frames: Maximum number of frames to process.
            
        Returns:
            all_landmarks: A list of landmarks for each frame.
        """
        if not self.is_opened and not self.open():
            return []
            
        all_landmarks = []
        frame_idx = 0
        
        # Create video writer if output path is provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        while True:
            # Break if max_frames is reached
            if max_frames is not None and frame_idx >= max_frames:
                break
                
            # Read a frame
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Process the frame
            processed_frame, landmarks = self.process_frame(frame)
            
            # Store landmarks
            all_landmarks.append(landmarks)
            
            # Write the frame to the output video
            if out is not None:
                out.write(processed_frame)
                
            # Display the frame
            if display:
                cv2.imshow("Processed Video", processed_frame)
                
                # Break if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            frame_idx += 1
        
        # Release resources
        if out is not None:
            out.release()
            
        if display:
            cv2.destroyAllWindows()
            
        return all_landmarks
    
    def process_next_frame(self):
        """
        Process the next frame in the video.
        
        Returns:
            frame: The processed frame.
            landmarks: A list of landmark positions.
            ret: True if a frame was read, False otherwise.
        """
        try:
            if not self.is_opened and not self.open():
                return None, [], False
                
            # Read a frame
            ret, frame = self.cap.read()
            if not ret:
                return None, [], False
                
            # Process the frame
            processed_frame, landmarks = self.process_frame(frame)
            
            return processed_frame, landmarks, True
        except Exception as e:
            if hasattr(self, 'error_handler'):
                self.error_handler.handle_video_input_error(e)
            else:
                print(f"Error processing frame: {e}")
            
            # Return a blank frame with the same dimensions as the original
            if hasattr(self, 'width') and hasattr(self, 'height'):
                blank_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                return blank_frame, [], False
            else:
                return None, [], False
    
    def get_video_info(self):
        """
        Get information about the video.
        
        Returns:
            info: A dictionary containing video information.
        """
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'frame_count': self.frame_count
        }
    
    def set_frame_position(self, frame_idx):
        """
        Set the position of the video to a specific frame.
        
        Args:
            frame_idx: The frame index.
            
        Returns:
            success: True if the position was set successfully, False otherwise.
        """
        if not self.is_opened and not self.open():
            return False
            
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    
    def get_frame_position(self):
        """
        Get the current frame position.
        
        Returns:
            frame_idx: The current frame index.
        """
        if not self.is_opened and not self.open():
            return -1
            
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
