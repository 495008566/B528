import numpy as np
import cv2
from .pose_detector import PoseDetector
from .bvh_parser import BVHParser

class SkeletonExtractor:
    """
    A class for extracting skeleton data from videos or BVH files.
    """
    
    def __init__(self):
        """
        Initialize the skeleton extractor.
        """
        self.pose_detector = PoseDetector()
        
    def extract_from_video(self, video_path, max_frames=None):
        """
        Extract skeleton data from a video.
        
        Args:
            video_path: Path to the video file.
            max_frames: Maximum number of frames to process.
            
        Returns:
            skeleton_data: A list of skeleton data for each frame.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return []
            
        skeleton_data = []
        frame_idx = 0
        
        while True:
            # Break if max_frames is reached
            if max_frames is not None and frame_idx >= max_frames:
                break
                
            # Read a frame
            ret, frame = cap.read()
            if not ret:
                break
                
            # Find pose in the frame
            frame, results = self.pose_detector.find_pose(frame, draw=False)
            
            # Find the positions of the landmarks
            landmarks = self.pose_detector.find_position(frame, draw=False)
            
            # Convert landmarks to a more structured format
            frame_skeleton = self._convert_landmarks_to_skeleton(landmarks)
            
            # Store skeleton data
            skeleton_data.append(frame_skeleton)
            
            frame_idx += 1
        
        # Release resources
        cap.release()
        
        return skeleton_data
    
    def extract_from_bvh(self, bvh_path):
        """
        Extract skeleton data from a BVH file.
        
        Args:
            bvh_path: Path to the BVH file.
            
        Returns:
            skeleton_data: A list of skeleton data for each frame.
        """
        bvh_parser = BVHParser(bvh_path)
        if not bvh_parser.parse():
            print(f"Error: Could not parse BVH file {bvh_path}")
            return []
            
        skeleton_data = []
        
        # Get the number of frames
        num_frames = bvh_parser.get_num_frames()
        
        # Extract skeleton data for each frame
        for frame_idx in range(num_frames):
            # Get joint positions for the current frame
            joint_positions = bvh_parser.get_joint_positions(frame_idx)
            
            # Convert joint positions to a more structured format
            frame_skeleton = self._convert_bvh_to_skeleton(joint_positions)
            
            # Store skeleton data
            skeleton_data.append(frame_skeleton)
        
        return skeleton_data
    
    def _convert_landmarks_to_skeleton(self, landmarks):
        """
        Convert MediaPipe landmarks to a structured skeleton format.
        
        Args:
            landmarks: A list of landmark positions from MediaPipe.
            
        Returns:
            skeleton: A dictionary mapping joint names to their positions.
        """
        skeleton = {}
        
        # MediaPipe pose landmarks mapping to joint names
        landmark_to_joint = {
            0: 'nose',
            11: 'left_shoulder',
            12: 'right_shoulder',
            13: 'left_elbow',
            14: 'right_elbow',
            15: 'left_wrist',
            16: 'right_wrist',
            23: 'left_hip',
            24: 'right_hip',
            25: 'left_knee',
            26: 'right_knee',
            27: 'left_ankle',
            28: 'right_ankle'
        }
        
        for landmark in landmarks:
            id, x, y, z, visibility = landmark
            if id in landmark_to_joint:
                joint_name = landmark_to_joint[id]
                skeleton[joint_name] = {
                    'position': [x, y, z],
                    'visibility': visibility
                }
        
        return skeleton
    
    def _convert_bvh_to_skeleton(self, joint_positions):
        """
        Convert BVH joint positions to a structured skeleton format.
        
        Args:
            joint_positions: A dictionary mapping joint names to their positions from BVH.
            
        Returns:
            skeleton: A dictionary mapping joint names to their positions.
        """
        skeleton = {}
        
        # BVH joint names mapping to standardized joint names
        bvh_to_joint = {
            'Hips': 'hips',
            'Chest': 'spine',
            'Chest2': 'chest',
            'Chest3': 'upper_chest',
            'Neck': 'neck',
            'Head': 'head',
            'LeftShoulder': 'left_shoulder',
            'LeftArm': 'left_upper_arm',
            'LeftForeArm': 'left_lower_arm',
            'LeftHand': 'left_hand',
            'RightShoulder': 'right_shoulder',
            'RightArm': 'right_upper_arm',
            'RightForeArm': 'right_lower_arm',
            'RightHand': 'right_hand',
            'LeftUpLeg': 'left_upper_leg',
            'LeftLeg': 'left_lower_leg',
            'LeftFoot': 'left_foot',
            'RightUpLeg': 'right_upper_leg',
            'RightLeg': 'right_lower_leg',
            'RightFoot': 'right_foot'
        }
        
        for bvh_joint, position in joint_positions.items():
            if bvh_joint in bvh_to_joint:
                joint_name = bvh_to_joint[bvh_joint]
                skeleton[joint_name] = {
                    'position': position,
                    'visibility': 1.0  # BVH data is always visible
                }
        
        return skeleton
    
    def visualize_skeleton(self, skeleton, img_size=(640, 480)):
        """
        Visualize a skeleton on an image.
        
        Args:
            skeleton: A dictionary mapping joint names to their positions.
            img_size: The size of the output image.
            
        Returns:
            img: The image with the skeleton drawn.
        """
        img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
        
        # Define connections between joints for drawing
        connections = [
            ('nose', 'neck'),
            ('neck', 'left_shoulder'),
            ('neck', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('right_shoulder', 'right_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_elbow', 'right_wrist'),
            ('neck', 'hips'),
            ('hips', 'left_hip'),
            ('hips', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('right_hip', 'right_knee'),
            ('left_knee', 'left_ankle'),
            ('right_knee', 'right_ankle')
        ]
        
        # Draw joints
        for joint_name, joint_data in skeleton.items():
            position = joint_data['position']
            visibility = joint_data['visibility']
            
            # Skip joints with low visibility
            if visibility < 0.5:
                continue
                
            # Convert position to pixel coordinates
            x, y = int(position[0]), int(position[1])
            
            # Draw a circle at the joint position
            cv2.circle(img, (x, y), 5, (0, 255, 0), cv2.FILLED)
            
            # Add joint name
            cv2.putText(img, joint_name, (x + 10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw connections
        for joint1_name, joint2_name in connections:
            if joint1_name in skeleton and joint2_name in skeleton:
                joint1_data = skeleton[joint1_name]
                joint2_data = skeleton[joint2_name]
                
                # Skip connections with low visibility
                if joint1_data['visibility'] < 0.5 or joint2_data['visibility'] < 0.5:
                    continue
                    
                # Convert positions to pixel coordinates
                x1, y1 = int(joint1_data['position'][0]), int(joint1_data['position'][1])
                x2, y2 = int(joint2_data['position'][0]), int(joint2_data['position'][1])
                
                # Draw a line between the joints
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        return img
