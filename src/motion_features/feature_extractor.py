import numpy as np
import math
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter

class FeatureExtractor:
    """
    A class for extracting motion features from skeleton data.
    """
    
    def __init__(self):
        """
        Initialize the feature extractor.
        """
        # Define key joints for feature extraction
        self.key_joints = [
            'nose', 'neck', 'left_shoulder', 'right_shoulder', 
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
            'hips', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle'
        ]
        
        # Define joint connections for angle calculations
        self.joint_connections = [
            ('left_shoulder', 'left_elbow', 'left_wrist'),  # Left arm angle
            ('right_shoulder', 'right_elbow', 'right_wrist'),  # Right arm angle
            ('left_hip', 'left_knee', 'left_ankle'),  # Left leg angle
            ('right_hip', 'right_knee', 'right_ankle'),  # Right leg angle
            ('neck', 'left_shoulder', 'left_elbow'),  # Left shoulder angle
            ('neck', 'right_shoulder', 'right_elbow'),  # Right shoulder angle
            ('hips', 'left_hip', 'left_knee'),  # Left hip angle
            ('hips', 'right_hip', 'right_knee'),  # Right hip angle
            ('left_shoulder', 'neck', 'right_shoulder'),  # Neck angle
            ('left_hip', 'hips', 'right_hip')  # Hip angle
        ]
    
    def extract_features(self, skeleton_sequence, smooth=True, window_length=5, polyorder=2):
        """
        Extract motion features from a sequence of skeleton data.
        
        Args:
            skeleton_sequence: A list of skeleton data for each frame.
            smooth: Whether to smooth the features using Savitzky-Golay filter.
            window_length: Window length for the Savitzky-Golay filter.
            polyorder: Polynomial order for the Savitzky-Golay filter.
            
        Returns:
            features: A dictionary of extracted features.
        """
        # Initialize features
        features = {
            'joint_positions': [],
            'joint_velocities': [],
            'joint_accelerations': [],
            'joint_angles': [],
            'joint_angle_velocities': [],
            'center_of_mass': [],
            'center_of_mass_velocity': [],
            'center_of_mass_acceleration': [],
            'pose_height': [],
            'pose_width': [],
            'pose_area': []
        }
        
        # Extract raw joint positions
        joint_positions = self._extract_joint_positions(skeleton_sequence)
        
        # Smooth joint positions if requested
        if smooth:
            joint_positions = self._smooth_joint_positions(joint_positions, window_length, polyorder)
        
        # Extract joint velocities and accelerations
        joint_velocities = self._extract_joint_velocities(joint_positions)
        joint_accelerations = self._extract_joint_accelerations(joint_velocities)
        
        # Extract joint angles
        joint_angles = self._extract_joint_angles(joint_positions)
        
        # Extract joint angle velocities
        joint_angle_velocities = self._extract_joint_angle_velocities(joint_angles)
        
        # Extract center of mass
        center_of_mass = self._extract_center_of_mass(joint_positions)
        
        # Extract center of mass velocity and acceleration
        center_of_mass_velocity = self._extract_center_of_mass_velocity(center_of_mass)
        center_of_mass_acceleration = self._extract_center_of_mass_acceleration(center_of_mass_velocity)
        
        # Extract pose dimensions
        pose_height, pose_width, pose_area = self._extract_pose_dimensions(joint_positions)
        
        # Store all features
        features['joint_positions'] = joint_positions
        features['joint_velocities'] = joint_velocities
        features['joint_accelerations'] = joint_accelerations
        features['joint_angles'] = joint_angles
        features['joint_angle_velocities'] = joint_angle_velocities
        features['center_of_mass'] = center_of_mass
        features['center_of_mass_velocity'] = center_of_mass_velocity
        features['center_of_mass_acceleration'] = center_of_mass_acceleration
        features['pose_height'] = pose_height
        features['pose_width'] = pose_width
        features['pose_area'] = pose_area
        
        return features
    
    def _extract_joint_positions(self, skeleton_sequence):
        """
        Extract joint positions from a sequence of skeleton data.
        
        Args:
            skeleton_sequence: A list of skeleton data for each frame.
            
        Returns:
            joint_positions: A dictionary mapping joint names to their positions over time.
        """
        joint_positions = {joint: [] for joint in self.key_joints}
        
        for skeleton in skeleton_sequence:
            for joint in self.key_joints:
                if joint in skeleton:
                    position = skeleton[joint]['position']
                    joint_positions[joint].append(position)
                else:
                    # If joint is missing, use the previous position or [0, 0, 0]
                    prev_position = joint_positions[joint][-1] if joint_positions[joint] else [0, 0, 0]
                    joint_positions[joint].append(prev_position)
        
        return joint_positions
    
    def _smooth_joint_positions(self, joint_positions, window_length=5, polyorder=2):
        """
        Smooth joint positions using Savitzky-Golay filter.
        
        Args:
            joint_positions: A dictionary mapping joint names to their positions over time.
            window_length: Window length for the Savitzky-Golay filter.
            polyorder: Polynomial order for the Savitzky-Golay filter.
            
        Returns:
            smoothed_positions: A dictionary mapping joint names to their smoothed positions over time.
        """
        smoothed_positions = {}
        
        for joint, positions in joint_positions.items():
            # Convert positions to numpy array
            positions_array = np.array(positions)
            
            # Check if we have enough data points for smoothing
            if len(positions_array) > window_length:
                # Smooth each dimension separately
                smoothed_x = savgol_filter(positions_array[:, 0], window_length, polyorder)
                smoothed_y = savgol_filter(positions_array[:, 1], window_length, polyorder)
                
                # Handle z dimension if available
                if positions_array.shape[1] > 2:
                    smoothed_z = savgol_filter(positions_array[:, 2], window_length, polyorder)
                    smoothed_array = np.column_stack((smoothed_x, smoothed_y, smoothed_z))
                else:
                    smoothed_array = np.column_stack((smoothed_x, smoothed_y))
                
                smoothed_positions[joint] = smoothed_array.tolist()
            else:
                # Not enough data points for smoothing, use original positions
                smoothed_positions[joint] = positions
        
        return smoothed_positions
    
    def _extract_joint_velocities(self, joint_positions):
        """
        Extract joint velocities from joint positions.
        
        Args:
            joint_positions: A dictionary mapping joint names to their positions over time.
            
        Returns:
            joint_velocities: A dictionary mapping joint names to their velocities over time.
        """
        joint_velocities = {}
        
        for joint, positions in joint_positions.items():
            velocities = []
            
            for i in range(1, len(positions)):
                # Calculate velocity as the difference between consecutive positions
                velocity = [
                    positions[i][0] - positions[i-1][0],
                    positions[i][1] - positions[i-1][1]
                ]
                
                # Handle z dimension if available
                if len(positions[i]) > 2:
                    velocity.append(positions[i][2] - positions[i-1][2])
                
                velocities.append(velocity)
            
            # Add a zero velocity for the first frame
            velocities.insert(0, [0] * len(positions[0]))
            
            joint_velocities[joint] = velocities
        
        return joint_velocities
    
    def _extract_joint_accelerations(self, joint_velocities):
        """
        Extract joint accelerations from joint velocities.
        
        Args:
            joint_velocities: A dictionary mapping joint names to their velocities over time.
            
        Returns:
            joint_accelerations: A dictionary mapping joint names to their accelerations over time.
        """
        joint_accelerations = {}
        
        for joint, velocities in joint_velocities.items():
            accelerations = []
            
            for i in range(1, len(velocities)):
                # Calculate acceleration as the difference between consecutive velocities
                acceleration = [
                    velocities[i][0] - velocities[i-1][0],
                    velocities[i][1] - velocities[i-1][1]
                ]
                
                # Handle z dimension if available
                if len(velocities[i]) > 2:
                    acceleration.append(velocities[i][2] - velocities[i-1][2])
                
                accelerations.append(acceleration)
            
            # Add a zero acceleration for the first frame
            accelerations.insert(0, [0] * len(velocities[0]))
            
            joint_accelerations[joint] = accelerations
        
        return joint_accelerations
    
    def _extract_joint_angles(self, joint_positions):
        """
        Extract joint angles from joint positions.
        
        Args:
            joint_positions: A dictionary mapping joint names to their positions over time.
            
        Returns:
            joint_angles: A dictionary mapping joint connections to their angles over time.
        """
        joint_angles = {}
        
        for connection in self.joint_connections:
            joint1, joint2, joint3 = connection
            angles = []
            
            for i in range(len(joint_positions[joint1])):
                # Get positions of the three joints
                pos1 = joint_positions[joint1][i]
                pos2 = joint_positions[joint2][i]
                pos3 = joint_positions[joint3][i]
                
                # Calculate vectors
                vector1 = [pos1[0] - pos2[0], pos1[1] - pos2[1]]
                vector2 = [pos3[0] - pos2[0], pos3[1] - pos2[1]]
                
                # Handle z dimension if available
                if len(pos1) > 2 and len(pos3) > 2:
                    vector1.append(pos1[2] - pos2[2])
                    vector2.append(pos3[2] - pos2[2])
                
                # Calculate angle using dot product
                angle = self._calculate_angle(vector1, vector2)
                angles.append(angle)
            
            joint_angles[connection] = angles
        
        return joint_angles
    
    def _extract_joint_angle_velocities(self, joint_angles):
        """
        Extract joint angle velocities from joint angles.
        
        Args:
            joint_angles: A dictionary mapping joint connections to their angles over time.
            
        Returns:
            joint_angle_velocities: A dictionary mapping joint connections to their angle velocities over time.
        """
        joint_angle_velocities = {}
        
        for connection, angles in joint_angles.items():
            velocities = []
            
            for i in range(1, len(angles)):
                # Calculate velocity as the difference between consecutive angles
                velocity = angles[i] - angles[i-1]
                velocities.append(velocity)
            
            # Add a zero velocity for the first frame
            velocities.insert(0, 0)
            
            joint_angle_velocities[connection] = velocities
        
        return joint_angle_velocities
    
    def _extract_center_of_mass(self, joint_positions):
        """
        Extract center of mass from joint positions.
        
        Args:
            joint_positions: A dictionary mapping joint names to their positions over time.
            
        Returns:
            center_of_mass: A list of center of mass positions over time.
        """
        center_of_mass = []
        
        # Get the number of frames
        num_frames = len(next(iter(joint_positions.values())))
        
        for i in range(num_frames):
            # Calculate the average position of all joints
            x_sum = 0
            y_sum = 0
            z_sum = 0
            count = 0
            
            for joint, positions in joint_positions.items():
                if i < len(positions):
                    x_sum += positions[i][0]
                    y_sum += positions[i][1]
                    
                    # Handle z dimension if available
                    if len(positions[i]) > 2:
                        z_sum += positions[i][2]
                    
                    count += 1
            
            # Calculate the average
            if count > 0:
                x_avg = x_sum / count
                y_avg = y_sum / count
                
                # Handle z dimension if available
                if len(next(iter(joint_positions.values()))[0]) > 2:
                    z_avg = z_sum / count
                    center_of_mass.append([x_avg, y_avg, z_avg])
                else:
                    center_of_mass.append([x_avg, y_avg])
            else:
                # No joints available, use [0, 0, 0]
                center_of_mass.append([0, 0, 0])
        
        return center_of_mass
    
    def _extract_center_of_mass_velocity(self, center_of_mass):
        """
        Extract center of mass velocity from center of mass positions.
        
        Args:
            center_of_mass: A list of center of mass positions over time.
            
        Returns:
            center_of_mass_velocity: A list of center of mass velocities over time.
        """
        center_of_mass_velocity = []
        
        for i in range(1, len(center_of_mass)):
            # Calculate velocity as the difference between consecutive positions
            velocity = [
                center_of_mass[i][0] - center_of_mass[i-1][0],
                center_of_mass[i][1] - center_of_mass[i-1][1]
            ]
            
            # Handle z dimension if available
            if len(center_of_mass[i]) > 2:
                velocity.append(center_of_mass[i][2] - center_of_mass[i-1][2])
            
            center_of_mass_velocity.append(velocity)
        
        # Add a zero velocity for the first frame
        center_of_mass_velocity.insert(0, [0] * len(center_of_mass[0]))
        
        return center_of_mass_velocity
    
    def _extract_center_of_mass_acceleration(self, center_of_mass_velocity):
        """
        Extract center of mass acceleration from center of mass velocities.
        
        Args:
            center_of_mass_velocity: A list of center of mass velocities over time.
            
        Returns:
            center_of_mass_acceleration: A list of center of mass accelerations over time.
        """
        center_of_mass_acceleration = []
        
        for i in range(1, len(center_of_mass_velocity)):
            # Calculate acceleration as the difference between consecutive velocities
            acceleration = [
                center_of_mass_velocity[i][0] - center_of_mass_velocity[i-1][0],
                center_of_mass_velocity[i][1] - center_of_mass_velocity[i-1][1]
            ]
            
            # Handle z dimension if available
            if len(center_of_mass_velocity[i]) > 2:
                acceleration.append(center_of_mass_velocity[i][2] - center_of_mass_velocity[i-1][2])
            
            center_of_mass_acceleration.append(acceleration)
        
        # Add a zero acceleration for the first frame
        center_of_mass_acceleration.insert(0, [0] * len(center_of_mass_velocity[0]))
        
        return center_of_mass_acceleration
    
    def _extract_pose_dimensions(self, joint_positions):
        """
        Extract pose dimensions (height, width, area) from joint positions.
        
        Args:
            joint_positions: A dictionary mapping joint names to their positions over time.
            
        Returns:
            pose_height: A list of pose heights over time.
            pose_width: A list of pose widths over time.
            pose_area: A list of pose areas over time.
        """
        pose_height = []
        pose_width = []
        pose_area = []
        
        # Get the number of frames
        num_frames = len(next(iter(joint_positions.values())))
        
        for i in range(num_frames):
            # Initialize min and max values
            min_x = float('inf')
            max_x = float('-inf')
            min_y = float('inf')
            max_y = float('-inf')
            
            # Find the bounding box of the pose
            for joint, positions in joint_positions.items():
                if i < len(positions):
                    min_x = min(min_x, positions[i][0])
                    max_x = max(max_x, positions[i][0])
                    min_y = min(min_y, positions[i][1])
                    max_y = max(max_y, positions[i][1])
            
            # Calculate dimensions
            width = max_x - min_x
            height = max_y - min_y
            area = width * height
            
            pose_width.append(width)
            pose_height.append(height)
            pose_area.append(area)
        
        return pose_height, pose_width, pose_area
    
    def _calculate_angle(self, vector1, vector2):
        """
        Calculate the angle between two vectors.
        
        Args:
            vector1: The first vector.
            vector2: The second vector.
            
        Returns:
            angle: The angle between the vectors in degrees.
        """
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in vector1))
        magnitude2 = math.sqrt(sum(a * a for a in vector2))
        
        # Calculate angle
        if magnitude1 * magnitude2 == 0:
            return 0
        
        cos_angle = dot_product / (magnitude1 * magnitude2)
        
        # Handle floating point errors
        cos_angle = max(-1, min(1, cos_angle))
        
        # Convert to degrees
        angle = math.degrees(math.acos(cos_angle))
        
        return angle
    
    def normalize_features(self, features):
        """
        Normalize features to have zero mean and unit variance.
        
        Args:
            features: A dictionary of extracted features.
            
        Returns:
            normalized_features: A dictionary of normalized features.
        """
        normalized_features = {}
        
        for feature_name, feature_data in features.items():
            if isinstance(feature_data, dict):
                # Handle dictionary features (e.g., joint_positions)
                normalized_dict = {}
                for key, values in feature_data.items():
                    # Convert to numpy array for easier manipulation
                    values_array = np.array(values)
                    
                    # Calculate mean and standard deviation
                    mean = np.mean(values_array, axis=0)
                    std = np.std(values_array, axis=0)
                    
                    # Avoid division by zero
                    std = np.where(std == 0, 1, std)
                    
                    # Normalize
                    normalized_values = (values_array - mean) / std
                    
                    normalized_dict[key] = normalized_values.tolist()
                
                normalized_features[feature_name] = normalized_dict
            else:
                # Handle list features (e.g., center_of_mass)
                values_array = np.array(feature_data)
                
                # Calculate mean and standard deviation
                mean = np.mean(values_array, axis=0)
                std = np.std(values_array, axis=0)
                
                # Avoid division by zero
                std = np.where(std == 0, 1, std)
                
                # Normalize
                normalized_values = (values_array - mean) / std
                
                normalized_features[feature_name] = normalized_values.tolist()
        
        return normalized_features
