import numpy as np
import re

class BVHParser:
    """
    A class for parsing BVH (Biovision Hierarchy) files.
    BVH files contain motion capture data with skeleton hierarchy and motion data.
    """
    
    def __init__(self, file_path):
        """
        Initialize the BVH parser with a file path.
        
        Args:
            file_path: Path to the BVH file.
        """
        self.file_path = file_path
        self.hierarchy = {}
        self.motion_data = []
        self.frame_time = 0
        self.num_frames = 0
        self.joint_channels = {}
        self.joint_offsets = {}
        self.joint_parents = {}
        
    def parse(self):
        """
        Parse the BVH file.
        
        Returns:
            success: True if parsing was successful, False otherwise.
        """
        try:
            with open(self.file_path, 'r') as f:
                content = f.read()
                
            # Split the file into hierarchy and motion sections
            hierarchy_section, motion_section = content.split('MOTION')
            
            # Parse the hierarchy section
            self._parse_hierarchy(hierarchy_section)
            
            # Parse the motion section
            self._parse_motion(motion_section)
            
            return True
        except Exception as e:
            print(f"Error parsing BVH file: {e}")
            return False
    
    def _parse_hierarchy(self, hierarchy_section):
        """
        Parse the hierarchy section of the BVH file.
        
        Args:
            hierarchy_section: The hierarchy section of the BVH file.
        """
        # Extract joint information using regex
        joint_pattern = r'(\w+)\s*{([^{}]*(?:{[^{}]*(?:{[^{}]*}[^{}]*)*}[^{}]*)*)'
        offset_pattern = r'OFFSET\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)'
        channels_pattern = r'CHANNELS\s+(\d+)((?:\s+\w+)+)'
        
        # Find all joints in the hierarchy
        joints = re.findall(joint_pattern, hierarchy_section)
        
        current_parent = None
        parent_stack = []
        
        for joint_name, joint_content in joints:
            # Extract offset
            offset_match = re.search(offset_pattern, joint_content)
            if offset_match:
                offset = [float(offset_match.group(1)), 
                          float(offset_match.group(2)), 
                          float(offset_match.group(3))]
                self.joint_offsets[joint_name] = offset
            
            # Extract channels
            channels_match = re.search(channels_pattern, joint_content)
            if channels_match:
                num_channels = int(channels_match.group(1))
                channel_names = channels_match.group(2).strip().split()
                self.joint_channels[joint_name] = channel_names
            
            # Set parent-child relationship
            if joint_name != 'Hips':  # Hips is the root joint
                self.joint_parents[joint_name] = current_parent
            
            # Check if this joint has children
            if 'JOINT' in joint_content or 'End Site' in joint_content:
                parent_stack.append(current_parent)
                current_parent = joint_name
            
            # Check if this joint is an end site
            if 'End Site' in joint_content:
                end_site_offset_match = re.search(offset_pattern, joint_content)
                if end_site_offset_match:
                    end_site_offset = [float(end_site_offset_match.group(1)), 
                                      float(end_site_offset_match.group(2)), 
                                      float(end_site_offset_match.group(3))]
                    self.joint_offsets[f"{joint_name}_end"] = end_site_offset
                    self.joint_parents[f"{joint_name}_end"] = joint_name
            
            # Check if this joint's definition is ending
            if joint_content.count('{') == joint_content.count('}'):
                if parent_stack:
                    current_parent = parent_stack.pop()
    
    def _parse_motion(self, motion_section):
        """
        Parse the motion section of the BVH file.
        
        Args:
            motion_section: The motion section of the BVH file.
        """
        lines = motion_section.strip().split('\n')
        
        # Extract number of frames
        frames_match = re.search(r'Frames:\s+(\d+)', lines[0])
        if frames_match:
            self.num_frames = int(frames_match.group(1))
        
        # Extract frame time
        frame_time_match = re.search(r'Frame Time:\s+([\d.]+)', lines[1])
        if frame_time_match:
            self.frame_time = float(frame_time_match.group(1))
        
        # Extract motion data
        for i in range(2, len(lines)):
            if lines[i].strip():
                frame_data = [float(x) for x in lines[i].strip().split()]
                self.motion_data.append(frame_data)
    
    def get_joint_positions(self, frame_idx=0):
        """
        Get the positions of all joints for a specific frame.
        
        Args:
            frame_idx: The frame index.
            
        Returns:
            joint_positions: A dictionary mapping joint names to their positions.
        """
        if frame_idx >= self.num_frames:
            print(f"Frame index {frame_idx} out of range (0-{self.num_frames-1})")
            return {}
        
        joint_positions = {}
        frame_data = self.motion_data[frame_idx]
        
        # Start with the root joint (Hips)
        root_channels = self.joint_channels.get('Hips', [])
        root_position = [0, 0, 0]
        root_rotation = [0, 0, 0]
        
        channel_idx = 0
        for i, channel in enumerate(root_channels):
            if channel in ['Xposition', 'Yposition', 'Zposition']:
                pos_idx = {'Xposition': 0, 'Yposition': 1, 'Zposition': 2}[channel]
                root_position[pos_idx] = frame_data[channel_idx]
            elif channel in ['Xrotation', 'Yrotation', 'Zrotation']:
                rot_idx = {'Xrotation': 0, 'Yrotation': 1, 'Zrotation': 2}[channel]
                root_rotation[rot_idx] = frame_data[channel_idx]
            channel_idx += 1
        
        joint_positions['Hips'] = root_position
        
        # Process all other joints
        for joint_name, parent_name in self.joint_parents.items():
            if parent_name is None:
                continue
                
            parent_position = joint_positions.get(parent_name, [0, 0, 0])
            joint_offset = self.joint_offsets.get(joint_name, [0, 0, 0])
            
            # Simple calculation: joint position = parent position + joint offset
            # Note: This is a simplified calculation that doesn't account for rotations
            joint_position = [
                parent_position[0] + joint_offset[0],
                parent_position[1] + joint_offset[1],
                parent_position[2] + joint_offset[2]
            ]
            
            joint_positions[joint_name] = joint_position
        
        return joint_positions
    
    def get_motion_data(self):
        """
        Get the motion data.
        
        Returns:
            motion_data: The motion data.
        """
        return self.motion_data
    
    def get_frame_time(self):
        """
        Get the frame time.
        
        Returns:
            frame_time: The frame time.
        """
        return self.frame_time
    
    def get_num_frames(self):
        """
        Get the number of frames.
        
        Returns:
            num_frames: The number of frames.
        """
        return self.num_frames
    
    def get_joint_channels(self):
        """
        Get the joint channels.
        
        Returns:
            joint_channels: The joint channels.
        """
        return self.joint_channels
    
    def get_joint_offsets(self):
        """
        Get the joint offsets.
        
        Returns:
            joint_offsets: The joint offsets.
        """
        return self.joint_offsets
    
    def get_joint_parents(self):
        """
        Get the joint parents.
        
        Returns:
            joint_parents: The joint parents.
        """
        return self.joint_parents
