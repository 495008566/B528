import numpy as np

class FeedbackGenerator:
    """
    A class for generating feedback based on movement comparison.
    """
    
    def __init__(self):
        """
        Initialize the feedback generator.
        """
        # Define feedback templates for different aspects
        self.feedback_templates = {
            'overall': {
                'excellent': "Your overall performance is excellent! Keep up the great work.",
                'good': "Your overall performance is good. With a bit more practice, you'll achieve excellence.",
                'average': "Your overall performance is satisfactory. Focus on the specific areas mentioned below.",
                'poor': "Your overall performance needs improvement. Don't worry, practice makes perfect!"
            },
            'timing': {
                'excellent': "Your timing is perfect! You're moving at the right pace.",
                'good': "Your timing is good. Try to maintain a consistent rhythm throughout the movement.",
                'average': "Your timing needs some adjustment. Pay attention to the rhythm of the movement.",
                'poor': "Your timing needs significant improvement. Try to follow the rhythm of the reference movement."
            },
            'posture': {
                'excellent': "Your posture is excellent! You're maintaining the correct body alignment.",
                'good': "Your posture is good. Focus on maintaining proper alignment throughout the movement.",
                'average': "Your posture needs some adjustment. Pay attention to your body alignment.",
                'poor': "Your posture needs significant improvement. Focus on maintaining proper body alignment."
            },
            'fluidity': {
                'excellent': "Your movements are very fluid and graceful!",
                'good': "Your movements are quite fluid. Try to make transitions even smoother.",
                'average': "Your movements could be more fluid. Focus on smooth transitions between poses.",
                'poor': "Your movements lack fluidity. Practice making smoother transitions between poses."
            },
            'stability': {
                'excellent': "Your stability is excellent! You're maintaining good balance throughout the movement.",
                'good': "Your stability is good. Focus on maintaining balance during challenging poses.",
                'average': "Your stability needs some improvement. Pay attention to your balance.",
                'poor': "Your stability needs significant improvement. Practice balance exercises to improve."
            }
        }
        
        # Define specific feedback for joint positions
        self.joint_feedback = {
            'left_shoulder': "Pay attention to your left shoulder position.",
            'right_shoulder': "Pay attention to your right shoulder position.",
            'left_elbow': "Your left elbow angle needs adjustment.",
            'right_elbow': "Your right elbow angle needs adjustment.",
            'left_wrist': "Watch the position of your left wrist.",
            'right_wrist': "Watch the position of your right wrist.",
            'left_hip': "Pay attention to your left hip position.",
            'right_hip': "Pay attention to your right hip position.",
            'left_knee': "Your left knee angle needs adjustment.",
            'right_knee': "Your right knee angle needs adjustment.",
            'left_ankle': "Watch the position of your left ankle.",
            'right_ankle': "Watch the position of your right ankle.",
            'neck': "Keep your neck aligned properly.",
            'hips': "Pay attention to your hip alignment."
        }
        
        # Define feedback for common dance issues
        self.common_issues = {
            'arm_extension': "Extend your arms fully during arm movements.",
            'leg_extension': "Extend your legs fully during leg movements.",
            'back_alignment': "Keep your back straight and aligned.",
            'head_position': "Keep your head up and aligned with your spine.",
            'weight_distribution': "Distribute your weight evenly between your feet.",
            'center_of_gravity': "Maintain your center of gravity during balance poses.",
            'rhythm': "Follow the rhythm of the movement more closely.",
            'expression': "Add more expression to your movements.",
            'energy': "Put more energy into your movements.",
            'precision': "Focus on the precision of your movements."
        }
    
    def generate_feedback(self, scores, reference_features, learner_features, path=None):
        """
        Generate feedback based on scores and feature comparison.
        
        Args:
            scores: A dictionary of scores for different aspects of the movement.
            reference_features: The reference movement features.
            learner_features: The learner movement features.
            path: The optimal warping path from DTW comparison.
            
        Returns:
            feedback: A dictionary of feedback for different aspects of the movement.
        """
        feedback = {}
        
        # Generate overall feedback
        overall_score = scores.get('overall', 0)
        if overall_score >= 90:
            feedback['overall'] = self.feedback_templates['overall']['excellent']
        elif overall_score >= 70:
            feedback['overall'] = self.feedback_templates['overall']['good']
        elif overall_score >= 50:
            feedback['overall'] = self.feedback_templates['overall']['average']
        else:
            feedback['overall'] = self.feedback_templates['overall']['poor']
        
        # Generate feedback for each aspect
        for aspect in ['timing', 'posture', 'fluidity', 'stability']:
            score = scores.get(aspect, 0)
            if score >= 90:
                feedback[aspect] = self.feedback_templates[aspect]['excellent']
            elif score >= 70:
                feedback[aspect] = self.feedback_templates[aspect]['good']
            elif score >= 50:
                feedback[aspect] = self.feedback_templates[aspect]['average']
            else:
                feedback[aspect] = self.feedback_templates[aspect]['poor']
        
        # Generate specific feedback based on joint positions
        joint_feedback = self._generate_joint_feedback(reference_features, learner_features)
        if joint_feedback:
            feedback['joints'] = joint_feedback
        
        # Generate feedback for common issues
        common_issues_feedback = self._generate_common_issues_feedback(reference_features, learner_features, path)
        if common_issues_feedback:
            feedback['common_issues'] = common_issues_feedback
        
        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(scores, reference_features, learner_features)
        if improvement_suggestions:
            feedback['improvement_suggestions'] = improvement_suggestions
        
        return feedback
    
    def _generate_joint_feedback(self, reference_features, learner_features):
        """
        Generate feedback based on joint positions.
        
        Args:
            reference_features: The reference movement features.
            learner_features: The learner movement features.
            
        Returns:
            joint_feedback: A list of feedback for joint positions.
        """
        joint_feedback = []
        
        # Check if joint_positions are available in both features
        if ('joint_positions' not in reference_features or 
            'joint_positions' not in learner_features):
            return joint_feedback
        
        ref_joint_positions = reference_features['joint_positions']
        learner_joint_positions = learner_features['joint_positions']
        
        # Compare joint positions
        for joint_name, ref_positions in ref_joint_positions.items():
            if joint_name not in learner_joint_positions:
                continue
                
            learner_positions = learner_joint_positions[joint_name]
            
            # Calculate the average distance between reference and learner positions
            distances = []
            for i in range(min(len(ref_positions), len(learner_positions))):
                ref_pos = ref_positions[i]
                learner_pos = learner_positions[i]
                
                # Calculate Euclidean distance
                distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(ref_pos, learner_pos)))
                distances.append(distance)
            
            # If the average distance is large, add feedback for this joint
            if distances and np.mean(distances) > 20:  # Threshold can be adjusted
                if joint_name in self.joint_feedback:
                    joint_feedback.append(self.joint_feedback[joint_name])
        
        return joint_feedback
    
    def _generate_common_issues_feedback(self, reference_features, learner_features, path):
        """
        Generate feedback for common dance issues.
        
        Args:
            reference_features: The reference movement features.
            learner_features: The learner movement features.
            path: The optimal warping path from DTW comparison.
            
        Returns:
            common_issues_feedback: A list of feedback for common issues.
        """
        common_issues_feedback = []
        
        # Check if joint_angles are available in both features
        if ('joint_angles' not in reference_features or 
            'joint_angles' not in learner_features):
            return common_issues_feedback
        
        ref_joint_angles = reference_features['joint_angles']
        learner_joint_angles = learner_features['joint_angles']
        
        # Check for arm extension issues
        arm_connections = [
            ('left_shoulder', 'left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow', 'right_wrist')
        ]
        
        arm_extension_issues = False
        for connection in arm_connections:
            if connection not in ref_joint_angles or connection not in learner_joint_angles:
                continue
                
            ref_angles = ref_joint_angles[connection]
            learner_angles = learner_joint_angles[connection]
            
            # Calculate the average difference in angles
            angle_diffs = []
            for i in range(min(len(ref_angles), len(learner_angles))):
                angle_diff = abs(ref_angles[i] - learner_angles[i])
                angle_diffs.append(angle_diff)
            
            # If the average angle difference is large, there might be arm extension issues
            if angle_diffs and np.mean(angle_diffs) > 20:  # Threshold can be adjusted
                arm_extension_issues = True
                break
        
        if arm_extension_issues:
            common_issues_feedback.append(self.common_issues['arm_extension'])
        
        # Check for leg extension issues
        leg_connections = [
            ('left_hip', 'left_knee', 'left_ankle'),
            ('right_hip', 'right_knee', 'right_ankle')
        ]
        
        leg_extension_issues = False
        for connection in leg_connections:
            if connection not in ref_joint_angles or connection not in learner_joint_angles:
                continue
                
            ref_angles = ref_joint_angles[connection]
            learner_angles = learner_joint_angles[connection]
            
            # Calculate the average difference in angles
            angle_diffs = []
            for i in range(min(len(ref_angles), len(learner_angles))):
                angle_diff = abs(ref_angles[i] - learner_angles[i])
                angle_diffs.append(angle_diff)
            
            # If the average angle difference is large, there might be leg extension issues
            if angle_diffs and np.mean(angle_diffs) > 20:  # Threshold can be adjusted
                leg_extension_issues = True
                break
        
        if leg_extension_issues:
            common_issues_feedback.append(self.common_issues['leg_extension'])
        
        # Check for rhythm issues using the warping path
        if path is not None:
            path_array = np.array(path)
            reference_indices = path_array[:, 0]
            learner_indices = path_array[:, 1]
            
            # Calculate the differences between consecutive indices
            reference_diffs = np.diff(reference_indices)
            learner_diffs = np.diff(learner_indices)
            
            # If the standard deviation of the differences is large, there might be rhythm issues
            if (np.std(reference_diffs) > 1.5 or np.std(learner_diffs) > 1.5):  # Threshold can be adjusted
                common_issues_feedback.append(self.common_issues['rhythm'])
        
        return common_issues_feedback
    
    def _generate_improvement_suggestions(self, scores, reference_features, learner_features):
        """
        Generate improvement suggestions based on scores and feature comparison.
        
        Args:
            scores: A dictionary of scores for different aspects of the movement.
            reference_features: The reference movement features.
            learner_features: The learner movement features.
            
        Returns:
            improvement_suggestions: A list of improvement suggestions.
        """
        improvement_suggestions = []
        
        # Identify the lowest scoring aspect
        lowest_aspect = min(scores.items(), key=lambda x: x[1] if x[0] != 'overall' else float('inf'))[0]
        lowest_score = scores[lowest_aspect]
        
        # Generate a suggestion for the lowest scoring aspect
        if lowest_score < 70:
            if lowest_aspect == 'timing':
                improvement_suggestions.append("Practice with a metronome to improve your timing.")
                improvement_suggestions.append("Watch the reference movement carefully and try to match its rhythm.")
            elif lowest_aspect == 'posture':
                improvement_suggestions.append("Practice in front of a mirror to check your posture.")
                improvement_suggestions.append("Focus on maintaining proper body alignment throughout the movement.")
            elif lowest_aspect == 'fluidity':
                improvement_suggestions.append("Practice slow, controlled movements to improve fluidity.")
                improvement_suggestions.append("Focus on smooth transitions between poses.")
            elif lowest_aspect == 'stability':
                improvement_suggestions.append("Practice balance exercises to improve stability.")
                improvement_suggestions.append("Focus on maintaining your center of gravity during challenging poses.")
        
        # Add a general improvement suggestion
        improvement_suggestions.append("Regular practice is key to improvement. Keep practicing!")
        
        return improvement_suggestions
    
    def format_feedback(self, feedback):
        """
        Format feedback for display.
        
        Args:
            feedback: A dictionary of feedback for different aspects of the movement.
            
        Returns:
            formatted_feedback: A formatted string of feedback.
        """
        formatted_feedback = []
        
        # Add overall feedback
        if 'overall' in feedback:
            formatted_feedback.append(f"Overall: {feedback['overall']}")
        
        # Add feedback for each aspect
        for aspect in ['timing', 'posture', 'fluidity', 'stability']:
            if aspect in feedback:
                formatted_feedback.append(f"{aspect.capitalize()}: {feedback[aspect]}")
        
        # Add joint feedback
        if 'joints' in feedback and feedback['joints']:
            formatted_feedback.append("\nJoint-specific feedback:")
            for joint_feedback in feedback['joints']:
                formatted_feedback.append(f"- {joint_feedback}")
        
        # Add common issues feedback
        if 'common_issues' in feedback and feedback['common_issues']:
            formatted_feedback.append("\nCommon issues:")
            for issue_feedback in feedback['common_issues']:
                formatted_feedback.append(f"- {issue_feedback}")
        
        # Add improvement suggestions
        if 'improvement_suggestions' in feedback and feedback['improvement_suggestions']:
            formatted_feedback.append("\nImprovement suggestions:")
            for suggestion in feedback['improvement_suggestions']:
                formatted_feedback.append(f"- {suggestion}")
        
        return "\n".join(formatted_feedback)
