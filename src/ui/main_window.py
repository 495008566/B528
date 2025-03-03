import sys
import cv2
import numpy as np
import time
import traceback
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QComboBox, 
                            QProgressBar, QTextEdit, QSplitter, QFrame,
                            QCheckBox, QSlider, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from ..utils.performance_optimizer import PerformanceOptimizer, FrameSkipper
from ..utils.error_handler import ErrorHandler, setup_global_error_handler

from ..skeleton_detection.video_processor import VideoProcessor
from ..skeleton_detection.skeleton_extractor import SkeletonExtractor
from ..motion_features.feature_extractor import FeatureExtractor
from ..dtw_comparison.movement_matcher import MovementMatcher
from ..scoring.real_time_evaluator import RealTimeEvaluator

class VideoThread(QThread):
    """
    A thread for processing video frames.
    """
    frame_ready = pyqtSignal(np.ndarray, list)
    fps_updated = pyqtSignal(float)
    
    def __init__(self, video_source=0):
        """
        Initialize the video thread.
        
        Args:
            video_source: The video source (camera index or video file path).
        """
        super().__init__()
        self.video_source = video_source
        self.running = False
        self.video_processor = None
        self.performance_optimizer = PerformanceOptimizer()
        self.frame_skipper = FrameSkipper(target_fps=30, max_skip=2)
        self.enable_frame_skipping = True
        self.enable_performance_monitoring = True
        self.downscale_factor = 1  # No downscaling by default
        
    def run(self):
        """
        Run the video thread.
        """
        self.running = True
        self.video_processor = VideoProcessor(
            video_path=self.video_source if isinstance(self.video_source, str) else None,
            camera_id=self.video_source if isinstance(self.video_source, int) else 0,
            use_camera=isinstance(self.video_source, int)
        )
        
        if not self.video_processor.open():
            self.running = False
            return
        
        fps_update_interval = 1.0  # Update FPS every second
        last_fps_update = time.time()
        
        while self.running:
            # Start frame processing timer
            if self.enable_performance_monitoring:
                self.performance_optimizer.start_timer("frame_processing")
            
            # Check if we should process this frame
            if self.enable_frame_skipping and not self.frame_skipper.should_process_frame():
                # Sleep a bit to avoid busy waiting
                time.sleep(0.001)
                continue
            
            # Process the frame
            if self.enable_performance_monitoring:
                self.performance_optimizer.start_timer("frame_capture")
            
            frame, landmarks, ret = self.video_processor.process_next_frame()
            
            if self.enable_performance_monitoring:
                self.performance_optimizer.stop_timer("frame_capture")
            
            if not ret:
                # End of video or error
                self.running = False
                break
            
            # Downscale the frame if needed
            if self.downscale_factor > 1:
                if self.enable_performance_monitoring:
                    self.performance_optimizer.start_timer("frame_downscale")
                
                height, width = frame.shape[:2]
                new_height = height // self.downscale_factor
                new_width = width // self.downscale_factor
                frame = cv2.resize(frame, (new_width, new_height))
                
                if self.enable_performance_monitoring:
                    self.performance_optimizer.stop_timer("frame_downscale")
            
            # Emit the frame
            self.frame_ready.emit(frame, landmarks)
            
            # Stop frame processing timer
            if self.enable_performance_monitoring:
                frame_time = self.performance_optimizer.stop_timer("frame_processing")
                self.performance_optimizer.record_frame_time(frame_time)
                
                # Update FPS periodically
                current_time = time.time()
                if current_time - last_fps_update >= fps_update_interval:
                    fps = self.performance_optimizer.get_fps()
                    self.fps_updated.emit(fps)
                    last_fps_update = current_time
            
            # Sleep to control frame rate
            time.sleep(0.001)
        
        self.video_processor.close()
    
    def stop(self):
        """
        Stop the video thread.
        """
        self.running = False
        self.wait()
    
    def set_video_source(self, video_source):
        """
        Set the video source.
        
        Args:
            video_source: The video source (camera index or video file path).
        """
        self.video_source = video_source
        
        # Restart the thread if it's running
        if self.running:
            self.stop()
            self.start()
    
    def set_frame_skipping(self, enable):
        """
        Enable or disable frame skipping.
        
        Args:
            enable: Whether to enable frame skipping.
        """
        self.enable_frame_skipping = enable
    
    def set_performance_monitoring(self, enable):
        """
        Enable or disable performance monitoring.
        
        Args:
            enable: Whether to enable performance monitoring.
        """
        self.enable_performance_monitoring = enable
    
    def set_downscale_factor(self, factor):
        """
        Set the downscale factor for frames.
        
        Args:
            factor: The downscale factor (1 = no downscaling, 2 = half size, etc.).
        """
        self.downscale_factor = max(1, factor)
    
    def get_performance_stats(self):
        """
        Get performance statistics.
        
        Returns:
            stats: A dictionary of performance statistics.
        """
        if not self.enable_performance_monitoring:
            return {}
            
        return self.performance_optimizer.get_all_stats()

class EvaluationThread(QThread):
    """
    A thread for evaluating dance movements.
    """
    evaluation_ready = pyqtSignal(dict)
    
    def __init__(self, reference_features=None):
        """
        Initialize the evaluation thread.
        
        Args:
            reference_features: The reference movement features.
        """
        super().__init__()
        self.running = False
        self.feature_extractor = FeatureExtractor()
        self.evaluator = RealTimeEvaluator(reference_features)
        self.landmarks_buffer = []
        self.buffer_size = 30
        self.evaluation_interval = 15  # Frames between evaluations
        self.frame_count = 0
        
    def run(self):
        """
        Run the evaluation thread.
        """
        self.running = True
        
        while self.running:
            # Sleep to avoid busy waiting
            time.sleep(0.01)
            
            # Skip if buffer is not full
            if len(self.landmarks_buffer) < self.buffer_size:
                continue
            
            # Increment frame count
            self.frame_count += 1
            
            # Skip if it's not time to evaluate
            if self.frame_count % self.evaluation_interval != 0:
                continue
            
            # Extract features from landmarks
            features = self._extract_features_from_landmarks()
            
            # Add features to evaluator
            self.evaluator.add_frame(features)
            
            # Evaluate
            evaluation_result, _ = self.evaluator.evaluate()
            
            # Emit the evaluation result
            self.evaluation_ready.emit(evaluation_result)
    
    def stop(self):
        """
        Stop the evaluation thread.
        """
        self.running = False
        self.wait()
    
    def add_landmarks(self, landmarks):
        """
        Add landmarks to the buffer.
        
        Args:
            landmarks: The landmarks to add.
        """
        self.landmarks_buffer.append(landmarks)
        
        # Keep only the last buffer_size landmarks
        if len(self.landmarks_buffer) > self.buffer_size:
            self.landmarks_buffer = self.landmarks_buffer[-self.buffer_size:]
    
    def set_reference_features(self, reference_features):
        """
        Set the reference features.
        
        Args:
            reference_features: The reference movement features.
        """
        self.evaluator.set_reference_features(reference_features)
    
    def _extract_features_from_landmarks(self):
        """
        Extract features from landmarks.
        
        Returns:
            features: The extracted features.
        """
        # Convert landmarks to skeleton format
        skeleton_data = []
        for landmarks in self.landmarks_buffer:
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
            
            skeleton_data.append(skeleton)
        
        # Extract features from skeleton data
        features = self.feature_extractor.extract_features(skeleton_data)
        
        return features

class MainWindow(QMainWindow):
    """
    The main window of the application.
    """
    
    def __init__(self):
        """
        Initialize the main window.
        """
        super().__init__()
        
        # Set up error handling
        self.error_handler = setup_global_error_handler()
        
        try:
            # Set window properties
            self.setWindowTitle("Dunhuang Dance Evaluation System")
            self.setGeometry(100, 100, 1200, 800)
            
            # Initialize variables
            self.video_thread = None
            self.evaluation_thread = None
            self.reference_features = None
            self.current_movement = None
            self.reference_movements = {}
            self.skeleton_extractor = SkeletonExtractor()
            self.feature_extractor = FeatureExtractor()
            self.movement_matcher = MovementMatcher()
            
            # Create the UI
            self._create_ui()
            
            # Start the video thread
            self._start_video_thread()
            
            # Start the evaluation thread
            self._start_evaluation_thread()
            
            # Set up error status timer
            self.error_status_timer = QTimer(self)
            self.error_status_timer.timeout.connect(self._update_error_status)
            self.error_status_timer.start(5000)  # Update every 5 seconds
        except Exception as e:
            self.error_handler.handle_ui_error(e, critical=True)
            QMessageBox.critical(self, "Initialization Error", 
                                f"An error occurred during initialization: {str(e)}")
            raise
    
    def _create_ui(self):
        """
        Create the user interface.
        """
        # Create the central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create the main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create the left panel (video and controls)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Create the video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")
        left_layout.addWidget(self.video_label)
        
        # Create the controls
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        
        # Create the video source selection
        self.source_combo = QComboBox()
        self.source_combo.addItem("Camera", 0)
        self.source_combo.addItem("Video File", "")
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        controls_layout.addWidget(QLabel("Video Source:"))
        controls_layout.addWidget(self.source_combo)
        
        # Create the movement selection
        self.movement_combo = QComboBox()
        self.movement_combo.addItem("Select Movement", "")
        self._populate_movement_combo()
        self.movement_combo.currentIndexChanged.connect(self._on_movement_changed)
        controls_layout.addWidget(QLabel("Reference Movement:"))
        controls_layout.addWidget(self.movement_combo)
        
        # Create the start/stop button
        self.start_stop_button = QPushButton("Stop")
        self.start_stop_button.clicked.connect(self._on_start_stop_clicked)
        controls_layout.addWidget(self.start_stop_button)
        
        left_layout.addWidget(controls_widget)
        
        # Create the right panel (scores and feedback)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Create the scores section
        scores_group = QFrame()
        scores_group.setFrameShape(QFrame.StyledPanel)
        scores_layout = QVBoxLayout(scores_group)
        
        # Create the overall score
        overall_layout = QHBoxLayout()
        overall_layout.addWidget(QLabel("Overall Score:"))
        self.overall_score_label = QLabel("0")
        self.overall_score_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.overall_score_label.setAlignment(Qt.AlignCenter)
        overall_layout.addWidget(self.overall_score_label)
        scores_layout.addLayout(overall_layout)
        
        # Create the detailed scores
        self.score_bars = {}
        for aspect in ["Timing", "Posture", "Fluidity", "Stability"]:
            aspect_layout = QHBoxLayout()
            aspect_layout.addWidget(QLabel(f"{aspect}:"))
            
            score_bar = QProgressBar()
            score_bar.setRange(0, 100)
            score_bar.setValue(0)
            score_bar.setTextVisible(True)
            score_bar.setFormat("%v")
            aspect_layout.addWidget(score_bar)
            
            scores_layout.addLayout(aspect_layout)
            self.score_bars[aspect.lower()] = score_bar
        
        right_layout.addWidget(scores_group)
        
        # Create the feedback section
        feedback_group = QFrame()
        feedback_group.setFrameShape(QFrame.StyledPanel)
        feedback_layout = QVBoxLayout(feedback_group)
        
        feedback_layout.addWidget(QLabel("Feedback:"))
        
        self.feedback_text = QTextEdit()
        self.feedback_text.setReadOnly(True)
        feedback_layout.addWidget(self.feedback_text)
        
        right_layout.addWidget(feedback_group)
        
        # Add the panels to the main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 600])
        main_layout.addWidget(splitter)
    
    def _start_video_thread(self):
        """
        Start the video thread.
        """
        if self.video_thread is not None:
            self.video_thread.stop()
        
        video_source = self.source_combo.currentData()
        self.video_thread = VideoThread(video_source)
        self.video_thread.frame_ready.connect(self._on_frame_ready)
        self.video_thread.start()
    
    def _start_evaluation_thread(self):
        """
        Start the evaluation thread.
        """
        if self.evaluation_thread is not None:
            self.evaluation_thread.stop()
        
        self.evaluation_thread = EvaluationThread(self.reference_features)
        self.evaluation_thread.evaluation_ready.connect(self._on_evaluation_ready)
        self.evaluation_thread.start()
    
    def _populate_movement_combo(self):
        """
        Populate the movement selection combo box.
        """
        # Clear the combo box
        while self.movement_combo.count() > 1:
            self.movement_combo.removeItem(1)
        
        # Add movements from the dataset
        categories = [
            "1-FeiTian", "2-PuSa", "3-LianHuaTongZi", 
            "4-LiShiWuJi", "5-JiGuJiYue", "6-PiPaJiYue", "7-QiTaWuZi"
        ]
        
        for category in categories:
            self.movement_combo.addItem(category, category)
    
    def _on_source_changed(self, index):
        """
        Handle the video source change event.
        
        Args:
            index: The index of the selected item.
        """
        if index == 1:  # Video File
            # TODO: Open a file dialog to select a video file
            # For now, use a default video file
            self.source_combo.setItemData(1, "~/dunhuang_dance_evaluation/sample.mp4")
        
        # Restart the video thread
        self._start_video_thread()
    
    def _on_movement_changed(self, index):
        """
        Handle the movement change event.
        
        Args:
            index: The index of the selected item.
        """
        movement = self.movement_combo.currentData()
        if not movement:
            return
        
        self.current_movement = movement
        
        # Load the reference features for the selected movement
        if movement in self.reference_movements:
            self.reference_features = self.reference_movements[movement]
        else:
            # TODO: Load the reference features from the dataset
            # For now, use a placeholder
            self.reference_features = None
        
        # Update the evaluation thread
        if self.evaluation_thread is not None:
            self.evaluation_thread.set_reference_features(self.reference_features)
    
    def _on_start_stop_clicked(self):
        """
        Handle the start/stop button click event.
        """
        if self.video_thread is not None and self.video_thread.running:
            self.video_thread.stop()
            self.start_stop_button.setText("Start")
        else:
            self._start_video_thread()
            self.start_stop_button.setText("Stop")
    
    def _on_frame_ready(self, frame, landmarks):
        """
        Handle the frame ready event.
        
        Args:
            frame: The processed frame.
            landmarks: The detected landmarks.
        """
        try:
            # Check if frame is valid
            if frame is None or frame.size == 0:
                self.error_handler.handle_ui_error(ValueError("Invalid frame received"), critical=False)
                return
            
            # Convert the frame to QImage
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            
            # Display the frame
            self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(
                self.video_label.width(), self.video_label.height(), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            # Add landmarks to the evaluation thread
            if self.evaluation_thread is not None and landmarks:
                self.evaluation_thread.add_landmarks(landmarks)
        except Exception as e:
            self.error_handler.handle_ui_error(e)
            # Display an error message on the video label
            self._display_error_on_video_label("Error processing frame")
    
    def _on_evaluation_ready(self, evaluation_result):
        """
        Handle the evaluation ready event.
        
        Args:
            evaluation_result: The evaluation result.
        """
        # Update the overall score
        overall_score = evaluation_result.get('weighted_score', 0)
        self.overall_score_label.setText(f"{overall_score:.1f}")
        
        # Update the score bars
        scores = evaluation_result.get('scores', {})
        for aspect, score_bar in self.score_bars.items():
            score = scores.get(aspect, 0)
            score_bar.setValue(int(score))
            
            # Set the color based on the score
            if score >= 80:
                score_bar.setStyleSheet("QProgressBar::chunk { background-color: green; }")
            elif score >= 60:
                score_bar.setStyleSheet("QProgressBar::chunk { background-color: yellow; }")
            else:
                score_bar.setStyleSheet("QProgressBar::chunk { background-color: red; }")
        
        # Update the feedback text
        feedback = evaluation_result.get('feedback', {})
        feedback_text = ""
        
        if 'overall' in feedback:
            feedback_text += f"Overall: {feedback['overall']}\n\n"
        
        for aspect in ['timing', 'posture', 'fluidity', 'stability']:
            if aspect in feedback:
                feedback_text += f"{aspect.capitalize()}: {feedback[aspect]}\n\n"
        
        if 'joints' in feedback and feedback['joints']:
            feedback_text += "Joint-specific feedback:\n"
            for joint_feedback in feedback['joints']:
                feedback_text += f"- {joint_feedback}\n"
            feedback_text += "\n"
        
        if 'common_issues' in feedback and feedback['common_issues']:
            feedback_text += "Common issues:\n"
            for issue_feedback in feedback['common_issues']:
                feedback_text += f"- {issue_feedback}\n"
            feedback_text += "\n"
        
        if 'improvement_suggestions' in feedback and feedback['improvement_suggestions']:
            feedback_text += "Improvement suggestions:\n"
            for suggestion in feedback['improvement_suggestions']:
                feedback_text += f"- {suggestion}\n"
        
        self.feedback_text.setText(feedback_text)
    
    def closeEvent(self, event):
        """
        Handle the window close event.
        
        Args:
            event: The close event.
        """
        # Stop the threads
        if self.video_thread is not None:
            self.video_thread.stop()
        
        if self.evaluation_thread is not None:
            self.evaluation_thread.stop()
        
        event.accept()

def run_application():
    """
    Run the application.
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_application()
