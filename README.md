# Dunhuang Dance Evaluation System

A computer vision system to evaluate and score Dunhuang flying dance movements in real-time, providing feedback to help learners improve their technique.

## Features

- Skeleton point detection using MediaPipe to track dancer movements
- Motion feature extraction from skeleton points to characterize movements
- Comparison of learner movements to standard reference movements using DTW algorithm
- Real-time scoring and feedback generation
- PyQt5-based UI to display scores and feedback

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dunhuang_dance_evaluation.git
cd dunhuang_dance_evaluation
```

2. Create a virtual environment and activate it:
```bash
python -m venv dunhuang_dance_env
source dunhuang_dance_env/bin/activate  # On Windows: dunhuang_dance_env\Scripts\activate
```

3. Install the package and its dependencies:
```bash
pip install -e .
```

## Usage

### Running the Application

To start the application with the default settings:
```bash
python src/main.py
```

### Command Line Arguments

- `--video PATH`: Path to a video file to process
- `--camera INDEX`: Camera index to use (default: 0)
- `--reference PATH`: Path to a reference movement file
- `--debug`: Enable debug mode

### Example

```bash
python src/main.py --camera 0 --reference data/1-FeiTian/movement1.bvh --debug
```

## System Architecture

The system consists of the following components:

1. **Skeleton Detection**: Extracts skeleton points from video input using MediaPipe.
2. **Motion Feature Extraction**: Extracts motion features from skeleton points.
3. **DTW Comparison**: Compares learner movements to reference movements using DTW algorithm.
4. **Scoring and Feedback**: Calculates similarity scores and generates improvement suggestions.
5. **User Interface**: Displays scores and feedback in real-time.

## Data Format

The system supports the following data formats:

- BVH (Biovision Hierarchy) files for reference movements
- Video files (MP4, AVI, MOV) for both reference and learner movements
- Live camera input for learner movements

## License

This project is licensed under the MIT License - see the LICENSE file for details.
