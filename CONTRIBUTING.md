# Contributing to Dunhuang Dance Evaluation System

Thank you for your interest in contributing to the Dunhuang Dance Evaluation System! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to foster an inclusive and respectful community.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Familiarity with PyQt5, OpenCV, and MediaPipe

### Setting Up the Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/dunhuang_dance_evaluation.git
   cd dunhuang_dance_evaluation
   ```
3. Create a virtual environment:
   ```bash
   python -m venv dunhuang_dance_env
   source dunhuang_dance_env/bin/activate  # On Windows: dunhuang_dance_env\Scripts\activate
   ```
4. Install the package in development mode:
   ```bash
   pip install -e .
   ```
5. Install development dependencies:
   ```bash
   pip install pytest pytest-cov flake8 black
   ```

## Development Workflow

### Branching Strategy

- `main`: The main branch contains the stable version of the code
- `develop`: The development branch contains the latest changes
- Feature branches: Create a new branch for each feature or bug fix

### Creating a Feature Branch

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

### Making Changes

1. Make your changes to the codebase
2. Write or update tests for your changes
3. Run the tests to ensure they pass:
   ```bash
   pytest
   ```
4. Format your code using Black:
   ```bash
   black .
   ```
5. Check your code with Flake8:
   ```bash
   flake8
   ```

### Committing Changes

1. Stage your changes:
   ```bash
   git add .
   ```
2. Commit your changes with a descriptive message:
   ```bash
   git commit -m "Add feature: your feature description"
   ```
3. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

### Creating a Pull Request

1. Go to the original repository on GitHub
2. Click on "Pull Requests" and then "New Pull Request"
3. Select your fork and the feature branch
4. Fill in the pull request template with details about your changes
5. Submit the pull request

## Project Structure

The project is organized into the following modules:

- `src/skeleton_detection/`: Skeleton point detection
- `src/motion_features/`: Motion feature extraction
- `src/dtw_comparison/`: DTW comparison
- `src/scoring/`: Scoring and feedback
- `src/ui/`: User interface
- `src/utils/`: Utility functions
- `tests/`: Tests for all modules

## Coding Standards

### Python Style Guide

- Follow PEP 8 for code style
- Use Black for code formatting
- Use docstrings for all functions, classes, and modules
- Use type hints where appropriate

### Documentation

- Update the documentation when making changes
- Document all public functions, classes, and modules
- Include examples where appropriate

### Testing

- Write tests for all new features and bug fixes
- Maintain or improve test coverage
- Run the test suite before submitting a pull request

## Adding New Features

### Skeleton Detection

To add support for a new skeleton detection method:

1. Create a new class in `src/skeleton_detection/`
2. Implement the required methods (see `pose_detector.py` for reference)
3. Update the factory method in `skeleton_extractor.py`

### Motion Features

To add a new motion feature:

1. Add the feature extraction method to `feature_extractor.py`
2. Update the feature selection logic in `feature_selector.py`
3. Add tests for the new feature

### DTW Comparison

To modify the DTW comparison:

1. Update the comparison logic in `dtw_comparator.py`
2. Tune the parameters in `parameter_tuner.py`
3. Test the changes with various movements

### Scoring and Feedback

To add a new scoring aspect:

1. Add the scoring logic to `score_calculator.py`
2. Add the feedback generation logic to `feedback_generator.py`
3. Update the UI to display the new aspect

### User Interface

To modify the UI:

1. Update the UI layout in `main_window.py`
2. Add new controls or displays as needed
3. Ensure the UI is responsive and user-friendly

## Release Process

1. Update the version number in `setup.py`
2. Update the changelog
3. Create a pull request to merge `develop` into `main`
4. Once approved, merge the pull request
5. Create a new release on GitHub with release notes
6. Build and publish the package

## Getting Help

If you need help with contributing, please:

1. Check the documentation
2. Look for similar issues on GitHub
3. Ask for help in the project's communication channels
4. Open an issue on GitHub

Thank you for contributing to the Dunhuang Dance Evaluation System!
