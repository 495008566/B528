# Installation Guide for Dunhuang Dance Evaluation System

This guide provides detailed instructions for installing and setting up the Dunhuang Dance Evaluation System.

## System Requirements

- **Operating System**: Linux, macOS, or Windows
- **Python**: Version 3.8 or higher
- **Hardware**: 
  - CPU: Intel i5/AMD Ryzen 5 or better (for real-time performance)
  - RAM: 8GB or more
  - Camera: Webcam or external camera for live input
  - GPU: Optional but recommended for improved performance

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/dunhuang_dance_evaluation.git
cd dunhuang_dance_evaluation
```

### 2. Create a Virtual Environment

#### On Linux/macOS:
```bash
python -m venv dunhuang_dance_env
source dunhuang_dance_env/bin/activate
```

#### On Windows:
```bash
python -m venv dunhuang_dance_env
dunhuang_dance_env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -e .
```

This will install all required dependencies listed in the `requirements.txt` file, including:
- numpy
- opencv-python
- mediapipe
- PyQt5
- scipy
- scikit-learn
- fastdtw
- matplotlib

### 4. Verify Installation

To verify that the installation was successful, run the test script:

```bash
./run_test.sh
```

You should see output indicating that all tests have passed.

## Troubleshooting

### Common Issues

#### MediaPipe Installation Issues

If you encounter issues with MediaPipe installation:

```bash
pip uninstall mediapipe
pip install mediapipe==0.8.9
```

#### OpenCV Issues

If you encounter issues with OpenCV:

```bash
pip uninstall opencv-python
pip install opencv-python==4.5.0
```

#### PyQt5 Issues

If you encounter issues with PyQt5:

```bash
pip uninstall PyQt5
pip install PyQt5==5.15.0
```

### Platform-Specific Issues

#### Linux

Ensure you have the required system libraries:

```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

#### macOS

Ensure you have Homebrew installed and then:

```bash
brew install cmake
```

#### Windows

Ensure you have the Visual C++ Redistributable installed.

## Data Setup

### Reference Movement Data

1. Extract the provided dataset to the `data` directory:
```bash
mkdir -p data
unzip path/to/dataset.zip -d data/
```

2. The system expects the following directory structure:
```
data/
├── 1-FeiTian/
│   ├── 01-Movement1.bvh
│   ├── 02-Movement2.bvh
│   └── ...
├── 2-PuSa/
│   ├── 01-Movement1.bvh
│   └── ...
└── ...
```

## Next Steps

After installation, refer to the [User Guide](USER_GUIDE.md) for instructions on how to use the system.
