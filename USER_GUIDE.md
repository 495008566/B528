# User Guide for Dunhuang Dance Evaluation System

This guide provides instructions on how to use the Dunhuang Dance Evaluation System for evaluating and improving Dunhuang flying dance movements.

## Getting Started

### Starting the Application

To start the application with default settings:

```bash
./run.sh
```

### Command Line Arguments

The application supports the following command line arguments:

- `--video PATH`: Path to a video file to process
- `--camera INDEX`: Camera index to use (default: 0)
- `--reference PATH`: Path to a reference movement file
- `--debug`: Enable debug mode

Examples:

```bash
# Use the default camera (index 0)
./run.sh --camera 0

# Process a video file
./run.sh --video path/to/video.mp4

# Use a specific reference movement
./run.sh --reference data/1-FeiTian/02-PanTuiXieShenShi.bvh

# Enable debug mode
./run.sh --debug
```

### Running the Demo

To run the demo with sample data:

```bash
./run_demo.sh
```

## User Interface

The main window of the application consists of the following components:

### Video Display

The main area of the window displays the video feed with skeleton points overlaid on the dancer. The skeleton points are color-coded to indicate the quality of the movement:

- **Green**: Good alignment with reference movement
- **Yellow**: Moderate alignment
- **Red**: Poor alignment

### Control Panel

The control panel on the right side of the window provides the following controls:

#### Input Source

- **Camera**: Select the camera to use for live input
- **Video File**: Select a video file to process
- **Browse**: Browse for a video file

#### Reference Movement

- **Category**: Select the category of reference movement
- **Movement**: Select the specific reference movement
- **Browse**: Browse for a reference movement file

#### Evaluation Settings

- **Sensitivity**: Adjust the sensitivity of the evaluation
- **Feedback Level**: Adjust the level of detail in the feedback
- **Real-time Mode**: Toggle real-time evaluation mode

#### Controls

- **Start**: Start the evaluation
- **Stop**: Stop the evaluation
- **Reset**: Reset the evaluation
- **Save Results**: Save the evaluation results

### Feedback Panel

The feedback panel at the bottom of the window displays the evaluation results and feedback:

#### Scores

- **Overall Score**: The overall similarity score (0-100)
- **Timing**: Score for timing accuracy (0-100)
- **Posture**: Score for posture accuracy (0-100)
- **Fluidity**: Score for movement fluidity (0-100)
- **Stability**: Score for stability (0-100)

#### Feedback

- **Overall Feedback**: General feedback on the performance
- **Aspect-specific Feedback**: Feedback on specific aspects of the performance
- **Improvement Suggestions**: Suggestions for improving the performance

## Workflow

### Basic Workflow

1. **Select Input Source**: Choose a camera or video file
2. **Select Reference Movement**: Choose a reference movement
3. **Start Evaluation**: Click the Start button
4. **Perform Dance**: Perform the dance movement
5. **View Feedback**: Review the scores and feedback
6. **Adjust and Retry**: Adjust your movement based on the feedback and try again

### Advanced Features

#### Recording and Playback

1. **Record**: Click the Record button to start recording
2. **Stop Recording**: Click the Stop button to stop recording
3. **Save Recording**: Enter a filename and click Save
4. **Load Recording**: Click the Load button and select a recording file
5. **Play Recording**: Click the Play button to play the recording

#### Comparing Multiple Performances

1. **Save Results**: Save the results of each performance
2. **Load Results**: Load multiple result files
3. **Compare**: Click the Compare button to see a comparison

#### Customizing Evaluation Parameters

1. **Settings**: Click the Settings button
2. **Parameters**: Adjust the evaluation parameters
3. **Save Settings**: Click Save to apply the settings

## Tips for Best Results

1. **Lighting**: Ensure good lighting conditions for accurate skeleton detection
2. **Camera Position**: Position the camera to capture the full body
3. **Clothing**: Wear clothing that contrasts with the background
4. **Space**: Ensure sufficient space for movement
5. **Practice**: Start with simple movements and progress to more complex ones

## Troubleshooting

### Common Issues

#### Poor Skeleton Detection

- Improve lighting conditions
- Ensure the full body is visible
- Wear clothing that contrasts with the background
- Avoid rapid movements

#### Low Scores Despite Good Performance

- Check that the correct reference movement is selected
- Adjust the sensitivity settings
- Ensure the camera is positioned correctly

#### Application Crashes

- Check the logs in the `logs` directory
- Ensure all dependencies are installed correctly
- Try restarting the application

## Getting Help

If you encounter issues not covered in this guide, please:

1. Check the logs in the `logs` directory
2. Refer to the [Troubleshooting](TROUBLESHOOTING.md) guide
3. Contact support at support@dunhuangdance.com
