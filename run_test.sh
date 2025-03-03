#!/bin/bash

# Activate the virtual environment
source ~/dunhuang_dance_env/bin/activate

# Create logs directory if it doesn't exist
mkdir -p ~/dunhuang_dance_evaluation/logs

# Run the test script
cd ~/dunhuang_dance_evaluation
python test_system.py

# Exit with the test script's exit code
exit $?
