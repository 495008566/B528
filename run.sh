#!/bin/bash

# Activate the virtual environment
source ~/dunhuang_dance_env/bin/activate

# Run the application
python ~/dunhuang_dance_evaluation/src/main.py "$@"
