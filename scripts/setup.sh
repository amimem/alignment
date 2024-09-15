#!/bin/bash

# Define log file path
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/setup.log"

# Create logs directory if it doesn't exist
if [ ! -d "$LOG_DIR" ]; then
    echo "Creating logs directory..."
    mkdir -p "$LOG_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create logs directory." >&2
        exit 1
    fi
fi

# Function to log messages
log_message() {
    echo "$1" >> "$LOG_FILE"
}

# Load environment variables
echo "Sourcing /etc/profile..." | tee -a "$LOG_FILE"
source /etc/profile

# Load necessary modules
echo "Loading Python, libffi, and CUDA toolkit modules..." | tee -a "$LOG_FILE"
module load python/3.10.lua libffi cudatoolkit/12.3.2
if [ $? -ne 0 ]; then
    log_message "Warning: Failed to load required modules."
fi

# Define variables
VENV_DIR="$HOME/.venv/align"
REQUIREMENTS_FILE="requirements.txt"

# Create virtual environment if it doesn't exist
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at '$VENV_DIR'." | tee -a "$LOG_FILE"
else
    echo "Creating virtual environment..." | tee -a "$LOG_FILE"
    python -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        log_message "Warning: Failed to create virtual environment."
    fi
fi

# Activate the virtual environment
echo "Activating virtual environment..." | tee -a "$LOG_FILE"
source "$VENV_DIR/bin/activate"
if [ $? -ne 0 ]; then
    log_message "Warning: Failed to activate virtual environment."
fi

# Check if requirements.txt exists and install dependencies
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    log_message "Warning: '$REQUIREMENTS_FILE' not found."
else
    echo "Installing dependencies..." | tee -a "$LOG_FILE"
    pip install --no-index -r "$REQUIREMENTS_FILE"
    if [ $? -ne 0 ]; then
        log_message "Warning: Failed to install dependencies."
    fi
fi

echo "Setup completed. Check '$LOG_FILE' for any issues." | tee -a "$LOG_FILE"