#!/bin/bash

# Set Python executable (e.g., python3)
PYTHON=python3

# Create a virtual environment (optional)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
fi

# Activate virtual environment (if created)
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies from requirements.txt
if [ -f requirements.txt ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found!"
    exit 1
fi

# Run the Python script
echo "Running Python script..."
$PYTHON classification.py

# Deactivate virtual environment (optional)
deactivate

