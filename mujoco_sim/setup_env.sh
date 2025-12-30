#!/bin/bash

# Define environment name
ENV_NAME="iris"

echo "--- Starting Configuration for IRIS Environment ---"

# 1. Create Conda environment with Python 3.11
echo ">>> Creating Conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.11 -y

# 2. Initialize shell for conda (needed for 'conda activate' in scripts)
eval "$(conda shell.bash hook)"

# 3. Activate the environment
echo ">>> Activating environment..."
conda activate $ENV_NAME

# 4. Install dependencies via pip
if [ -f "requirements.txt" ]; then
    echo ">>> Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo ">>> requirements.txt not found. Installing core packages manually..."
    pip install mujoco numpy pynput
fi

echo "----------------------------------------------------"
echo "Setup Complete!"
echo "To start your simulation, run:"
echo "conda activate $ENV_NAME"
echo "python teleop_script.py"
echo "----------------------------------------------------"