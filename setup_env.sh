#!/bin/bash

echo "Loading modules..."
module purge
# module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0 #3.10.4
module load OpenCV/4.6.0-foss-2022a-contrib #3.10.4

# Check if the virtual environment exists
if [ ! -d "$HOME/.envs/thesis_env" ]; then
    echo "Creating virtual environment..."
    # Create the virtual environment
    python3 -m venv $HOME/.envs/thesis_env
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source $HOME/.envs/thesis_env/bin/activate

# make sure the requirements are installed
echo "Installing requirements..."
pip3 install --upgrade pip
pip3 install --upgrade wheel
pip3 install -v -r ./requirements.txt