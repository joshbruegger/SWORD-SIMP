#!/bin/bash

echo "Loading modules..."
module update
module purge

module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0
module load OpenCV/4.6.0-foss-2022a-contrib
module load Boost/1.79.0-GCC-11.3.0

# module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
# module load tqdm/4.64.0-GCCcore-11.3.0
# module load matplotlib/3.5.2-foss-2022a
# module unload protobuf/3.19.4-GCCcore-11.3.0
# module unload protobuf-python/3.19.4-GCCcore-11.3.0
# module unload Pillow/9.1.1-GCCcore-11.3.0

module list


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

pip list

pip3 install --upgrade -v --no-cache-dir -r ./requirements.txt