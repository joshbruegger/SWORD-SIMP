#!/bin/bash
#SBATCH --job-name=thesis
#SBATCH --output=job-%j.log
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --time=00:30:00

# Clear the module environment
module purge
# Load the Python version that has been used to construct the virtual environment
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0

# Check if the virtual environment exists
if [ ! -d "$HOME/.envs/thesis_env" ]; then
    # Create the virtual environment
    python3 -m venv $HOME/.envs/thesis_env
fi

# Activate the virtual environment
source $HOME/.envs/thesis_env/bin/activate

# Run the setup script
chmod +x setup.sh
bash setup.sh

deactivate