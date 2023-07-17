#!/bin/bash
#SBATCH --gpus-per-node=a100:1
#SBATCH --job-name=thesis_train
#SBATCH --output=train-%j.log
#SBATCH --time=04:00:00
#SBATCH --mem=10G

# Help function
function usage {
    echo "Usage: $0 [-e <number>]"
    echo "  -e <number>: number of epochs (default = 10)"
    echo "  -b <number>: batch size (default = 32)"
    exit 1
}

# Process flags
e=100
b=32
echo "Epochs: $e"
echo "Batch size: $b"

WORKDIR=$(pwd)
SCRATCH=/scratch/$USER

# Set up the environment (module load, virtual environment, requirements)
chmod +x $WORKDIR/setup_env.sh
source $WORKDIR/setup_env.sh

python3 -u train.py -d $SCRATCH/dataset -e $e -b $b

deactivate