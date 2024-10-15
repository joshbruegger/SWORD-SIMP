#!/bin/bash
#SBATCH --job-name=thesis_preprocess
#SBATCH --output=export_paintings-%j.log
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=01:00:00

SCRATCH=/scratch/$USER

# Set up the environment (module load, virtual environment, requirements)
chmod +x setup_env.sh
source setup_env.sh

python3 -u export_paintings.py $SCRATCH/dataset/source/images --output $SCRATCH/dataset/exported --max_dim 11520