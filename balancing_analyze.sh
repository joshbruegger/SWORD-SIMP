#!/bin/bash
#SBATCH --job-name=thesis
#SBATCH --output=balancing_analyze-%j.log
#SBATCH --mem=16G
#SBATCH --time=04:00:00

SCRATCH=/scratch/$USER

source setup_env.sh

python -u balancing_analyze.py $SCRATCH/dataset

deactivate