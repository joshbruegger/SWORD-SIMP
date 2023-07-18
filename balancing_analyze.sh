#!/bin/bash
#SBATCH --job-name=thesis
#SBATCH --output=balancing_analyze-%j.log
#SBATCH --mem=16G
#SBATCH --time=00:10:00

source setup_env.sh

python -u balancing_analyze.py /scratch/s4361687/small

deactivate