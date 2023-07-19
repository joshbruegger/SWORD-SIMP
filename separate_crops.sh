#!/bin/bash
#SBATCH --job-name=separate_crops
#SBATCH --output=separate_crops-%j.log
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=01:00:00

SCRATCH=/scratch/$USER

source setup_env.sh

separate=$1

# separate crops in train/val/test sets (force if flag is set), if it fails exit
python3 -u separate_crops.py $SCRATCH/dataset/source/ -o $SCRATCH/dataset/ $separate || exit

deactivate