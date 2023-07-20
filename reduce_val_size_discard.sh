#!/bin/bash
#SBATCH --job-name=thesis
#SBATCH --output=reduce_val_size_discard-%j.log
#SBATCH --mem=16G
#SBATCH --time=00:10:00

module purge
module load Python/3.10.4-GCCcore-11.3.0

python -u balancing_discard_files.py reduce_val_size_to_discard_out.txt