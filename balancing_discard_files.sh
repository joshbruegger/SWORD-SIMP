#!/bin/bash
#SBATCH --job-name=thesis
#SBATCH --output=balancing_discard_files-%j.log
#SBATCH --mem=16G
#SBATCH --time=00:10:00

module purge
module load Python/3.10.4-GCCcore-11.3.0

python -u balancing_discard_files.py balancing_analyze_to_discard_out.txt