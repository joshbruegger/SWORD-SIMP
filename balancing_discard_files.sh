#!/bin/bash
#SBATCH --job-name=thesis
#SBATCH --output=balancing_analyze-%j.log
#SBATCH --mem=16G
#SBATCH --time=01:00:00

python -u balancing_discard_files.py balancing_files_to_discard.txt