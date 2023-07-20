#!/bin/bash
#SBATCH --job-name=thesis
#SBATCH --output=reduce_val_size-%j.log
#SBATCH --mem=4G
#SBATCH --time=00:10:00

source setup_env.sh

python -u reduce_val_size.py /scratch/s4361687/dataset 65 --not-under 200

deactivate