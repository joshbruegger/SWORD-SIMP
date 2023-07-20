#!/bin/bash
#SBATCH --gpus-per-node=a100:1
#SBATCH --job-name=val_chkpt
#SBATCH --output=val_ckpt-%j.log
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --ntasks=2

source setup_env.sh
python3 -u validate_checkpoint.py testcheckpoints/test/average_model.pth /scratch/$USER/dataset/valid
deactivate