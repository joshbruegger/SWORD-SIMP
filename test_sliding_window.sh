#!/bin/bash
#SBATCH --job-name=sliding_window
#SBATCH --output=sliding_window-%j.log
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --mem=30G

source setup_env.sh

python -u sliding_window.py /scratch/s4361687/dataset/test/images/07_Traino_S_Domenico_2.psb checkpoints/run_300-400cyclic/ckpt_best.pth 28 -s 1080 -g /scratch/s4361687/dataset/test/labels/07_Traino_S_Domenico_2.txt -y /scratch/s4361687/dataset/dataset.yaml -v -d
