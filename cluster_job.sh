#!/bin/bash
#SBATCH --job-name=thesis
#SBATCH --output=job-%j.log
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --time=00:30:00

# Clear the module environment
module purge
# Load the Python version that has been used to construct the virtual environment
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0 #3.10.4
module load OpenCV/4.6.0-foss-2022a-contrib #3.10.4

# Save working directory in variable
WORKDIR=$SLURM_SUBMIT_DIR

cd $TMPDIR

# Check if the virtual environment exists
if [ ! -d "$HOME/.envs/thesis_env" ]; then
    # Create the virtual environment
    python3 -m venv $HOME/.envs/thesis_env
fi

# Activate the virtual environment
source $HOME/.envs/thesis_env/bin/activate

# make sure the requirements are installed
pip install --upgrade pip
pip install -r requirements.txt

# download dataset using the download script in work dir
python $WORKDIR/download.py $SCRATCH/dataset/source/images

# generate the bboxes
python $WORKDIR/generate_bboxes.py $SCRATCH/dataset/source/images -o $SCRATCH/dataset/source/labels

# generate crops
python $WORKDIR/generate_crops.py $SCRATCH/dataset/source/ 10

deactivate