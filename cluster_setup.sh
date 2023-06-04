#!/bin/bash
#SBATCH --job-name=thesis
#SBATCH --output=job-%j.log
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=00:10:00

# Accept flags from the command line:
# -d: force download of dataset
# -b: force generation of bboxes
# -c: force generation of crops
# -n <number>: number of crops to generate
# combination of flags is possible (e.g. -bc), except for -n

# Help function
function usage {
    echo "Usage: $0 [-d] [-b] [-c] [-n <number>]"
    echo "  -d: force download of dataset"
    echo "  -b: force generation of bboxes"
    echo "  -c: force generation of crops"
    echo "  -n <number>: number of crops to generate (default = 10)"
    echo "  combination of flags is possible (e.g. -bc), except for -n"
    exit 1
}

# Process flags
d=false
b=false
c=false
n=10
while getopts ":dbcn:" opt; do
    case $opt in
        d) d=true;;
        b) b=true;;
        c) c=true;;
        n) n="$OPTARG"
            if ! [[ "$n" =~ ^[0-9]+$ ]] ; then
                echo "error: -n argument is not a number" >&2
                usage
            fi;;
        \?) echo "Invalid option -$OPTARG" >&2
            usage;;
    esac
done

# Print flags
echo "Starting the job."
echo "Force download of database = $d"
echo "Force generation of bboxes = $b"
echo "Force generation of crops = $c"
echo "Number of crops = $n"

# Clear the module environment
module purge
# Load the Python version that has been used to construct the virtual environment
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0 #3.10.4
module load OpenCV/4.6.0-foss-2022a-contrib #3.10.4

# Save working directory in variable
WORKDIR=$SLURM_SUBMIT_DIR
SCRATCH=/scratch/$USER

# Create scratch directory
# Check if the virtual environment exists
if [ ! -d "$HOME/.envs/thesis_env" ]; then
    # Create the virtual environment
    python3 -m venv $HOME/.envs/thesis_env
fi

# Activate the virtual environment
source $HOME/.envs/thesis_env/bin/activate

# make sure the requirements are installed
pip install --upgrade pip
pip install -r $WORKDIR/requirements.txt

# download dataset using the download script in work dir (force if flag is set)
if [ "$d" = true ] ; then
    python $WORKDIR/download.py $SCRATCH/dataset/source/images -f
else
    python $WORKDIR/download.py $SCRATCH/dataset/source/images
fi

# generate the bboxes (force if flag is set)
if [ "$b" = true ] ; then
    python $WORKDIR/extract_bboxes.py $SCRATCH/dataset/source/images -o $SCRATCH/dataset/source/labels -f
else
    python $WORKDIR/extract_bboxes.py $SCRATCH/dataset/source/images -o $SCRATCH/dataset/source/labels
fi

# generate crops (force if flag is set)
if [ "$c" = true ] ; then
    python $WORKDIR/generate_crops.py $SCRATCH/dataset/source/ $n -f
else
    python $WORKDIR/generate_crops.py $SCRATCH/dataset/source/ $n
fi

deactivate