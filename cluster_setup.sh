#!/bin/bash
#SBATCH --job-name=thesis
#SBATCH --output=job-%j.log
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=01:00:00

# Accept flags from the command line:
# -d: force download of dataset
# -b: force generation of bboxes
# -c: force generation of crops
# -n <number>: number of crops to generate
# combination of flags is possible (e.g. -bc), except for -n

# Help function
function usage {
    echo "Usage: $0 [-d] [-b] [-c] [-e] [-n <number>]"
    echo "  -d: force download of dataset"
    echo "  -b: force generation of bboxes"
    echo "  -c: force generation of crops"
    echo "  -e: force generation of environment"
    echo "  -n <number>: number of crops to generate (default = 10)"
    echo "  combination of flags is possible (e.g. -bc), except for -n"
    exit 1
}

# Process flags
d=false
b=false
c=false
e=false
n=10
while getopts ":dbcen:" opt; do
    case $opt in
        d) d=true;;
        b) b=true;;
        c) c=true;;
        e) e=true;;
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
echo "Starting the job!"
echo "Force download of database = $d"
echo "Force generation of bboxes = $b"
echo "Force generation of crops = $c"
echo "Force generation of environment = $e"
echo "Number of crops = $n"

echo "Loading modules..."
module purge
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0 #3.10.4
module load OpenCV/4.6.0-foss-2022a-contrib #3.10.4

# Save working directory in variable
WORKDIR=$SLURM_SUBMIT_DIR
SCRATCH=/scratch/$USER

# Create scratch directory
# Check if the virtual environment exists
if [ ! -d "$HOME/.envs/thesis_env" ] || [ "$e" = true ] ; then
    echo "Creating virtual environment..."
    # Create the virtual environment
    python3 -m venv $HOME/.envs/thesis_env
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source $HOME/.envs/thesis_env/bin/activate

# make sure the requirements are installed
echo "Installing requirements..."
pip3 install --upgrade pip
pip3 install -r $WORKDIR/requirements.txt

# download dataset using the download script in work dir (force if flag is set)
if [ "$d" = true ] ; then
    python3 -u $WORKDIR/download.py $SCRATCH/dataset/source/images -f
else
    python3 -u $WORKDIR/download.py $SCRATCH/dataset/source/images
fi

# generate the bboxes (force if flag is set)
if [ "$b" = true ] ; then
    python3 -u $WORKDIR/extract_bboxes.py $SCRATCH/dataset/source/images -o $SCRATCH/dataset/source/labels -f
else
    python3 -u $WORKDIR/extract_bboxes.py $SCRATCH/dataset/source/images -o $SCRATCH/dataset/source/labels
fi

# generate crops (force if flag is set)
if [ "$c" = true ] ; then
    python3 -u $WORKDIR/generate_crops.py $SCRATCH/dataset/source/ $n -f
else
    python3 -u $WORKDIR/generate_crops.py $SCRATCH/dataset/source/ $n
fi

deactivate