#!/bin/bash
#SBATCH --job-name=thesis_preprocess
#SBATCH --output=setup-%j.log
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=05:00:00

# Accept flags from the command line:
# -d: force download of dataset
# -b: force generation of bboxes
# -c: force generation of crops
# -n <number>: number of crops to generate
# combination of flags is possible (e.g. -bc), except for -n

# Help function
function usage {
    echo "Usage: $0 [-d] [-b] [-c] [-e] [-s] [-P] [-n <number>]"
    echo "  -d: force download of dataset"
    echo "  -b: force generation of bboxes"
    echo "  -c: force generation of crops"
    echo "  -e: force generation of environment"
    echo "  -s: force separation of crops in train/val/test sets"
    echo "  -P: force ALL preprocessing steps"
    echo "  -n <number>: number of crops to generate (default = 50)"
    echo "  combination of flags is possible (e.g. -bc), except for -n"
    exit 1
}

# Process flags
download=""
bboxes=""
crops=""
separate=""
env=false
preprocessAll=false
numCrops=50
while getopts ":dbcsPen:" opt; do
    case $opt in
        d) download="-f";;
        b) bboxes="-f";;
        c) crops="-f";;
        s) separate="-f";;
        e) env=true;;
        P) preprocessAll=true;;
        n) numCrops="$OPTARG"
            if ! [[ "$numCrops" =~ ^[0-9]+$ ]] ; then
                echo "error: -n argument is not a number" >&2
                usage
            fi;;
        \?) echo "Invalid option -$OPTARG" >&2
            usage;;
    esac
done

# If P flag is set, set all flags
if [ "$preprocessAll" = true ] ; then
    bboxes="-f"
    crops="-f"
    separate="-f"
fi

# Pretty print welcome message
echo ""
echo ""
echo "88888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888"
echo "SWORD-SIMP Dataset Preprocessor"
echo "88888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888"
echo "Dataset location: $SCRATCH/dataset/source"
echo "Force download of database = $download"
echo "Force generation of bboxes = $bboxes"
echo "Force generation of crops = $crops"
echo "Force generation of environment = $env"
echo "Force separation of crops = $separate"
echo "Number of crops = $numCrops"
echo "88888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888"
echo ""
echo ""

WORKDIR=$(pwd)
SCRATCH=/scratch/$USER

# If e flag is set, delete the virtual environment if it exists
if [ "$env" = true ] ; then
    if [ -d "$HOME/.envs/thesis_env" ] ; then
        echo "Deleting virtual environment..."
        rm -rf $HOME/.envs/thesis_env
    fi
fi

# Set up the environment (module load, virtual environment, requirements)
chmod +x $WORKDIR/setup_env.sh
source $WORKDIR/setup_env.sh

# download dataset using the download script in work dir (force if flag is set), if it fails exit
python3 -u $WORKDIR/download.py $SCRATCH/dataset/source/images $download || exit

# generate the bboxes (force if flag is set), if it fails exit
python3 -u $WORKDIR/extract_bboxes.py $SCRATCH/dataset/source/images -o $SCRATCH/dataset/source/labels $bboxes || exit

# generate crops (force if flag is set), if it fails exit
python3 -u $WORKDIR/generate_crops.py $SCRATCH/dataset/source/ $numCrops $crops || exit

# separate crops in train/val/test sets (force if flag is set), if it fails exit
python3 -u $WORKDIR/separate_crops.py $SCRATCH/dataset/source/ -o $SCRATCH/dataset/ $separate || exit

deactivate