#!/bin/bash
#SBATCH --job-name=thesis_preprocess
#SBATCH --output=setup-%j.log
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=01:00:00

# Accept flags from the command line:
# -d: force download of dataset
# -b: force generation of bboxes
# -c: force generation of crops
# -n <number>: number of crops to generate
# combination of flags is possible (e.g. -bc), except for -n

# Help function
function usage {
    echo "Usage: $0 [-d] [-b] [-c] [-e] [-s] [-n <number>]"
    echo "  -d: force download of dataset"
    echo "  -b: force generation of bboxes"
    echo "  -c: force generation of crops"
    echo "  -e: force generation of environment"
    echo "  -s: force separation of crops in train/val/test sets"
    echo "  -n <number>: number of crops to generate (default = 10)"
    echo "  combination of flags is possible (e.g. -bc), except for -n"
    exit 1
}

# Process flags
d=""
b=""
c=""
s=""
e=false
n=10
while getopts ":dbcsen:" opt; do
    case $opt in
        d) d="-f";;
        b) b="-f";;
        c) c="-f";;
        s) s="-f";;
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
echo "Force separation of crops = $s"
echo "Number of crops = $n"

WORKDIR=$(pwd)
SCRATCH=/scratch/$USER

# If e flag is set, delete the virtual environment if it exists
if [ "$e" = true ] ; then
    if [ -d "$HOME/.envs/thesis_env" ] ; then
        echo "Deleting virtual environment..."
        rm -rf $HOME/.envs/thesis_env
    fi
fi

# Set up the environment (module load, virtual environment, requirements)
chmod +x $WORKDIR/setup_env.sh
source $WORKDIR/setup_env.sh

# download dataset using the download script in work dir (force if flag is set)
python3 -u $WORKDIR/download.py $SCRATCH/dataset/source/images $d

# generate the bboxes (force if flag is set)
python3 -u $WORKDIR/extract_bboxes.py $SCRATCH/dataset/source/images -o $SCRATCH/dataset/source/labels $b

# generate crops (force if flag is set)
python3 -u $WORKDIR/generate_crops.py $SCRATCH/dataset/source/ $n $c
python3 -u $WORKDIR/separate_crops.py $SCRATCH/dataset/source/ -o $SCRATCH/dataset/ $s

deactivate