#!/bin/bash
#SBATCH --gpus-per-node=a100:1
#SBATCH --job-name=thesis_train
#SBATCH --output=train-%j.log
#SBATCH --time=00:10:00
#SBATCH --mem=4G

# Help function
function usage {
    echo "Usage: $0 [-e <number>]"po
    echo "  -e <number>: number of epochs (default = 10)"
    exit 1
}

# Process flags
e=10
while getopts ":e:" opt; do
    case $opt in
        n) n="$OPTARG"
            if ! [[ "$e" =~ ^[0-9]+$ ]] ; then
                echo "error: -e argument is not a number" >&2
                usage
            fi;;
        \?) echo "Invalid option -$OPTARG" >&2
            usage;;
    esac
done

WORKDIR=$(pwd)
SCRATCH=/scratch/$USER

# Set up the environment (module load, virtual environment, requirements)
chmod +x $WORKDIR/setup_env.sh
source $WORKDIR/setup_env.sh

python3 -u train.py -d $SCRATCH/dataset -c $SCRATCH/dataset/classes.txt -e $e

deactivate