#!/bin/bash
#SBATCH --gpus-per-node=a100:1
#SBATCH --job-name=thesis_train
#SBATCH --output=train-%j.log
#SBATCH --time=04:00:00
#SBATCH --mem=16G

# Help function
function usage {
    echo "Usage: $0 [-e <number>]"
    echo "  -e <number>: number of epochs (default = 10)"
    echo "  -b <number>: batch size (default = 32)"
    exit 1
}

# Process flags
EPOCHS=100
BATCH_SIZE=32
RESUBMITS=0
while getopts ':e:b:r:' opt; do
    case $opt in
    #   (v)   ((VERBOSE++));;
      (e)   EPOCHS=$OPTARG;;
      (b)   BATCH_SIZE=$OPTARG;;
      (r)   RESUBMITS=$OPTARG;;
      (\?)  usage;;
      (:)   # "optional arguments" (missing option-argument handling)
            case $OPTARG in
              (e) usage;; # error, according to our syntax
              (b) usage;; # error, according to our syntax
            #   (l) :;;      # acceptable but does nothing
            esac;;
    esac
done

shift "$OPTIND"
# remaining is "$@"

echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Resubmits: $RESUBMITS"

WORKDIR=$(pwd)
SCRATCH=/scratch/$USER

# Set up the environment (module load, virtual environment, requirements)
chmod +x $WORKDIR/setup_env.sh
source $WORKDIR/setup_env.sh

python3 -u train.py -d $SCRATCH/dataset -e $EPOCHS -b $BATCH_SIZE -r $RESUBMITS

deactivate