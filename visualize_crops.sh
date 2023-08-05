#!/bin/bash
#SBATCH --job-name=visualize_crops
#SBATCH --output=visualize_crops-%j.log
#SBATCH --time=00:20:00
#SBATCH --mem=16G

for i in {1..50}
do

IMAGE=$(ls /scratch/s4361687/dataset/valid/images | shuf -n 1)

# Extract the name of the image without the file extension
NAME=$(echo $IMAGE | cut -f 1 -d '.')
echo $NAME

IMAGE=/scratch/s4361687/dataset/valid/images/$IMAGE
LABEL=/scratch/s4361687/dataset/valid/labels/$NAME.txt

echo $LABEL
echo $IMAGE

# # Run the script
pdm run -v python -u sliding_window.py $IMAGE checkpoints/run_400-500cyclic/ckpt_best.pth 28 -g $LABEL -y /scratch/s4361687/dataset/dataset.yaml -v &

wait $!

mv ground_truth.png valid_eval/${NAME}_ground_truth.png
mv prediction.png valid_eval/${NAME}_prediction.png

done