#! /bin/sh

# make sure the requirements are installed
pip install --upgrade pip
pip install -r requirements.txt

# download dataset using the download script
python download.py dataset/source/images

# generate the bboxes
python generate_bboxes.py dataset/source/images -o dataset/source/labels

# generate crops
python generate_crops.py dataset/source/ 10