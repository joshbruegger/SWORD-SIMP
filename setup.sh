#! /bin/sh

# make sure the requirements are installed
pip install --upgrade pip
pip install -r requirements.txt

# download dataset using the download script
python download.py dataset/source/

# generate crops
python generate_crops.py dataset/source/ 2