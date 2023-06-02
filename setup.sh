# make sure the requirements are installed
pip install -r requirements.txt

# download dataset using the download script
python download.py

# generate crops
python generate_crops.py