import os
from collections import defaultdict
from tqdm import tqdm
import argparse
from natsort import os_sorted
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import numpy as np
import math
from pathlib import Path

# Set up logger
logging.basicConfig(level=logging.INFO)

def read_label_files(folder: Path, crops : pd.DataFrame):
    """
    Read labels from files and return a dictionary with classes and counts per file.
    Returns a dictionary with class names as keys and a dictionary with filenames and counts as values.
    """
    filename_classes = pd.DataFrame(columns=["filename", "classes"])
    files = []
    classes = defaultdict(lambda: defaultdict(int))
    for file in tqdm(folder.iterdir(), desc="Reading files"):
        if file.suffix != ".txt" or "_classes" in file.stem:
            continue
        files.append(file)
        classes_in_file = []
        with file.open() as f:
            for line in f:
                class_name = line.strip().split()[0]
                classes_in_file.append(class_name)
                classes[class_name][file.stem] += 1
        
        # add a new row to filename_classes
        filename_classes = pd.concat([filename_classes, pd.DataFrame({"filename": [file.stem], "classes": [classes_in_file]})])

    return classes, files, crops.merge(filename_classes, on="filename")

def read_crops_location(file : Path):
    """Read the crop location from a file.
    The file should contain one line per crop, with the following format:
    <filename> <x_min> <y_min> <x_max> <y_max>
    Returns a dataframe with the filename and crop location, and the largest dimension of the all the crops.
    """
    df = pd.DataFrame(columns=["filename", "location"])
    max_dimension = 0
    with file.open() as f:
        curr_painting = ""
        for line in f:
            line = line.strip().split()
            painting = line[0]

            idx = 0 if painting != curr_painting else idx + 1
            curr_painting = painting
            filename = f"{painting}_{idx}"

            x_min, y_min, x_max, y_max = [int(x) for x in line[1:]]
            dimension = max(x_max - x_min, y_max - y_min)
            max_dimension = max(max_dimension, dimension)

            df = pd.concat([df, pd.DataFrame({"filename": [filename], "location": [line[1:]]})])

    return df, max_dimension

def read_painting_sizes(file : Path):
    """
    Read the painting sizes from a file.
    The file should contain one line per painting, with the following format:
    <paintingName> <width> <height>
    Returns a dataframe with the paintingName, width, height.
    """
    df = pd.DataFrame(columns=["paintingName", "width", "height"])
    with file.open() as f:
        for line in f:
            line = line.strip().split()
            name = line[0]
            width, height = [int(x) for x in line[1:]]
            df = pd.concat([df, pd.DataFrame({"paintingName": [name], "width": [width], "height": [height]})])

    return df

def find_test_painting(classes):
    # ignore classes that only appear in one painting
    most_shared = defaultdict(int)
    for class_name, paintings in tqdm(classes.items(), desc="Finding most shared classes"):
        # continue if class only appears in one painting
        if len(paintings) == 1:
            # print(f"Skipping {class_name} because it only appears in one painting, namely {list(paintings.keys())[0]}")
            continue
        for painting_name, count in paintings.items():
            most_shared[painting_name] += 1

    # sort by number of shared classes
    most_shared = sorted(most_shared.items(), key=lambda x: x[1], reverse=True)
    return most_shared

def analyze_distribution(train_files, validation_files, args):
    print(f"Number of files in training set: {len(train_files)}")
    print(f"Number of files in validation set: {len(validation_files)}")
    # Make a histogram of the classes in the training and validation sets
    train_classes = defaultdict(int)
    validation_classes = defaultdict(int)
    for filename in tqdm(train_files, desc="Creating histogram of training classes"):
        with open(os.path.join(args.location, "cropped", "labels", filename), "r") as file:
            for line in file:
                class_name = line.strip().split()[0]
                train_classes[class_name] += 1
    for filename in tqdm(validation_files, desc="Creating histogram of validation classes"):
        with open(os.path.join(args.location, "cropped", "labels", filename), "r") as file:
            for line in file:
                class_name = line.strip().split()[0]
                validation_classes[class_name] += 1
    # Sort the histograms
    train_classes = sorted(train_classes.items(), key=lambda x: x[1], reverse=True)
    validation_classes = sorted(validation_classes.items(), key=lambda x: x[1], reverse=True)
    # Calculate the total number of instances in the training and validation sets
    total_train = sum([x[1] for x in train_classes])
    total_validation = sum([x[1] for x in validation_classes])
    # Print the table
    print("Class name".ljust(20) + "Training set".ljust(20) + "Validation set".ljust(20))
    for i in range(len(train_classes)):
        print(train_classes[i][0].ljust(20) + f"{round(train_classes[i][1] / total_train * 100, 2)}%".ljust(20) + f"{round(validation_classes[i][1] / total_validation * 100, 2)}%".ljust(20))

def move_files(files: list, src_folder: Path, dest_folder: Path):
    """Move files from the source to the destination folder."""
    for file in tqdm(files, desc=f"Moving {len(files)} files to {dest_folder}"):
        src_file = src_folder / file
        if src_file.exists():
            shutil.move(str(src_file), str(dest_folder))

def split_train_valid_test_files(label_files : list, test_file: str):
    test_files = []

    df_file_classes = pd.DataFrame(columns=["file", "classes"]) # dataframe with the file name and a list of classes in that file
    class_counts = {}
    tot_classes = 0

    for label_file in tqdm(label_files, desc="Creating list of classes"):\
    
        if test_file in label_file.stem:
            test_files.append(label_file)
            continue

        classes_in_file = []
        with label_file.open() as file:
            for line in file:
                class_name = line.strip().split()[0]
                classes_in_file.append(class_name)
                tot_classes += 1
                if class_name in class_counts:
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1
        
        df_file_classes = pd.concat([df_file_classes, pd.DataFrame({"file": [label_file], "classes": [classes_in_file]})])

    # Check if there are any classes that only appear in one file
    classes_in_one_file = []
    for class_name, count in class_counts.items():
        if count == 1:
            classes_in_one_file.append(class_name)
    if len(classes_in_one_file) > 0:
        print(f"WARNING: {len(classes_in_one_file)} classes only appear in one file")
        print(classes_in_one_file)

    train_files, validation_files = train_test_split(df_file_classes, test_size=0.20, random_state=42, stratify=df_file_classes["classes"])

    return train_files, validation_files, test_files

def max_cells(length, cell_min_size, gutter_size):
    """
    Calculate the maximum number of cells of size >= cell_min_size that can fit in a given length, with a fixed gutter size between the cells.
    """
    return math.floor((length + gutter_size) / (cell_min_size + gutter_size))

def separate_crops_into_cells(paintings, crops, crop_size=1080):
    gutter_size = crop_size # The fixed size of the gutter between the grid cells
    min_cell_size = 2 * crop_size # The minimum size of a grid cell

    paintings['n_cells_width'] = paintings['width'].apply(lambda x: max_cells(x, min_cell_size, gutter_size))
    paintings['n_cells_height'] = paintings['height'].apply(lambda x: max_cells(x, min_cell_size, gutter_size))

    # Extract 'paintingName' from 'filename' in the crops dataframe
    # use regex to remove the last _number from the filename
    crops['paintingName'] = crops['filename'].str.split('_').str[0:-1].str.join('_')

    def get_grid_cell(row):
        # Get the grid size for the current painting
        painting = paintings.loc[paintings['paintingName'] == row['paintingName']].iloc[0]

        cell_width = (painting['width'] - (painting['n_cells_width'] - 1) * gutter_size) / painting['n_cells_width']
        cell_height = (painting['height'] - (painting['n_cells_height'] - 1) * gutter_size) / painting['n_cells_height']

        x_min, y_min, x_max, y_max = [int(x) for x in row['location']]
        center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2

        # Determine the grid cell coordinates
        grid_x = math.floor(center_x / (cell_width + gutter_size))
        grid_y = math.floor(center_y / (cell_height + gutter_size))

        return (grid_x, grid_y)

    # Apply the grid cell identification function to each crop
    crops['grid_cell'] = crops.apply(get_grid_cell, axis=1)

    return crops

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("location", help="Location of the dataset")
    parser.add_argument("--output", '-o', help="Location to save the separated dataset", default=None)
    parser.add_argument('--force', '-f', help="Force overwrite of existing files", action='store_true')
    return parser.parse_args()

def print_welcome(args):
    padding = 140
    print("\n\n")
    print(" SWORD-SIMP Dataset Separator ".center(padding, "8"))
    print(f" Dataset location: {args.location} ".center(padding))
    print(f" Output location: {args.output} ".center(padding))
    print(f" Force overwrite: {args.force} ".center(padding))
    print("".center(padding, "8"))
    print("\n\n")

def main():
    args = get_args()
    print_welcome(args)

    dataset_dir = Path(args.location)
    output_dir = Path(args.output) if args.output else dataset_dir / "separated"

    if output_dir.exists() and not args.force:
        logging.error("Dataset has already been separated. Use the -f flag to overwrite existing files.")
        return
    
    crops, max_crop_dimension = read_crops_location(dataset_dir / "cropped" / "crops.txt")
    painting_classes, label_files, crops = read_label_files(dataset_dir / "cropped" / "labels", crops)
    test_painting = find_test_painting(painting_classes)
    print(f"Painting  that will be used for testing: {test_painting[0][0]} ({test_painting[0][1]} classes)")


    paintings = read_painting_sizes(dataset_dir / "painting_sizes.txt")
    crops = separate_crops_into_cells(paintings, crops, max_crop_dimension)
    # train_files, validation_files, test_files = split_train_valid_test_files(label_files, test_painting[0][0])

    print(crops)

    exit()

    # Create output directories
    for split in ["train", "valid", "test"]:
        for folder in ["images", "labels"]:
            os.makedirs(output_dir / split / folder, exist_ok=True)

    # Move files
    move_files(train_files, dataset_dir / "cropped" / "images", output_dir / "train" / "images")
    move_files(train_files, dataset_dir / "cropped" / "labels", output_dir / "train" / "labels")
    move_files(validation_files, dataset_dir / "cropped" / "images", output_dir / "valid" / "images")
    move_files(validation_files, dataset_dir / "cropped" / "labels", output_dir / "valid" / "labels")
    move_files(test_files, dataset_dir / "cropped" / "images", output_dir / "test" / "images")
    move_files(test_files, dataset_dir / "cropped" / "labels", output_dir / "test" / "labels")

    # save yaml file
    with open(dataset_dir / "classes.txt", "r") as file:
        classes = [line.strip() for line in file]
    with open(dataset_dir / "data.yaml", "w") as file:
        file.write(f"train: {os.path.join('train', 'images')}\n")
        file.write(f"val: {os.path.join('valid', 'images')}\n")
        file.write(f"test: {os.path.join('test', 'images')}\n")
        file.write(f"nc: {len(classes)}\n")
        file.write("names:\n")
        for class_name in classes:
            file.write(f"- '{class_name}'\n")

if __name__ == "__main__":
    main()
