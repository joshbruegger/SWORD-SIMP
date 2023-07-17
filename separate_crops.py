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
from collections import Counter
import math
from pathlib import Path

# Set up logger
logging.basicConfig(level=logging.INFO)


def read_label_files(folder: Path, crops: pd.DataFrame):
    """
    Read labels from files and return a dictionary with classes and counts per file.
    Returns a dictionary with class names as keys and a dictionary with filenames and counts as values.
    """
    filename_classes = pd.DataFrame(columns=["filename", "classes"])
    files = []
    classes = defaultdict(lambda: defaultdict(int))
    for file in tqdm(
        folder.iterdir(), desc="Reading files", total=len(list(folder.iterdir()))
    ):
        if file.suffix != ".txt" or "_classes" in file.stem:
            continue
        files.append(file)
        classes_in_file = []
        with file.open() as f:
            for line in f:
                class_name = line.strip().split()[0]
                classes_in_file.append(class_name)
                classes[class_name]["_".join(file.stem.split("_")[:-1])] += 1

        # add a new row to filename_classes
        filename_classes = pd.concat([
            filename_classes,
            pd.DataFrame({"filename": [file.stem], "classes": [classes_in_file]}),
        ])

    return classes, crops.merge(filename_classes, on="filename")


def read_crops_location(file: Path):
    """
    Read the crop location from a file.
    The file should contain one line per crop, with the following format:
    <filename> <x_min> <y_min> <x_max> <y_max>
    Returns a dataframe with the filename and crop location, and the largest dimension of the all the crops.
    """
    df = pd.DataFrame(columns=["filename", "paintingName", "location"])
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

            df = pd.concat([
                df,
                pd.DataFrame({
                    "filename": [filename],
                    "paintingName": painting,
                    "location": [line[1:]],
                }),
            ])

    return df, max_dimension


def read_painting_sizes(file: Path):
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
            df = pd.concat([
                df,
                pd.DataFrame(
                    {"paintingName": [name], "width": [width], "height": [height]}
                ),
            ])

    return df


def move_files(
    files: pd.DataFrame,
    label_dir: Path,
    image_dir: Path,
    label_dest_dir: Path,
    image_dest_dir: Path,
    name="files",
):
    """Move files mathcing the "filename" column in the files dataframe from the label_dir and image_dir to the label_dest_dir and image_dest_dir respectively."""
    for _, row in tqdm(files.iterrows(), desc=f"Moving {name}", total=len(files)):
        shutil.move(
            label_dir / f"{row['filename']}.txt",
            label_dest_dir / f"{row['filename']}.txt",
        )
        shutil.move(
            image_dir / f"{row['filename']}.jpg",
            image_dest_dir / f"{row['filename']}.jpg",
        )


def max_cells(length, cell_min_size, gutter_size):
    """
    Calculate the maximum number of cells of size >= cell_min_size that can fit in a given length, with a fixed gutter size between the cells.
    """
    return math.floor((length + gutter_size) / (cell_min_size + gutter_size))


def separate_crops_into_cells(paintings, crops, crop_size=1080):
    gutter_size = crop_size  # The fixed size of the gutter between the grid cells
    min_cell_size = 2 * crop_size  # The minimum size of a grid cell

    paintings["n_cells_width"] = paintings["width"].apply(
        lambda x: max_cells(x, min_cell_size, gutter_size)
    )
    paintings["n_cells_height"] = paintings["height"].apply(
        lambda x: max_cells(x, min_cell_size, gutter_size)
    )

    def get_grid_cell(row):
        painting = paintings.loc[paintings["paintingName"] == row["paintingName"]]

        cell_width = (
            painting["width"] - (painting["n_cells_width"] - 1) * gutter_size
        ) / painting["n_cells_width"]
        cell_height = (
            painting["height"] - (painting["n_cells_height"] - 1) * gutter_size
        ) / painting["n_cells_height"]

        x_min, y_min, x_max, y_max = [int(x) for x in row["location"]]
        center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2

        # Determine the grid cell coordinates
        grid_x = math.floor(center_x / (cell_width + gutter_size))
        grid_y = math.floor(center_y / (cell_height + gutter_size))

        return (grid_x, grid_y)

    # Apply the grid cell identification function to each crop
    crops["grid_cell"] = crops.apply(get_grid_cell, axis=1)

    return crops


def count_classes(df):
    color_count = Counter()
    for marble_list in df["classes"]:
        color_count += Counter(marble_list)
    return color_count


def counter_diff(counter1, counter2):
    diff = 0
    for key in set(counter1.keys()).union(set(counter2.keys())):
        diff += abs(counter1[key] - counter2[key])
    return diff


def assign_train_val_sets(
    df, train_percentage, val_percentage, capacity_constraint_multiplier
):
    """
    Assign each crop to either the train or validation set, while trying to keep the number of crops of each color in each set as close as possible.
    capacity_constraint_multiplier: higher value means more weight on capacity constraint
    """
    total_crops = len(df.index)
    train_capacity = total_crops * train_percentage
    val_capacity = total_crops * val_percentage
    df["set"] = None

    for paintingName in df["paintingName"].unique():
        for grid_cell in df[df["paintingName"] == paintingName]["grid_cell"].unique():
            bucket_df = df[
                (df["paintingName"] == paintingName) & (df["grid_cell"] == grid_cell)
            ]
            bucket_color_count = count_classes(bucket_df)

            set_color_counts = [
                count_classes(df[df["set"] == "train"]),
                count_classes(df[df["set"] == "val"]),
            ]

            color_diffs = [
                counter_diff(set_color_counts[0], bucket_color_count),
                counter_diff(set_color_counts[1], bucket_color_count),
            ]

            # Increase the weight of color difference if the set exceeds capacity
            if len(df[df["set"] == "train"]) > train_capacity:
                color_diffs[0] *= capacity_constraint_multiplier
            if len(df[df["set"] == "val"]) > val_capacity:
                color_diffs[1] *= capacity_constraint_multiplier

            if color_diffs[0] <= color_diffs[1]:
                df.loc[bucket_df.index, "set"] = "train"
            else:
                df.loc[bucket_df.index, "set"] = "val"


def divide_train_val_sets(
    df, train_percentage=0.8, val_percentage=0.2, capacity_constraint_multiplier=3
):
    assign_train_val_sets(
        df, train_percentage, val_percentage, capacity_constraint_multiplier
    )
    return [df[df["set"] == "train"], df[df["set"] == "val"]]


def print_stats(train_files, validation_files, len_both):
    num_classes_train = count_classes(train_files)
    for key in num_classes_train:
        num_classes_train[key] /= len(train_files)
    # sort the dictionary by value
    num_classes_train = dict(
        sorted(num_classes_train.items(), key=lambda item: item[1], reverse=True)
    )

    num_classes_val = count_classes(validation_files)
    for key in num_classes_val:
        num_classes_val[key] /= len(validation_files)
    # sort the dictionary by value
    num_classes_val = dict(
        sorted(num_classes_val.items(), key=lambda item: item[1], reverse=True)
    )

    logging.info("Train and Validaton set class distribution:")
    logging.info("only in train set:")
    for key in num_classes_train:
        if key not in num_classes_val:
            logging.info(f"{key}: {num_classes_train[key]}")
    logging.info("only in validation set:")
    for key in num_classes_val:
        if key not in num_classes_train:
            logging.info(f"{key}: {num_classes_val[key]}")

    logging.info("in both sets:")
    distribution_df = pd.DataFrame(columns=["train", "valid"])
    for key in num_classes_train:
        if key in num_classes_val:
            distribution_df.loc[key] = [num_classes_train[key], num_classes_val[key]]
    logging.info(distribution_df)

    # Print the actual split percentage
    logging.info(
        f"Train set size: {len(train_files.index)} ({len(train_files.index) / len_both * 100}% of all crops excluding test painting crops)"
    )
    logging.info(
        f"Validation set size: {len(validation_files.index)} ({len(validation_files.index) / len_both * 100}% of all crops excluding test painting crops)"
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("location", help="Location of the dataset")
    parser.add_argument(
        "--output", "-o", help="Location to save the separated dataset", default=None
    )
    parser.add_argument(
        "--force", "-f", help="Force overwrite of existing files", action="store_true"
    )
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

    # paths to the files/directories
    CROPS_LOCATION_FILE = dataset_dir / "cropped" / "crops.txt"
    PAINTING_SIZES_FILE = dataset_dir / "painting_sizes.txt"
    CLASSES_FILE = dataset_dir / "classes.txt"
    OUT_YAML_FILE = output_dir / "dataset.yaml"
    TEST_PAINTING_FILE = dataset_dir / "test_painting.txt"
    LABELS_DIR = dataset_dir / "cropped" / "labels"
    IMAGES_DIR = dataset_dir / "cropped" / "images"
    TEST_IMAGES_DIR = dataset_dir / "images"
    TEST_LABELS_DIR = dataset_dir / "labels"
    OUTPUT_DIRS = {}

    # Create the output directories
    for split in ["train", "valid", "test"]:
        for folder in ["images", "labels"]:
            path = output_dir / split / folder

            if path.exists():
                if not args.force:
                    logging.error(
                        f"Output directory {path} already exists. Use -f to force overwrite."
                    )
                    exit()
                logging.warning(f"Output directory {path} already exists. Overwriting.")
                shutil.rmtree(path)

            os.makedirs(path, exist_ok=True)
            OUTPUT_DIRS[split + "_" + folder] = path

    # Read test painting file
    with open(TEST_PAINTING_FILE, "r") as file:
        test_painting = file.readline().strip()

    # Read the crops location file and the label files
    crops, max_crop_dimension = read_crops_location(CROPS_LOCATION_FILE)
    painting_classes, crops = read_label_files(LABELS_DIR, crops)
    paintings = read_painting_sizes(PAINTING_SIZES_FILE)

    # Get the ordered list of classes
    with open(CLASSES_FILE, "r") as file:
        classes = [line.strip() for line in file]

    # Separate the crops into cells
    crops = separate_crops_into_cells(paintings, crops, max_crop_dimension)

    # TODO Visualize the cells

    len_both = len(crops.index)  # used for printing stats

    # Separate the crops into train and validation sets
    train_files, validation_files = divide_train_val_sets(crops)

    # Print stats
    print_stats(train_files, validation_files, len_both)

    # TODO Visualize the train and validation sets

    move_files(
        train_files,
        LABELS_DIR,
        IMAGES_DIR,
        OUTPUT_DIRS["train_labels"],
        OUTPUT_DIRS["train_images"],
        "train set",
    )
    move_files(
        validation_files,
        LABELS_DIR,
        IMAGES_DIR,
        OUTPUT_DIRS["valid_labels"],
        OUTPUT_DIRS["valid_images"],
        "validation set",
    )

    # Copy the test image psb into the test set
    shutil.copy(
        TEST_IMAGES_DIR / f"{test_painting}.psb",
        OUTPUT_DIRS["test_images"] / f"{test_painting}.psb",
    )
    shutil.copy(
        TEST_LABELS_DIR / f"{test_painting}.txt",
        OUTPUT_DIRS["test_labels"] / f"{test_painting}.txt",
    )

    # save yaml file in the yolov5 format
    with open(OUT_YAML_FILE, "w") as file:
        file.write(f"train: {OUTPUT_DIRS['train_images'].relative_to(output_dir)}\n")
        file.write(f"val: {OUTPUT_DIRS['valid_images'].relative_to(output_dir)}\n")
        file.write(f"test: {OUTPUT_DIRS['test_images'].relative_to(output_dir)}\n")
        file.write(f"nc: {len(classes)}\n")
        file.write("names:\n")
        for c in classes:
            file.write(f"- {c}\n")


if __name__ == "__main__":
    main()
