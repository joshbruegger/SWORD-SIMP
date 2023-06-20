import os
from collections import defaultdict
from tqdm import tqdm
import argparse
from natsort import os_sorted
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

# Set up logger
logging.basicConfig(level=logging.INFO)

def read_label_files(folder: Path):
    """Read labels from files and return a dictionary with classes and counts per file."""
    txt_files = [f for f in folder.iterdir() if f.suffix == ".txt" and "_classes" not in f.stem]
    
    classes = defaultdict(lambda: defaultdict(int))
    for file in tqdm(txt_files, desc="Reading files"):
        with file.open() as f:
            for line in f:
                class_name = line.strip().split()[0]
                classes[class_name][file.stem] += 1

    return classes, txt_files

def most_shared_classes(classes):
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
    for file in tqdm(files["file"], desc=f"Moving {len(files)} files to {dest_folder}"):
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
    
    painting_classes, label_files = read_label_files(dataset_dir / "cropped" / "labels")
    most_shared = most_shared_classes(painting_classes)
    print(f"Painting that shares the most classes with others: {most_shared[0][0]} ({most_shared[0][1]} classes)")

    train_files, validation_files, test_files = split_train_valid_test_files(label_files, most_shared[0][0])

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
