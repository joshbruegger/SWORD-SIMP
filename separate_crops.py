import os
from collections import defaultdict
from tqdm import tqdm
import argparse
import shutil
from sklearn.model_selection import train_test_split

def read_label_files(folder):
    classes = defaultdict(lambda: defaultdict(int))
    txt_files = [
        f
        for f in os.listdir(folder)
        if f.endswith(".txt") and not f.endswith("_classes.txt")
    ]

    for i, filename in enumerate(tqdm(txt_files, desc="Reading files")):
        with open(os.path.join(folder, filename), "r") as file:
            filename = filename.split(".")[0]
            print(f"Reading file {i+1}/{len(txt_files)}: {filename}")
            for line in file:
                class_name = line.strip().split()[0]
                classes[class_name][filename] += 1
    return classes

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("location", help="Location of the dataset")
    parser.add_argument("--output", '-o', help="Location to save the separated dataset", default="None")
    parser.add_argument('--force', '-f', help="Force overwrite of existing files", action='store_true')
    args = parser.parse_args()

    # PRETTY PRINT WELCOME MESSAGE & ARGUMENTS.
    padding = 140
    print("\n\n")
    print(" SWORD-SIMP Dataset Separator ".center(padding, "8"))
    print(f" Dataset location: {args.location} ".center(padding))
    print(f" Output location: {args.output} ".center(padding))
    print(f" Force overwrite: {args.force} ".center(padding))
    print("".center(padding, "8"))
    print("\n\n")

    output_dir = args.output
    if output_dir == "None":
        output_dir = os.path.join(args.location, "separated")

    # Check if the dataset has already been separated
    if os.path.exists(output_dir):
        if args.force:
            print("Overwriting existing files")
        else:
            print("Dataset has already been separated. Use the -f flag to overwrite existing files.")
            exit()

    # Move the classes.txt file to the output folder
    shutil.move(os.path.join(args.location, "classes.txt"), os.path.join(output_dir, "classes.txt"))

    painting_classes = read_label_files(os.path.join(args.location, "labels"))
    most_shared = most_shared_classes(painting_classes)
    print(most_shared)
    print(f"Painting that shares the most classes with others: {most_shared[0][0]} ({most_shared[0][1]} classes)")

    train_images_dir = os.path.join(output_dir, "train", "images")
    train_labels_dir = os.path.join(output_dir, "train", "labels")
    val_images_dir = os.path.join(output_dir, "val", "images")
    val_labels_dir = os.path.join(output_dir, "val", "labels")
    test_images_dir = os.path.join(output_dir, "test", "images")
    test_labels_dir = os.path.join(output_dir, "test", "labels")

    # Create the folders if they don't exist
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(test_labels_dir, exist_ok=True)

    # Moving the crops from the painting that shares the most classes with others to the test folder
    for filename in tqdm(os.listdir(os.path.join(args.location, "cropped", "images")), desc="Moving test images for test set"):
        if most_shared[0][0] in filename:
            shutil.move(os.path.join(args.location, 'cropped', 'images', filename), os.path.join(test_images_dir, filename))
    for filename in tqdm(os.listdir(os.path.join(args.location, "cropped", "labels")), desc="Moving labels for test set"):
        if most_shared[0][0] in filename:
            shutil.move(os.path.join(args.location, 'cropped', 'labels', filename), os.path.join(test_labels_dir, filename))

    # Make a list of all classes and paintings
    all_classes = []
    all_files = []
    label_files = [f for f in os.listdir(os.path.join(args.location, "cropped", "labels")) if f.endswith(".txt") and not f.endswith("_classes.txt") and most_shared[0][0] not in f]

    for label_file in tqdm(label_files, desc="Creating list of classes"):
        with open(os.path.join(args.location, "cropped", "labels", label_file), "r") as file:
            for line in file:
                class_name = line.strip().split()[0]
                all_classes.append(class_name)
                all_files.append(label_file)

    # Use train_test_split to create training and validation sets
    # Try stratifying by class, if that doesn't work, don't stratify
    try:
        train_files, validation_files, _, _ = train_test_split(all_files, all_classes, test_size=0.2, stratify=all_classes, random_state=420)
    except ValueError:
        train_files, validation_files, _, _ = train_test_split(all_files, all_classes, test_size=0.2, random_state=420)

    
    # analyze_distribution(train_files, validation_files, args) #! Uncomment this line to print the distribution of classes in the training and validation sets. Takes a long time to run.

    # Then move the images/labels to their respective folders based on the split
    for filename in tqdm(os.listdir(os.path.join(args.location, "cropped", "images")), desc="Moving images for training and validation sets"):
        if filename.replace(".jpg", ".txt") in train_files:
            shutil.move(os.path.join(args.location, 'cropped', 'images', filename), os.path.join(train_images_dir, filename))
        elif filename.replace(".jpg", ".txt") in validation_files:
            shutil.move(os.path.join(args.location, 'cropped', 'images', filename), os.path.join(val_images_dir, filename))

    for filename in tqdm(os.listdir(os.path.join(args.location, "cropped", "labels")), desc="Moving labels for training and validation sets"):
        if filename in train_files:
            shutil.move(os.path.join(args.location, 'cropped', 'labels', filename), os.path.join(train_labels_dir, filename))
        elif filename in validation_files:
            shutil.move(os.path.join(args.location, 'cropped', 'labels', filename), os.path.join(val_labels_dir, filename))

    # Make a classes.txt file containing all the classes in the dataset line by line
    with open(os.path.join(output_dir, "classes.txt"), "w") as file:
        for class_name in painting_classes:
            file.write(class_name + "\n")

if __name__ == "__main__":
    main()
