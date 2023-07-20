import os
import glob
import argparse
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms


def plot_histogram(class_counts, title, filename):
    """Plot the class counts as a histogram and save the figure."""
    # Sort the dictionary by count
    class_counts = {
        k: v
        for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True)
    }
    print(f"Class counts:\n{class_counts}")

    plt.clf()
    plt.figure(figsize=(15, 10))

    plt.margins(x=0.01)

    bar_container = plt.bar(
        [str(k) for k in class_counts.keys()], class_counts.values()
    )

    plt.bar_label(bar_container, fmt="{:.0f}")

    plt.xlabel("Class ID")
    plt.ylabel("Count")
    plt.suptitle(title, fontweight="bold")
    value_sum = sum(class_counts.values())
    length = len(class_counts)
    plt.title(
        f"{value_sum} annotations accross {length} classes (on averrage {value_sum/length:.1f} annotations per class)"
    )
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()


def read_files(label_files):
    # Dictionary to keep track of the number of instances of each class
    class_counts = defaultdict(int)
    # Dictionary to keep track of the classes for each file
    file_classes = defaultdict(list)

    for file in tqdm(label_files, desc="Processing files"):
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                class_id = int(line.split()[0])
                class_counts[class_id] += 1
                file_classes[file].append(class_id)

    return class_counts, file_classes


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root_dir", type=str, help="path to the root directory of the dataset"
    )
    parser.add_argument(
        "reduction_percentage", type=int, help="target percentage  of the new size"
    )
    parser.add_argument(
        "--not-under",
        type=int,
        default=None,
        help="don't discard files with classes with less than this number of instances",
    )

    args = parser.parse_args()
    root_dir = args.root_dir
    reduction_percentage = args.reduction_percentage
    not_under = args.not_under

    # Get the list of all label files in each directory
    labels_path = os.path.join(root_dir, "valid", "labels", "*.txt")
    label_files = glob.glob(labels_path)
    print(f"Number of label files: {len(label_files)}")

    class_counts, file_classes = read_files(label_files)

    print(f"Number of classes:\n{len(class_counts)}")

    plot_histogram(
        class_counts,
        "Class Histogram Before Discarding Files",
        "reduce_val_size_before_out.png",
    )

    import random

    target_discard = reduction_percentage * len(label_files) / 100
    num_before = len(label_files)

    files_to_discard = set()
    while len(files_to_discard) < target_discard:
        file = random.choice(label_files)
        if not_under and any(class_counts[c] < not_under for c in file_classes[file]):
            continue
        label_files.remove(file)
        files_to_discard.add(file)
        for class_id in file_classes[file]:
            class_counts[class_id] -= 1

    num_discard = len(files_to_discard)
    num_after = num_before - num_discard
    percentage_remaining = num_after / num_before * 100

    print(f"Number of files to discard: {num_discard}")
    print(f"Number of files before discarding: {num_before}")
    print(
        f"Number of files after discarding: {num_after} ({percentage_remaining:.2f}% of original)"
    )

    plot_histogram(
        class_counts,
        "Class Histogram After Discarding Files",
        "reduce_val_size_after_out.png",
    )

    with open("reduce_val_size_to_discard_out.txt", "w") as f:
        for file in files_to_discard:
            f.write(file + "\n")


if __name__ == "__main__":
    main()
