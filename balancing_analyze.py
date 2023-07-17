import os
import glob
import argparse
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms


def plot_histogram(class_counts, title, filename, threshold=None):
    """Plot the class counts as a histogram and save the figure."""
    plt.clf()
    plt.figure(figsize=(15, 10))

    plt.margins(x=0.01)

    bar_container = plt.bar(
        [str(k) for k in class_counts.keys()], class_counts.values()
    )

    plt.bar_label(bar_container, fmt="{:.0f}")

    if threshold is not None:
        plt.yticks(np.append(plt.yticks()[0], threshold))
        plt.axhline(y=threshold, color="r", linestyle="-")
        # Add text Threshold to the right of the line
        plt.text(
            1,
            threshold,
            " Threshold",
            horizontalalignment="left",
            verticalalignment="center",
            transform=transforms.blended_transform_factory(
                plt.gca().transAxes, plt.gca().transData
            ),
            color="r",
        )

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


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root_dir", type=str, help="path to the root directory of the dataset"
    )

    args = parser.parse_args()
    root_dir = args.root_dir

    # Get the list of all label files in each directory
    labels_path = os.path.join(root_dir, "train", "labels", "*.txt")
    label_files = glob.glob(labels_path)
    print(f"Number of label files: {len(label_files)}")

    class_counts = defaultdict(int)
    # Dictionary to keep track of the number of instances of each class
    class_files = defaultdict(
        list
    )  # Dictionary to keep track of the files where each class appears
    file_classes = defaultdict(
        list
    )  # Dictionary to keep track of the classes for each file

    for file in tqdm(label_files):
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                class_id = int(line.split()[0])
                class_counts[class_id] += 1
                class_files[class_id].append(file)
                file_classes[file].append(class_id)

    # Sort the dictionary by count
    class_counts = {
        k: v
        for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True)
    }
    print("Train class counts:")
    print(class_counts)

    # Calculate the 25, 35, 40, 50 percentiles
    percentile_25 = int(np.percentile(list(class_counts.values()), 25))
    percentile_35 = int(np.percentile(list(class_counts.values()), 35))
    percentile_40 = int(np.percentile(list(class_counts.values()), 40))
    percentile_50 = int(np.percentile(list(class_counts.values()), 50))
    print(f"25th percentile: {percentile_25}")
    print(f"35th percentile: {percentile_35}")
    print(f"40th percentile: {percentile_40}")
    print(f"50th percentile: {percentile_50}")

    threshold = percentile_35
    print(f"Threshold: {threshold}")

    plot_histogram(
        class_counts,
        "Class Histogram Before Discarding Files",
        "balancing_analyze_before_out.png",
        threshold=threshold,
    )

    # Compute the over-representation score of each file
    file_scores = {
        file: sum(
            class_counts[c_id] - threshold
            for c_id in file_classes[file]
            if class_counts[c_id] > threshold
        )
        for file in label_files
    }

    # Sort the files by their scores in descending order
    sorted_files = sorted(file_scores.items(), key=lambda item: item[1], reverse=True)

    # Determine the files to discard
    files_to_discard = set()
    for file, score in sorted_files:  # For each file, starting from the highest score
        over_represented_classes = [
            c for c in file_classes[file] if class_counts[c] > threshold
        ]
        unique_classes = set(file_classes[file])
        # If the file contains over-represented classes and no under-represented classes, discard the file
        if over_represented_classes and not unique_classes.difference(
            over_represented_classes
        ):
            # Calculate the inbalance score before and after discarding the file
            file_inbalance_score_before = sum(
                abs(class_counts[c] - threshold) for c in unique_classes
            )
            file_inbalance_score_after = sum(
                abs(class_counts[c] - file_classes[file].count(c) - threshold)
                for c in unique_classes
            )
            # If the inbalance score is worse after discarding the file, don't discard it
            if file_inbalance_score_after > file_inbalance_score_before:
                continue

            files_to_discard.add(file)  # Mark the file to be discarded
            # Reduce the count for each over-represented class instance in the file
            for class_id in unique_classes:
                class_counts[class_id] -= file_classes[file].count(class_id)

            # After discarding a file, check if all classes are now balanced. If so, stop
            if all(
                class_counts[c] <= threshold for c in class_counts.keys()
            ):  # If all classes are now balanced
                break

    num_discard = len(files_to_discard)
    num_before = len(label_files)
    num_after = num_before - num_discard
    percentage_remaining = num_after / num_before * 100

    print(f"Number of files to discard: {num_discard}")
    print(f"Number of files before discarding: {num_before}")
    print(
        f"Number of files after discarding: {num_after} ({percentage_remaining:.2f}% of original)"
    )
    print("Class counts after discarding:")
    print(class_counts)
    plot_histogram(
        class_counts,
        "Class Histogram After Discarding Files",
        "balancing_analyze_after_out.png",
    )
    # copy the dictionary
    # train_class_counts_after = class_counts.copy()
    # # remove the counts for the discarded files
    # for file in files_to_discard:
    #     for class_id in file_classes[file]:
    #         train_class_counts_after[class_id] -= 1
    # print(train_class_counts_after)
    # Plot the class counts as a histogram

    # Save the list of files to discard
    with open("balancing_analyze_to_discard_out.txt", "w") as f:
        for file in files_to_discard:
            f.write(file + "\n")
