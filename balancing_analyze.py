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


def calculate_file_scores(
    c_hist, cs_in_file, label_files, files_to_discard, thres, class_files
):
    file_scores = []
    file_scores.extend(
        (
            f,
            sum(c_hist[c] - thres for c in cs_in_file[f] if c_hist[c] > thres),
            len([c for c in set(cs_in_file[f]) if c_hist[c] > thres]),
        )
        for f in label_files
        if f not in files_to_discard
    )
    return file_scores


def read_files(label_files):
    # Dictionary to keep track of the number of instances of each class
    class_counts = defaultdict(int)
    # Dictionary to keep track of the files where each class appears
    class_files = defaultdict(list)
    # Dictionary to keep track of the classes for each file
    file_classes = defaultdict(list)

    for file in tqdm(label_files, desc="Processing files"):
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                class_id = int(line.split()[0])
                class_counts[class_id] += 1
                class_files[class_id].append(file)
                file_classes[file].append(class_id)
    return class_counts, file_classes, class_files


def get_files_to_discard(label_files, c_hist, cs_in_file, class_files, thres):
    files_to_discard = set()
    file_scores = calculate_file_scores(
        c_hist,
        cs_in_file,
        label_files,
        files_to_discard,
        thres,
        class_files,
    )

    while file_scores:
        if all(c_hist[c] <= thres for c in c_hist.keys()):
            break

        file_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        file = file_scores.pop(0)[0]
        over_represented = [c for c in cs_in_file[file] if c_hist[c] > thres]
        unique = set(cs_in_file[file])

        if not over_represented or unique.difference(over_represented):
            continue

        imbal_kept = sum(abs(c_hist[c] - thres) for c in unique)
        imbal_removed = sum(
            abs(c_hist[c] - cs_in_file[file].count(c) - thres) for c in unique
        )

        if imbal_removed > imbal_kept:
            continue

        files_to_discard.add(file)

        for class_id in unique:
            c_hist[class_id] -= cs_in_file[file].count(class_id)

        file_scores = calculate_file_scores(
            c_hist,
            cs_in_file,
            label_files,
            files_to_discard,
            thres,
            class_files,
        )

    return files_to_discard


def main():
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

    class_counts, file_classes, class_files = read_files(label_files)

    # Sort the dictionary by count
    class_counts = {
        k: v
        for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True)
    }

    print(f"Class counts:\n{class_counts}")

    percentile = 35
    threshold = int(np.percentile(list(class_counts.values()), percentile))
    print(f"Threshold: {threshold} ({percentile} percentile)")

    plot_histogram(
        class_counts,
        "Class Histogram Before Discarding Files",
        "balancing_analyze_before_out.png",
        threshold=threshold,
    )

    files_to_discard = get_files_to_discard(
        label_files, class_counts, file_classes, class_files, threshold
    )

    num_discard = len(files_to_discard)
    num_before = len(label_files)
    num_after = num_before - num_discard
    percentage_remaining = num_after / num_before * 100

    print(f"Number of files to discard: {num_discard}")
    print(f"Number of files before discarding: {num_before}")
    print(
        f"Number of files after discarding: {num_after} ({percentage_remaining:.2f}% of original)"
    )

    # Sort the dictionary by count
    class_counts = {
        k: v
        for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True)
    }

    print(f"Class counts after discarding:\n{class_counts}")
    plot_histogram(
        class_counts,
        "Class Histogram After Discarding Files",
        "balancing_analyze_after_out.png",
        threshold=threshold,
    )

    with open("balancing_analyze_to_discard_out.txt", "w") as f:
        for file in files_to_discard:
            f.write(file + "\n")


if __name__ == "__main__":
    main()
