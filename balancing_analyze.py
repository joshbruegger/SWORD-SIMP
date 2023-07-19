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


def score_tuple(c_hist, cs_in_file, unique, thres, f):
    return (
        sum(c_hist[c] - thres for c in cs_in_file[f] if c_hist[c] > thres),
        len([c for c in unique if c_hist[c] > thres]),
    )


def get_file_scores(c_hist, cs_in_file, label_files, files_to_discard, thres):
    file_scores = defaultdict(tuple[int, int])
    for f in label_files:
        if f not in files_to_discard:
            unique = set(cs_in_file[f])
            file_scores[f] = score_tuple(c_hist, cs_in_file, unique, thres, f)

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


def get_discarded(label_files, c_hist, cs_in_file, thres, class_files):
    to_discard = set()
    file_scores = get_file_scores(c_hist, cs_in_file, label_files, to_discard, thres)

    t = tqdm(desc="Discarding files", total=len(file_scores))
    while file_scores:
        t.update(1)
        if all(c_hist[c] <= thres for c in c_hist.keys()):
            break
        file, (imbal, u) = max(file_scores.items(), key=lambda x: (x[1][0], x[1][1]))
        file_scores.pop(file)
        over_represented = [c for c in cs_in_file[file] if c_hist[c] > thres]
        unique = set(cs_in_file[file])

        if not over_represented or unique.difference(over_represented):
            continue

        imbal_kept, imbal_removed = 0, 0
        for c in unique:
            imbal_kept += abs(c_hist[c] - thres)
            imbal_removed += abs(c_hist[c] - cs_in_file[file].count(c) - thres)

        if imbal_removed > imbal_kept:
            continue

        to_discard.add(file)

        for class_id in unique:
            c_hist[class_id] -= cs_in_file[file].count(class_id)

        for class_id in unique:
            for f in class_files[class_id]:
                if f in file_scores:
                    u = set(cs_in_file[f])
                    file_scores[f] = score_tuple(c_hist, cs_in_file, u, thres, f)

    return to_discard


def print_info(label_files, threshold, files_to_discard, class_counts):
    num_discard = len(files_to_discard)
    num_before = len(label_files)
    num_after = num_before - num_discard
    percentage_remaining = num_after / num_before * 100

    print(f"Number of files to discard: {num_discard}")
    print(f"Number of files before discarding: {num_before}")
    print(
        f"Number of files after discarding: {num_after} ({percentage_remaining:.2f}% of original)"
    )

    # calculate imbalance score
    imbalance_score = 0
    for _, count in class_counts.items():
        imbalance_score += abs(count - threshold)
    print(f"Imbalance score: {imbalance_score}")

    plot_histogram(
        class_counts,
        "Class Histogram After Discarding Files",
        f"balancing_analyze_after_out.png",
        threshold=threshold,
    )


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
    labels = glob.glob(labels_path)
    print(f"Number of label files: {len(labels)}")

    c_hist, cs_in_file, class_files = read_files(labels)

    percentile = 35
    thres = int(np.percentile(list(c_hist.values()), percentile))
    print(f"Threshold: {thres} ({percentile} percentile)")

    plot_histogram(
        c_hist,
        "Class Histogram Before Discarding Files",
        "balancing_analyze_before_out.png",
        threshold=thres,
    )

    discarded = get_discarded(
        labels,
        c_hist,
        cs_in_file,
        thres,
        class_files,
    )

    print_info(labels, thres, discarded, c_hist)

    with open(f"balancing_analyze_to_discard_out.txt", "w") as f:
        for file in discarded:
            f.write(file + "\n")


if __name__ == "__main__":
    main()
