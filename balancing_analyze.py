import os
import glob
import argparse
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def plot_histogram(class_counts, title, filename, median=None):
    """Plot the class counts as a histogram and save the figure."""
    # Sort the class counts in descending order
    class_counts = {k: v for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True)}
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel('Class ID')
    plt.ylabel('Count')
    # Plot median line if provided
    if median is not None:
        plt.axhline(y=median, color='r', linestyle='-')
    plt.title(title)
    # Make every class ID visible on the x-axis
    plt.xticks(list(class_counts.keys()))
    # Add text labels for each bar
    for x, y in class_counts.items():
        plt.text(x, y, str(y))
    # Add a total count label
    plt.text(0.5, 0.95, f'Total: {sum(class_counts.values())}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.savefig(filename)
    plt.clf()

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type=str, help='path to the root directory of the dataset')

    args = parser.parse_args()
    root_dir = args.root_dir

    # Get the list of all label files in each directory
    labels_path = os.path.join(root_dir, 'train', 'labels', '*.txt')
    label_files = glob.glob(labels_path)
    print(f'Number of label files: {len(label_files)}')

    class_counts = defaultdict(int) # Dictionary to keep track of the number of instances of each class
    class_files = defaultdict(list) # Dictionary to keep track of the files where each class appears
    file_classes = defaultdict(list)  # Dictionary to keep track of the classes for each file

    for file in tqdm(label_files):
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                class_id = int(line.split()[0])
                class_counts[class_id] += 1
                class_files[class_id].append(file)
                file_classes[file].append(class_id) 

    #Sort the dictionary by count
    # class_counts = dict(sorted(class_counts.items(), key=lambda item: item[1]))
    class_counts = {k: v for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True)}
    print('Train class counts:')
    print(class_counts)

    median_count = int(np.median(list(class_counts.values())))
    plot_histogram(class_counts, 'Class Counts Before Discarding Files', 'balancing_before.png', median=median_count)

    # Determine the files to discard
    files_to_discard = set()

    for class_id, count in class_counts.items(): # For each class
        if count > median_count: # If the count is greater than the median
            class_specific_files = list(class_files[class_id]) # Get a list of all files where this class appears
            for file in class_specific_files: # Iterate through the files where this class appears
                # Check if the file contains only classes that are over the median count
                if all(class_counts[c_id] > median_count for c_id in file_classes[file]):
                    if file not in files_to_discard:
                        files_to_discard.add(file) # Mark the file to be discarded
                        # Reduce the count for each class instance of the class in the file
                        class_counts[class_id] -= file_classes[file].count(class_id)
                        if class_counts[class_id] <= median_count:
                            break
    
                            
    num_discard = len(files_to_discard)
    num_before = len(label_files)
    num_after = num_before - num_discard
    percentage_remaining = num_after / num_before * 100

    print(f'Number of files to discard: {num_discard}')
    print(f'Number of files before discarding: {num_before}')
    print(f'Number of files after discarding: {num_after} ({percentage_remaining:.2f}% of original)')
    print('Class counts after discarding:')
    print(class_counts)
    plot_histogram(class_counts, 'Class Counts After Discarding Files', 'balancing_after.png')
    # copy the dictionary
    # train_class_counts_after = class_counts.copy()
    # # remove the counts for the discarded files
    # for file in files_to_discard:
    #     for class_id in file_classes[file]:
    #         train_class_counts_after[class_id] -= 1
    # print(train_class_counts_after)
    # Plot the class counts as a histogram

    # Save the list of files to discard
    with open('balancing_to_discard.txt', 'w') as f:
        for file in files_to_discard:
            f.write(file + '\n')
