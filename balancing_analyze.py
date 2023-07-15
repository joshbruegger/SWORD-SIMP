import os
import glob
import argparse
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def plot_histogram(class_counts, title, filename, percentile_50=None, percentile_25=None, percentile_35=None, percentile_40=None):
    """Plot the class counts as a histogram and save the figure."""
    plt.bar(class_counts.keys(), class_counts.values())

    # Make every class ID visible on the x-axis
    plt.xticks(list(class_counts.keys()))
    # Add text labels for each bar
    for x, y in class_counts.items():
        plt.text(x, y, str(y))

    # Plot percentile lines if they are provided
    if percentile_25 is not None:
        plt.axhline(y=percentile_25, color='g', linestyle='-', label='25th Percentile')
    if percentile_35 is not None:
        plt.axhline(y=percentile_35, color='b', linestyle='-', label='35th Percentile')
    if percentile_40 is not None:
        plt.axhline(y=percentile_40, color='y', linestyle='-', label='40th Percentile')
    if percentile_50 is not None:
        plt.axhline(y=percentile_50, color='r', linestyle='-', label='50th Percentile')

    # Add a total count label to the top right corner right above the plot
    plt.text(0.99, 0.99, f'Total Count: {sum(class_counts.values())}', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)

    plt.legend()
    plt.xlabel('Class ID')
    plt.ylabel('Count')
    plt.title(title)
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
    class_counts = {k: v for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True)}
    print('Train class counts:')
    print(class_counts)

    # Calculate the 25, 35, 40, 50 percentiles
    percentile_25 = int(np.percentile(list(class_counts.values()), 25))
    percentile_35 = int(np.percentile(list(class_counts.values()), 35))
    percentile_40 = int(np.percentile(list(class_counts.values()), 40))
    percentile_50 = int(np.percentile(list(class_counts.values()), 50))
    print(f'25th percentile: {percentile_25}')
    print(f'35th percentile: {percentile_35}')
    print(f'40th percentile: {percentile_40}')

    #Set the threshold to be the 25th percentile
    threshold = percentile_35
    print(f'Threshold: {threshold} (35th percentile)')

    plot_histogram(class_counts, 'Class Counts Before Discarding Files', 'balancing_before_out.png', percentile_50=percentile_50, percentile_25=percentile_25, percentile_35=percentile_35, percentile_40=percentile_40)

    # Compute the over-representation score of each file
    file_scores = {file: sum(class_counts[c_id] - threshold for c_id in file_classes[file] if class_counts[c_id] > threshold) for file in label_files}

    # Sort the files by their scores in descending order
    sorted_files = sorted(file_scores.items(), key=lambda item: item[1], reverse=True)

    # Determine the files to discard
    files_to_discard = set()
    for file, score in sorted_files: # For each file, starting from the highest score
        over_represented_classes = [c_id for c_id in file_classes[file] if class_counts[c_id] > threshold]
        if over_represented_classes: # If the file contains over-represented classes
            files_to_discard.add(file) # Mark the file to be discarded
            # Reduce the count for each over-represented class instance in the file
            for class_id in over_represented_classes:
                class_counts[class_id] -= file_classes[file].count(class_id)
                if class_counts[class_id] <= threshold:
                    break
        if all(class_counts[c_id] <= threshold for c_id in class_counts.keys()): # If all classes are now balanced
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
    plot_histogram(class_counts, 'Class Counts After Discarding Files', 'balancing_after_out.png')
    # copy the dictionary
    # train_class_counts_after = class_counts.copy()
    # # remove the counts for the discarded files
    # for file in files_to_discard:
    #     for class_id in file_classes[file]:
    #         train_class_counts_after[class_id] -= 1
    # print(train_class_counts_after)
    # Plot the class counts as a histogram

    # Save the list of files to discard
    with open('balancing_to_discard_out.txt', 'w') as f:
        for file in files_to_discard:
            f.write(file + '\n')
