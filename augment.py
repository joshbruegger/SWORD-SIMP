import os
import argparse
import shutil
import cv2
import numpy as np
import albumentations as A
from psd_tools import PSDImage


def generate_random_crop_coordinates(image_width, image_height, min_size=640, max_size=1024, factor=32):
    sizes = range(min_size, max_size + 1, factor)
    width, height = np.random.choice(sizes), np.random.choice(sizes)

    # Ensure crop dimensions do not exceed image dimensions
    width, height = min(width, image_width), min(height, image_height)

    x_min = np.random.randint(0, image_width - width)
    y_min = np.random.randint(0, image_height - height)
    x_max = x_min + width
    y_max = y_min + height

    return x_min, y_min, x_max, y_max


def get_augmentation(crop_coordinates, min_visibility):
    return A.Compose(
        [A.Crop(*crop_coordinates, always_apply=True)],
        bbox_params=A.BboxParams(
            format='yolo',
            min_visibility=min_visibility,
            label_fields=['labels']
        )
    )


def read_labels(labels_dir, filename):
    label_filename = os.path.splitext(filename)[0] + '.txt'
    with open(os.path.join(labels_dir, label_filename)) as f:
        labels_and_bboxes = [line.strip().split() for line in f]
        labels = [int(line[0]) for line in labels_and_bboxes]
        bboxes = [list(map(float, line[1:])) for line in labels_and_bboxes]
    return bboxes, labels


def save_crops_and_labels(crops, crop_bboxes, crop_labels, output_images_dir, output_labels_dir, filename):
    for i, (crop, boxes, labels) in enumerate(zip(crops, crop_bboxes, crop_labels)):  # Loop through crops
        # Save crop
        cv2.imwrite(os.path.join(output_images_dir,
                    f'{os.path.splitext(filename)[0]}_{i}.jpg'), crop)
        # Save labels
        with open(os.path.join(output_labels_dir, f'{os.path.splitext(filename)[0]}_{i}.txt'), 'w') as f:
            for box, label in zip(boxes, labels):
                f.write(' '.join(map(str, [label] + list(box))) + '\n')


def generate_crops(image, bboxes, labels, n_crops, min_crops, min_visibility):
    crops = []
    crop_bboxes = []
    crop_labels = []
    history = []

    while len(crops) < n_crops:
        crop_coordinates = generate_random_crop_coordinates(
            image.shape[1], image.shape[0])
        transform = get_augmentation(crop_coordinates, min_visibility)
        augmented = transform(image=image, bboxes=bboxes, labels=labels)
        if len(augmented['bboxes']) >= min_crops:
            crops.append(augmented['image'])
            crop_bboxes.append(augmented['bboxes'])
            crop_labels.append(augmented['labels'])
            history.append(crop_coordinates)

    return crops, crop_bboxes, crop_labels, history


def process_images(images_dir, labels_dir, output_images_dir, output_labels_dir, args, history_txt):
    for filename in os.listdir(images_dir):  # Loop through images
        print(f'Generating crops for {filename}')

        # Read image
        if filename.endswith('.psb') or filename.endswith('.psd'):
            psd = PSDImage.open(os.path.join(images_dir, filename))
            image = cv2.cvtColor(psd[0].numpy(), cv2.COLOR_RGB2BGR)
            image *= 255
        elif filename.endswith('.jpg') or filename.endswith('.png'):
            image = cv2.imread(os.path.join(images_dir, filename))
        else:
            print(f'Unsupported file format: {filename}')
            continue

        # Read labels
        bboxes, labels = read_labels(labels_dir, filename)

        # Generate crops
        crops, crop_bboxes, crop_labels, history = generate_crops(
            image, bboxes, labels, args.n_crops, args.min_crops, args.min_visibility)

        # Write history to txt file,  formatted as filename x_min y_min x_max y_max for each crop
        for crop in history:
            history_txt.write(filename + ' ' + ' '.join(map(str, crop)) + '\n')

        # print length of crops
        print("Generated " + str(len(crops)) + " crops")

        # Save crops and labels
        save_crops_and_labels(crops, crop_bboxes, crop_labels,
                              output_images_dir, output_labels_dir, filename)

        print("Saved crops for " + filename + " successfully")


def get_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='Working directory')
    parser.add_argument('n_crops', type=int, help='Number of crops per image')
    parser.add_argument('--min_crops', type=int, default=1,
                        help='Minimum number of bounding boxes per crop')
    parser.add_argument('--min_visibility', type=float, default=0.1,
                        help='Minimum visibility for bounding boxes')
    return parser.parse_args()


def main():
    args = get_args()

    # Get input and output directories
    images_dir = os.path.join(args.dir, 'images')
    labels_dir = os.path.join(args.dir, 'labels')
    output_images_dir = os.path.join(args.dir, 'augmented', 'images')
    output_labels_dir = os.path.join(args.dir, 'augmented', 'labels')

    # Check if augmented folder already exists
    augmented_folder = os.path.join(args.dir, 'augmented')
    if os.path.exists(augmented_folder):
        user_input = input(
            "The 'augmented' folder already exists. Do you want to delete it and proceed? (y/n): ")
        while user_input.lower() not in ['y', 'n']:
            user_input = input(
                "The 'augmented' folder already exists. Do you want to delete it and proceed? (y/n): ")
        if user_input.lower() == 'y':
            shutil.rmtree(augmented_folder)
        elif user_input.lower() == 'n':
            print("Exiting...")
            return

    # Create output directories
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    with open(os.path.join(args.dir, 'crops.txt'), 'w+') as crops_txt:
        process_images(images_dir, labels_dir, output_images_dir,
                       output_labels_dir, args, crops_txt)


if __name__ == '__main__':
    main()
