
import os
import argparse
import cv2
import numpy as np
import albumentations as A


def generate_random_crop_size(min_size=640, max_size=1024, factor=32):
    sizes = range(min_size, max_size+1, factor)
    return np.random.choice(sizes), np.random.choice(sizes)


def get_augmentation(crop_size, min_visibility):
    return A.Compose(
        [A.RandomCrop(height=crop_size[0], width=crop_size[1], p=1.0)],
        bbox_params=A.BboxParams(
            format='yolo',
            min_visibility=min_visibility,
            label_fields=['labels']
        )
    )


def generate_crops(image, bboxes, labels, n_crops, min_crops, min_visibility):
    crops = []
    crop_bboxes = []
    crop_labels = []

    while len(crops) < n_crops:
        crop_size = generate_random_crop_size()
        transform = get_augmentation(crop_size, min_visibility)
        augmented = transform(image=image, bboxes=bboxes, labels=labels)
        if len(augmented['bboxes']) >= min_crops:
            crops.append(augmented['image'])
            crop_bboxes.append(augmented['bboxes'])
            crop_labels.append(augmented['labels'])

    return crops, crop_bboxes, crop_labels


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='Working directory')
    parser.add_argument('n_crops', type=int, help='Number of crops per image')
    parser.add_argument('--min_crops', type=int, default=1,
                        help='Minimum number of bounding boxes per crop')
    parser.add_argument('--min_visibility', type=float, default=0.1,
                        help='Minimum visibility for bounding boxes')
    args = parser.parse_args()

    # Get input and output directories
    images_dir = os.path.join(args.dir, 'images')
    labels_dir = os.path.join(args.dir, 'labels')
    output_images_dir = os.path.join(args.dir, 'augmented', 'images')
    output_labels_dir = os.path.join(args.dir, 'augmented', 'labels')

    # Create output directories
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    for filename in os.listdir(images_dir):  # Loop through images
        print(f'Generating crops for {filename}')

        # Read image
        image = cv2.imread(os.path.join(images_dir, filename))

        # Read labels
        label_filename = os.path.splitext(filename)[0] + '.txt'
        with open(os.path.join(labels_dir, label_filename)) as f:
            labels_and_bboxes = [line.strip().split() for line in f]
            labels = [int(line[0]) for line in labels_and_bboxes]
            bboxes = [list(map(float, line[1:])) for line in labels_and_bboxes]

        # Generate crops
        crops, crop_bboxes, crop_labels = generate_crops(
            image, bboxes, labels, args.n_crops, args.min_crops, args.min_visibility)

        # print lenght of crops
        print("Generated " + str(len(crops)) + " crops")

        # Save crops and labels
        for i, (crop, boxes, labels) in enumerate(zip(crops, crop_bboxes, crop_labels)):  # Loop through crops
            # Save crop
            cv2.imwrite(os.path.join(output_images_dir,
                        f'{os.path.splitext(filename)[0]}_{i}.jpg'), crop)
            # Save labels
            with open(os.path.join(output_labels_dir, f'{os.path.splitext(filename)[0]}_{i}.txt'), 'w') as f:
                for box, label in zip(boxes, labels):
                    f.write(' '.join(map(str, [label] + list(box))) + '\n')

        print("Saved crops for " + filename + " successfully")


if __name__ == '__main__':
    main()
