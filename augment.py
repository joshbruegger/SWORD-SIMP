import os
import argparse
import shutil
import cv2
import numpy as np
import albumentations as A
from psd_tools import PSDImage


class ImageAugmenter:
    def __init__(self, dir, n_crops, min_crops=1, min_visibility=0.1, force=False):
        self.dir = dir
        self.n_crops = n_crops
        self.min_crops = min_crops
        self.min_visibility = min_visibility

        self.images_dir = os.path.join(self.dir, 'images')
        self.labels_dir = os.path.join(self.dir, 'labels')
        self.output_images_dir = os.path.join(self.dir, 'augmented', 'images')
        self.output_labels_dir = os.path.join(self.dir, 'augmented', 'labels')
        self.augmented_folder = os.path.join(self.dir, 'augmented')

        # Ask user if they want to delete the augmented folder, if force is not set to true
        if os.path.exists(self.augmented_folder):
            if force:
                shutil.rmtree(self.augmented_folder)
            else:
                while True:
                    print('Augmented folder already exists, delete it? (y/n)')
                    if input().lower() == 'y':
                        shutil.rmtree(self.augmented_folder)
                        break
                    elif input().lower() == 'n':
                        exit()

        os.makedirs(self.output_images_dir, exist_ok=True)
        os.makedirs(self.output_labels_dir, exist_ok=True)

    @staticmethod
    def generate_random_crop_coordinates(image_width, image_height, min_size=640, max_size=1024, factor=32):
        sizes = range(min_size, max_size + 1, factor)
        width, height = np.random.choice(sizes), np.random.choice(sizes)

        width, height = min(width, image_width), min(height, image_height)

        x_min = np.random.randint(0, image_width - width)
        y_min = np.random.randint(0, image_height - height)
        x_max = x_min + width
        y_max = y_min + height

        return x_min, y_min, x_max, y_max

    def get_augmentation(self, crop_coordinates):
        return A.Compose(
            [A.Crop(*crop_coordinates, always_apply=True)],
            bbox_params=A.BboxParams(
                format='yolo',
                min_visibility=self.min_visibility,
                label_fields=['labels']
            )
        )

    def read_labels(self, filename):
        label_filename = os.path.splitext(filename)[0] + '.txt'
        with open(os.path.join(self.labels_dir, label_filename)) as f:
            labels_and_bboxes = [line.strip().split() for line in f]
            labels = [int(line[0]) for line in labels_and_bboxes]
            bboxes = [list(map(float, line[1:])) for line in labels_and_bboxes]
        return bboxes, labels

    def save_crops_and_labels(self, crops, crop_bboxes, crop_labels, filename):
        for i, (crop, boxes, labels) in enumerate(zip(crops, crop_bboxes, crop_labels)):  # Loop through crops
            cv2.imwrite(os.path.join(self.output_images_dir,
                        f'{os.path.splitext(filename)[0]}_{i}.jpg'), crop)

            with open(os.path.join(self.output_labels_dir, f'{os.path.splitext(filename)[0]}_{i}.txt'), 'w') as f:
                for box, label in zip(boxes, labels):
                    f.write(' '.join(map(str, [label] + list(box))) + '\n')

    def generate_crops(self, image, bboxes, labels, history, filename):
        crops = []
        crop_bboxes = []
        crop_labels = []

        while len(crops) < self.n_crops:
            print('trying to generate crop')
            crop_coordinates = self.generate_random_crop_coordinates(
                image.shape[1], image.shape[0])
            transform = self.get_augmentation(crop_coordinates)
            augmented = transform(image=image, bboxes=bboxes, labels=labels)
            if len(augmented['bboxes']) >= self.min_crops:
                print('crop generated')
                crops.append(augmented['image'])
                crop_bboxes.append(augmented['bboxes'])
                crop_labels.append(augmented['labels'])
                history.append([filename] + list(crop_coordinates))
                print(history)

        return crops, crop_bboxes, crop_labels

    def process_images(self):
        history = []

        for filename in os.listdir(self.images_dir):
            print(f'Generating crops for {filename}')

            if filename.endswith('.psb') or filename.endswith('.psd'):
                psd = PSDImage.open(os.path.join(self.images_dir, filename))
                image = cv2.cvtColor(psd[0].numpy(), cv2.COLOR_RGB2BGR)
                image *= 255
            elif filename.endswith('.jpg') or filename.endswith('.png'):
                image = cv2.imread(os.path.join(self.images_dir, filename))
            else:
                print(f'Unsupported file format: {filename}')
                continue

            bboxes, labels = self.read_labels(filename)

            crops, crop_bboxes, crop_labels = self.generate_crops(
                image, bboxes, labels, history, filename)

            print("Generated " + str(len(crops)) + " crops")

            self.save_crops_and_labels(
                crops, crop_bboxes, crop_labels, filename)

            print("Saved crops for " + filename + " successfully")

        with open(os.path.join(self.augmented_folder, 'crops.txt'), 'a') as crops_txt:
            print("Writing crop history to crops.txt")
            for crop in history:
                crops_txt.write(' '.join(map(str, crop)) + '\n')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='Working directory')
    parser.add_argument('n_crops', type=int, help='Number of crops per image')
    parser.add_argument('--min_crops', type=int, default=1,
                        help='Minimum number of bounding boxes per crop')
    parser.add_argument('--min_visibility', type=float, default=0.1,
                        help='Minimum visibility for bounding boxes')
    parser.add_argument('--force', action='store_true',
                        help="Force the removal of the existing 'augmented' directory if it exists")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    augmenter = ImageAugmenter(
        args.dir, args.n_crops, args.min_crops, args.min_visibility, args.force)

    augmenter.process_images()
