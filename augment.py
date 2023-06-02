import argparse
import cv2
import numpy as np
import albumentations as A
import shutil
from pathlib import Path
from psd_tools import PSDImage


class ImageAugmenter:
    def __init__(self, dir, n_crops, min_crops=1, min_visibility=0.1, force=False, crop_file=None):
        self.dir = Path(dir)
        self.n_crops = n_crops
        self.min_crops = min_crops
        self.min_visibility = min_visibility
        self.augmented_folder = self.dir / 'augmented'
        self.images_dir = self.dir / 'images'
        self.labels_dir = self.dir / 'labels'
        self.output_images_dir = self.augmented_folder / 'images'
        self.output_labels_dir = self.augmented_folder / 'labels'
        self.crop_file = crop_file
        if self.crop_file:
            # Recover crop coordinates from file for images that have already been augmented
            with open(self.crop_file, 'r') as f:
                self.to_regenerate = {}
                for line in f:
                    line = line.strip().split()
                    if line[0] not in self.to_regenerate:
                        self.to_regenerate[line[0]] = []
                    self.to_regenerate[line[0]].append(
                        list(map(int, line[1:])))

        if self.augmented_folder.exists() and not force:
            while True:
                print('Augmented folder already exists, delete it? (y/n)')
                user_input = input().lower()
                if user_input == 'y':
                    shutil.rmtree(self.augmented_folder)
                    break
                elif user_input == 'n':
                    exit()

        self.output_images_dir.mkdir(parents=True, exist_ok=True)
        self.output_labels_dir.mkdir(parents=True, exist_ok=True)

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
        label_filename = (self.labels_dir / filename.stem).with_suffix('.txt')
        with open(label_filename, 'r') as f:
            labels_and_bboxes = [line.strip().split() for line in f]
            labels = [int(line[0]) for line in labels_and_bboxes]
            bboxes = [list(map(float, line[1:])) for line in labels_and_bboxes]
        return bboxes, labels

    def save_crops_and_labels(self, crops, crop_bboxes, crop_labels, filename):
        for i, (crop, boxes, labels) in enumerate(zip(crops, crop_bboxes, crop_labels)):
            cv2.imwrite(str(self.output_images_dir /
                        f'{filename.stem}_{i}.jpg'), crop)
            with open(self.output_labels_dir / f'{filename.stem}_{i}.txt', 'w') as f:
                for box, label in zip(boxes, labels):
                    f.write(' '.join(map(str, [label] + list(box))) + '\n')

    def generate_crops(self, image, bboxes, labels, history, filename):
        print(f'Generating crops for {filename.stem}')
        crops, crop_bboxes, crop_labels = [], [], []

        if self.crop_file:
            print("regenerating crops for", filename.stem)
            if filename.stem in self.to_regenerate:
                for crop_coordinates in self.to_regenerate[filename.stem]:
                    transform = self.get_augmentation(crop_coordinates)
                    augmented = transform(
                        image=image, bboxes=bboxes, labels=labels)
                    crops.append(augmented['image'])
                    crop_bboxes.append(augmented['bboxes'])
                    crop_labels.append(augmented['labels'])
                    history.append([filename.stem] + crop_coordinates)
            else:
                print("fatal: no crop coordinates found for image", filename.stem)
                exit()

        while len(crops) < self.n_crops:
            crop_coordinates = self.generate_random_crop_coordinates(
                image.shape[1], image.shape[0])
            transform = self.get_augmentation(crop_coordinates)
            augmented = transform(image=image, bboxes=bboxes, labels=labels)
            if len(augmented['bboxes']) >= self.min_crops:
                crops.append(augmented['image'])
                crop_bboxes.append(augmented['bboxes'])
                crop_labels.append(augmented['labels'])
                history.append([filename.stem] + list(crop_coordinates))
        return crops, crop_bboxes, crop_labels

    def process_images(self):
        history = []
        for filename in self.images_dir.iterdir():
            if filename.suffix not in ['.psb', '.psd', '.jpg', '.png']:
                print(f'Unsupported file format: {filename}')
                continue

            bboxes, labels = self.read_labels(filename)
            image = cv2.imread(str(filename)) if filename.suffix in ['.jpg', '.png'] else \
                cv2.cvtColor(PSDImage.open(filename)[
                             0].numpy(), cv2.COLOR_RGB2BGR) * 255

            crops, crop_bboxes, crop_labels = self.generate_crops(
                image, bboxes, labels, history, filename)

            self.save_crops_and_labels(
                crops, crop_bboxes, crop_labels, filename)

        with open(self.augmented_folder / 'crops.txt', 'a') as crops_txt:
            crops_txt.write(
                '\n'.join([' '.join(map(str, crop)) for crop in history]))


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
    parser.add_argument('--crop_file', type=str, default=None,
                        help="File containing pre-generated crop coordinates")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    augmenter = ImageAugmenter(
        args.dir, args.n_crops, args.min_crops, args.min_visibility, args.force, args.crop_file)
    augmenter.process_images()
