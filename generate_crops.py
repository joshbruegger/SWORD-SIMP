import argparse
import cv2
import time
import os
import gc
import shutil
from joblib import Parallel, delayed
import numpy as np
import albumentations as A
from pathlib import Path
from psd_tools import PSDImage


class ImageCropper:
    def __init__(self, dir, n_crops, min_crops=1, min_visibility=0.1, force=False, crop_file=None, processes=1):
        self.dir = Path(dir)
        self.n_crops = n_crops
        self.min_crops = min_crops
        self.min_visibility = min_visibility
        self.augmented_folder = self.dir / 'cropped'
        self.images_dir = self.dir / 'images'
        self.labels_dir = self.dir / 'labels'
        self.output_images_dir = self.augmented_folder / 'images'
        self.output_labels_dir = self.augmented_folder / 'labels'
        self.crop_file = crop_file
        self.processes = processes
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

        if self.augmented_folder.exists():
            if not force:
                print("Cropped folder already exists, use --force to overwrite")
                exit()
            else:
                print("Removing existing cropped folder")
                shutil.rmtree(self.augmented_folder)

        self.output_images_dir.mkdir(parents=True, exist_ok=True)
        self.output_labels_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_random_crop_coordinates(image_width, image_height, min_size=1080, max_size=1080, factor=32):
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
            labels = [line[0] for line in labels_and_bboxes]
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
            print("Regenerating crops for", filename.stem)
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
                raise RuntimeError("No crop coordinates found for image")

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

    def process_image(self, filename):
        start_time = time.time()
        print(f'Processing {filename}')
        
        if filename.suffix not in ['.psb', '.psd', '.jpg', '.png']:
            print(f'Unsupported file format: {filename}')
            return []

        bboxes, labels = self.read_labels(filename)

        if filename.suffix in ['.jpg', '.png']:
            image = cv2.imread(str(filename))
        else:
            psd = PSDImage.open(filename)
            image = cv2.cvtColor(psd[0].numpy(), cv2.COLOR_RGB2BGR) * 255
            del psd
            gc.collect()

        history = []
        crops, crop_bboxes, crop_labels = self.generate_crops(
            image, bboxes, labels, history, filename)
        
        del image
        gc.collect()

        self.save_crops_and_labels(
            crops, crop_bboxes, crop_labels, filename)

        print(f'Processed {filename.stem} in {time.time() - start_time:.2f}s')
        return history

    def process_images(self):
        print(f'Processing images using {self.processes} processes')

        histories = Parallel(n_jobs=self.processes)(delayed(self.process_image)(
            filename) for filename in self.images_dir.iterdir())

        with open(self.augmented_folder / 'crops.txt', 'a') as crops_txt:
            for history in histories:
                crops_txt.write('\n'.join([' '.join(map(str, crop)) for crop in history]))
                crops_txt.write('\n')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='Working directory')
    parser.add_argument('n_crops', type=int, help='Number of crops per image')
    parser.add_argument('-c', '--min_crops', type=int, default=1,
                        help='Minimum number of bounding boxes per crop')
    parser.add_argument('-v', '--min_visibility', type=float, default=0.1,
                        help='Minimum visibility for bounding boxes')
    parser.add_argument('-f', '--force', action='store_true',
                        help="Force the removal of the existing 'cropped' directory if it exists")
    parser.add_argument('-r', '--crop_file', type=str, default=None,
                        help="File containing pre-generated crop coordinates")
    parser.add_argument('-p', '--processes', type=int, default=None, help='Number of processes to use for multiprocessing')
    return parser.parse_args()


if __name__ == '__main__':
    
    args = get_args()
    
    # if the processes flag has been set, use that number of processes. otherwise, see if int(os.environ['SLURM_JOB_CPUS_PER_NODE']) is set, and use that number of processes. otherwise, use just 1 process.
    if args.processes:
        n_jobs = args.processes
    elif 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
        n_jobs = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    else:
        n_jobs = 1
        
    # PRETTY PRINT WELCOME MESSAGE & ARGUMENTS.
    padding = 140
    print("\n\n")
    print(" SWORD-SIMP Crop Generator ".center(padding, "8"))
    print(f" Working directory: {args.dir} ".center(padding))
    print(f" Number of crops per image: {args.n_crops} ".center(padding))
    print(f" Minimum number of bounding boxes per crop: {args.min_crops} ".center(padding))
    print(f" Minimum visibility for bounding boxes: {args.min_visibility} ".center(padding))
    print(f" Force the removal of the existing 'cropped' directory if it exists: {args.force} ".center(padding))
    print(f" File containing pre-generated crop coordinates: {args.crop_file} ".center(padding))
    print(f" Number of processes to use for multiprocessing: {args.processes} ".center(padding))
    print("".center(padding, "8"))
    print("\n\n")


    augmenter = ImageCropper(
        args.dir, args.n_crops, args.min_crops, args.min_visibility, args.force, args.crop_file, n_jobs)

    start_time = time.time()
    augmenter.process_images()
    end_time = time.time()

    duration = end_time - start_time
    if (duration) > 3600:
        print(f'Finished in {(duration)/3600:.2f} hours')
    elif (duration) > 60:
        print(f'Finished in {(duration)/60:.2f} minutes')
    else:
        print(f'Finished in {duration:.2f} seconds')
