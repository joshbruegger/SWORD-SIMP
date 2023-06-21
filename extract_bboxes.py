import os
import argparse
import gc
import shutil
from psd_tools import PSDImage
from natsort import os_sorted

class PSDProcessor:
    __slots__ = ['file_path', 'output_folder', 'classes', 'paintings', 'painting_sizes']

    def __init__(self, file_path, output_folder):
        self.file_path = file_path
        self.output_folder = output_folder
        self.classes = set()
        self.paintings = {} # Dictionary with filename as key and list of bounding boxes as value
        self.painting_sizes = {} # Dictionary with filename as key and [width, height] as value

        os.makedirs(self.output_folder, exist_ok=True) # Create output folder if it doesn't exist


    @staticmethod
    def clean_class_name(folder):
        return folder.replace('?', '').replace('-grandi', '')

    @staticmethod
    def get_layer_center(layer):
        bounds = layer.bbox
        doc_width, doc_height = layer._psd.size
        center_x = ((bounds[0] + bounds[2]) / 2) / doc_width
        center_y = ((bounds[1] + bounds[3]) / 2) / doc_height
        width = (bounds[2] - bounds[0]) / doc_width
        height = (bounds[3] - bounds[1]) / doc_height

        return center_x, center_y, width, height

    def process_layer(self, layer, bboxes, parent_folder_name):
        # Ignore background layers and groups
        if layer.name.startswith(('Sfondo', 'Background')) or layer.is_group():
            return

        center_x, center_y, width, height = self.get_layer_center(layer)

        # Ignore layers with no width or height
        if width == 0 or height == 0:
            return

        if not parent_folder_name:
            print(f"Layer {layer.name} has no parent folder!!")
            return
        
        class_name = self.clean_class_name(parent_folder_name)
        self.classes.add(class_name)
        bboxes.append({
            'class': class_name,
            'centerX': center_x,
            'centerY': center_y,
            'width': width,
            'height': height
        })

    # Recursively process layers
    def process_layers(self, layers, bboxes, parent_folder_name):
        for layer in layers:
            if layer.is_group():
                self.process_layers(layer, bboxes, layer.name)
                print(f"Processing folder {layer.name}")
            else:
                self.process_layer(layer, bboxes, parent_folder_name)

    def write_bounding_boxes_to_file(self, bboxes, filename):
        file_path = os.path.join(self.output_folder, f"{os.path.basename(filename).split('.')[0]}.txt")

        with open(file_path, 'w+', encoding='UTF-8') as file:
            for bbox in bboxes:
                if not bbox['class']:
                    continue
                
                line = f"{bbox['class']} {bbox['centerX']:.17f} {bbox['centerY']:.17f} {bbox['width']:.17f} {bbox['height']:.17f}\n"
                file.write(line)

        print(f"Bounding boxes for {filename} have been saved.")

    def process_psd_file(self, filename):
        print(f"Processing {filename}...")
        
        psd = PSDImage.open(filename)
        bboxes = []
        self.process_layers(psd, bboxes,  None)
        self.paintings[filename] = bboxes
        self.painting_sizes[filename] = psd.size

        del psd
        gc.collect()

    def process_files(self):
        # Process all PSD files in the folder
        for filename in os.listdir(self.file_path):
            if filename.endswith((".psd", ".psb")): # Only process PSD files
                self.process_psd_file(os.path.join(self.file_path, filename))
        
        self.classes = os_sorted(self.classes)

        # Write classes to file
        for filename, bboxes in self.paintings.items():
            # Convert class names to class indices based on the sorted classes list
            for bbox in bboxes:
                bbox['class'] = self.classes.index(bbox['class'])

            self.write_bounding_boxes_to_file(bboxes, filename)

        with open(os.path.join(os.path.dirname(self.file_path), 'classes.txt'), 'w+', encoding='UTF-8') as file:
            file.writelines(f"{c}\n" for c in self.classes)

        with open(os.path.join(os.path.dirname(self.file_path), 'painting_sizes.txt'), 'w+', encoding='UTF-8') as file:
            for filename, size in self.painting_sizes.items():
                file.write(f"{os.path.basename(filename).split('.')[0]} {size[0]} {size[1]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help='Path to the PSD file or directory with PSD files.')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output folder for the generated text files.')
    parser.add_argument('-f', '--force', action='store_true', help="Force the overwriting of existing files.")
    args = parser.parse_args()

    # PRETTY PRINT WELCOME MESSAGE & ARGUMENTS.
    padding = 140
    print("\n\n")
    print(" SWORD-SIMP Bounding Box Extraction Script ".center(padding, "8"))
    print(f" File path: {args.file_path} ".center(padding))
    print(f" Output folder: {args.output} ".center(padding))
    print(f" Force overwrite: {args.force} ".center(padding))
    print("".center(padding, "8"))
    print("\n\n")

    output_folder = args.output if args.output else os.path.dirname(args.file_path)
    if os.path.exists(output_folder):
        if not args.force:
            print("Output folder already exists. Please delete the folder or use the --force option.")
            exit()
        shutil.rmtree(output_folder)

    print("Extracting bounding boxes...")
    PSDProcessor(args.file_path, output_folder).process_files()
