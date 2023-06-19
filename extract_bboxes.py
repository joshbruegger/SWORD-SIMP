import os
from psd_tools import PSDImage
import argparse
import gc
import shutil

def get_layer_center(layer):
    bounds = layer.bbox
    docWidth = layer._psd.width
    docHeight = layer._psd.height
    centerX = ((bounds[0] + bounds[2]) / 2) / docWidth
    centerY = ((bounds[1] + bounds[3]) / 2) / docHeight
    width = (bounds[2] - bounds[0]) / docWidth
    height = (bounds[3] - bounds[1]) / docHeight

    return {
        'x': centerX,
        'y': centerY,
        'width': width,
        'height': height
    }


def process_layer(layer, layer_info, parent_folder_name):
    # Ignore background layers
    if not (layer.name.startswith('Sfondo') or layer.name.startswith('Background')):
        layer_center = get_layer_center(layer)
        if (layer_center['width'] > 0 and layer_center['height'] > 0):
            layer_info.append({
                'name': layer.name,
                'folder': parent_folder_name,
                'centerX': layer_center['x'],
                'centerY': layer_center['y'],
                'width': layer_center['width'],
                'height': layer_center['height']
            })

def process_layers(layers, layer_info, parent_folder_name):
    for layer in layers:
        if layer.is_group():
            process_layers(layer, layer_info, layer.name)
        else:
            process_layer(layer, layer_info, parent_folder_name)


def write_bouding_boxes_to_file(layer_info, filename, output):
    file_path = output
    file_name = os.path.basename(filename).split('.')[0]
    os.makedirs(file_path, exist_ok=True)
    file = open(os.path.join(file_path, file_name + '.txt'),
                'w+', encoding='UTF-8')

    for info in layer_info:
        # Skip background layers (no folder)
        if info['folder'] is None:
            continue
        file.write("{} {:.8f} {:.8f} {:.8f} {:.8f}\n".format(info['folder'].replace('?', '').replace('-grandi', ''),
                                                             info['centerX'],
                                                             info['centerY'],
                                                             info['width'],
                                                             info['height']))
    file.close()
    print("Bounding boxes have been saved.".format(file_name))


def write_classes_to_file(layer_info, filename, output):
    unique_folders = set()
    for info in layer_info:
        folder = info['folder'] or 'Root'
        unique_folders.add(folder.replace('?', '').replace('-grandi', ''))

    file_path = output
    file_name = os.path.basename(filename).split('.')[0]
    os.makedirs(file_path, exist_ok=True)
    classes = []
    with open(os.path.join(file_path, file_name + '_classes.txt'), 'w+', encoding='UTF-8') as file:
        for folder in unique_folders:
            c = folder.replace('?', '').replace('-grandi', '')
            file.write("{}\n".format(c))
            classes.append(c)
    print("Class names have been saved.".format(file_name))
    return classes


def process_psd_file(file_path, output_folder):
    psd = PSDImage.open(file_path)
    layer_info = []
    process_layers(psd, layer_info, None)
    write_bouding_boxes_to_file(layer_info, file_path, output_folder)
    c = write_classes_to_file(layer_info, file_path, output_folder)
    # free memory
    del psd
    del layer_info
    gc.collect()
    return c


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str,
                        help='File path to the PSD file or directory with PSD files.')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output folder for the generated text files.')
    parser.add_argument('-f', '--force', action='store_true',
                        help="Force the overwriting of existing files.")
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

    # Check if output folder already exists
    if os.path.exists(output_folder):
        if args.force:
            print("Output folder already exists. Deleting folder...")
            shutil.rmtree(output_folder)
        else:
            # If it exists and force is not set, exit
            print("Output folder already exists. Please delete the folder or use the --force option.")
            exit()

    classes = []
    
    # Check if given path is a directory or a file
    if os.path.isdir(file_path):
        # If it's a directory, process all files in that directory
        for filename in os.listdir(file_path):
            if filename.endswith(".psd") or filename.endswith(".psb"):
                print("Processing {}".format(filename))
                c = process_psd_file(os.path.join(
                    file_path, filename), output_folder)
                classes.extend(c)
    elif file_path.endswith(".psd") or file_path.endswith(".psb"):
        # If it's a file, process the file
        c = process_psd_file(file_path, output_folder)
        classes.extend(c)
    
    # Remove duplicates and sort
    classes = list(set(classes))
    classes.sort()

    # Write classes to file
    with open(os.path.join(os.path.dirname(file_path), 'classes.txt'), 'w+', encoding='UTF-8') as file:
        for c in classes:
            file.write("{}\n".format(c))
