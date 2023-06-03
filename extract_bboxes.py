import os
from psd_tools import PSDImage
import argparse


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
    if not layer.name.startswith('Sfondo') or layer.name.startswith('Background'):
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
        folder = info['folder'] or 'Root'
        file.write("{} {:.8f} {:.8f} {:.8f} {:.8f}\n".format(folder.replace('?', ''),
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
        unique_folders.add(folder.replace('?', ''))

    file_path = output
    file_name = os.path.basename(filename).split('.')[0]
    os.makedirs(file_path, exist_ok=True)
    file = open(os.path.join(file_path, file_name +
                '_classes.txt'), 'w+', encoding='UTF-8')

    for folder in unique_folders:
        file.write("{}\n".format(folder))

    file.close()
    print("Class names have been saved.".format(file_name))


def process_psd_file(file_path, output_folder):
    psd = PSDImage.open(file_path)
    layer_info = []
    process_layers(psd, layer_info, None)
    write_bouding_boxes_to_file(layer_info, file_path, output_folder)
    write_classes_to_file(layer_info, file_path, output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str,
                        help='File path to the PSD file or directory with PSD files.')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output folder for the generated text files.')
    args = parser.parse_args()

    file_path = args.file_path
    output_folder = args.output
    if output_folder is None:
        output_folder = os.path.dirname(file_path)

    # Check if given path is a directory or a file
    if os.path.isdir(file_path):
        # If it's a directory, process all files in that directory
        for filename in os.listdir(file_path):
            if filename.endswith(".psd") or filename.endswith(".psb"):
                print("Processing {}".format(filename))
                process_psd_file(os.path.join(
                    file_path, filename), output_folder)
    elif file_path.endswith(".psd") or file_path.endswith(".psb"):
        # If it's a file, process the file
        process_psd_file(file_path, output_folder)
