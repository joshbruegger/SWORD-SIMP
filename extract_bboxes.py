import os
from psd_tools import PSDImage
import numpy as np


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


def write_bouding_boxes_to_file(layer_info, filename):
    file_path = os.path.dirname(filename)
    file_name = os.path.basename(filename).split('.')[0]
    file = open(os.path.join(file_path, file_name + '.txt'),
                'w', encoding='UTF-8')

    for info in layer_info:
        folder = info['folder'] or 'Root'
        file.write("{} {:.8f} {:.8f} {:.8f} {:.8f}\n".format(folder.replace('?', ''),
                                                             info['centerX'],
                                                             info['centerY'],
                                                             info['width'],
                                                             info['height']))
    file.close()
    print("Bounding boxes have been saved to {}.txt in the same folder as the open file.".format(file_name))


def write_classes_to_file(layer_info, filename):
    unique_folders = set()
    for info in layer_info:
        folder = info['folder'] or 'Root'
        unique_folders.add(folder.replace('?', ''))

    file_path = os.path.dirname(filename)
    file_name = os.path.basename(filename).split('.')[0]
    file = open(os.path.join(file_path, file_name +
                '_classes.txt'), 'w', encoding='UTF-8')

    for folder in unique_folders:
        file.write("{}\n".format(folder))

    file.close()
    print("Class names have been saved to {}_classes.txt in the same folder as the open file.".format(file_name))


if __name__ == "__main__":
    file_path = './dataset/all/images/fatto_unito.psb'
    psd = PSDImage.open(file_path)
    layer_info = []
    process_layers(psd, layer_info, None)

    write_bouding_boxes_to_file(layer_info, file_path)
    write_classes_to_file(layer_info, file_path)
