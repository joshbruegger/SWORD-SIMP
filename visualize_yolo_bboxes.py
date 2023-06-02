import argparse
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from psd_tools import PSDImage


def visualize_yolo_bboxes(image_path, output_path=None, labels_dir=None):
    # Check if the .txt file exists
    if labels_dir is not None:
        txt_path = os.path.join(labels_dir, os.path.splitext(
            os.path.basename(image_path))[0] + '.txt')
    else:
        txt_path = os.path.splitext(image_path)[0] + '.txt'

    if not os.path.exists(txt_path):
        print(f"No .txt file found for image: {image_path}")
        return

    # Read the image
    if image_path.endswith('.psb') or image_path.endswith('.psd'):
        psd = PSDImage.open(image_path)
        image = cv2.cvtColor(psd[0].numpy(), cv2.COLOR_RGB2BGR)
        image *= 255
    else:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    # Read the bounding boxes
    with open(txt_path, 'r') as f:
        bboxes = f.readlines()

    # Visualize the bounding boxes
    fig, ax = plt.subplots()
    ax.imshow(image)

    for bbox in bboxes:
        class_id, cx, cy, w, h = map(float, bbox.strip().split())

        x = (cx - w/2) * width
        y = (cy - h/2) * height
        w = w * width
        h = h * height

        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y-5, str(int(class_id)), fontsize=12, color='r')

    if output_path is not None:
        dpi = fig.get_dpi()
        fig.set_size_inches(width / dpi, height / dpi)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize YOLO bounding boxes")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("-o", "--output", dest="output_path", default=None,
                        help="Path to save the visualization image or flag to save with default name")
    parser.add_argument("-l", "--labels", dest="labels_dir", default=None,
                        help="Path to the directory containing the labels")
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = os.path.join(os.path.dirname(
            args.image_path), f"visualization_{os.path.basename(args.image_path)}")

    labels_dir = args.labels_dir
    if args.labels_dir is None:
        labels_dir = os.path.dirname(args.image_path)

    visualize_yolo_bboxes(args.image_path, args.output_path, labels_dir)
